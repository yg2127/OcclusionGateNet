#!/usr/bin/env python3
"""Eval-only prediction export for Compare_* DMS baselines.

Supports these project variants because they share the same V5-style structure:
  - Compare_Pose-guided Multi-task
  - Compare_SkateFormer
  - Compare_Spatiotemporal

Purpose:
  - Load an already trained checkpoint from artifacts/results
  - Rebuild test_clean/test_masked loaders from the original config
  - Export per-clip probability CSVs for ROC/PR/AUROC/AUPRC

Output files under run_dir:
  test_clean_predictions.csv
  test_masked_predictions.csv
  test_clean_action_clip_predictions.csv
  test_clean_gaze_clip_predictions.csv
  ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def add_path(p: str | Path) -> Path:
    p = Path(p).resolve()
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    return p


def load_cfg(path: Path) -> dict:
    import yaml
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def first_existing(paths: list[Path | None]) -> Path | None:
    for p in paths:
        if p is not None and Path(p).exists():
            return Path(p)
    return None


def load_state_dict(path: Path, device: str) -> dict:
    ck = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ck, dict) and "model_state_dict" in ck:
        state = ck["model_state_dict"]
    elif isinstance(ck, dict) and "state_dict" in ck:
        state = ck["state_dict"]
    elif isinstance(ck, dict) and "model" in ck:
        state = ck["model"]
    else:
        state = ck

    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(ck)}")

    cleaned = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k.replace("module.", "", 1)
        cleaned[k] = v
    return cleaned


def as_list(x: Any, n: int | None = None):
    if x is None:
        return [None] * int(n or 0)
    if isinstance(x, (list, tuple)):
        return list(x)
    if torch.is_tensor(x):
        if x.ndim == 0:
            return [x.item()]
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return [x] * int(n or 1)


def aggregate_clip_probs(items: list[dict], num_classes: int, mode: str = "topk_mean", topk: int = 3):
    probs = np.stack([x["prob"] for x in items], axis=0)

    if mode == "topk_mean":
        k = max(1, min(int(topk or 1), probs.shape[0]))
        score = np.zeros((num_classes,), dtype=np.float32)
        for c in range(num_classes):
            score[c] = float(np.mean(np.sort(probs[:, c])[-k:]))
    elif mode in ("mean", "avg", "average"):
        score = probs.mean(axis=0)
    elif mode == "max":
        score = probs.max(axis=0)
    else:
        score = probs.mean(axis=0)

    s = float(score.sum())
    if np.isfinite(s) and s > 0:
        score = score / s
    else:
        score = np.ones((num_classes,), dtype=np.float32) / num_classes
    return score.astype(float)


def export_split_predictions(model, loader, split_name: str, cfg: dict, run_dir: Path, device: str):
    from src.training.loops import HEAD_NAMES, IGNORE_LABEL
    from src.data.clip_builder import NUM_ACTION_CLASSES, NUM_HANDS_CLASSES, NUM_TALK_CLASSES
    from constants.gaze_zones import NUM_GAZE_ZONES

    head_names = list(HEAD_NAMES)
    num_classes = {
        "action": int(NUM_ACTION_CLASSES),
        "gaze": int(NUM_GAZE_ZONES),
        "hands": int(NUM_HANDS_CLASSES),
        "talk": int(NUM_TALK_CLASSES),
    }

    agg_mode = cfg.get("eval", {}).get("clip_agg_mode", cfg.get("eval", {}).get("agg_mode", "topk_mean"))
    topk = int(cfg.get("eval", {}).get("clip_topk", cfg.get("eval", {}).get("topk", 3)) or 3)

    model.eval()
    bucket: dict[str, dict[str, list[dict]]] = {h: {} for h in head_names}
    n_windows = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            xb = batch["x_body"].to(device, non_blocking=True)
            xf = batch["x_face"].to(device, non_blocking=True)
            xocc = batch.get("x_occ", None)
            if xocc is not None:
                xocc = xocc.to(device, non_blocking=True)

            logits = model(xb, xf, x_occ=xocc)
            bs = int(xb.shape[0])
            clip_ids = as_list(batch.get("clip_id", None), bs)

            for head, logit_name, tgt_key in (
                ("action", "action", "y_action"),
                ("gaze", "gaze", "y_gaze_fine"),
                ("hands", "hands", "y_hands"),
                ("talk", "talk", "y_talk"),
            ):
                if logit_name not in logits or tgt_key not in batch:
                    continue

                prob = torch.softmax(logits[logit_name], dim=1).detach().cpu().numpy()
                y = batch[tgt_key].detach().cpu().numpy().astype(int)
                cnum = num_classes[head]

                for i in range(bs):
                    yi = int(y[i])
                    if yi == int(IGNORE_LABEL) or yi < 0 or yi >= cnum:
                        continue

                    cid = clip_ids[i]
                    if cid is None:
                        cid = f"{split_name}_batch{batch_idx:06d}_idx{i:03d}"
                    cid = str(cid)

                    bucket[head].setdefault(cid, []).append({
                        "y_true": yi,
                        "prob": prob[i, :cnum].astype(float),
                    })

            n_windows += bs

    all_rows = []
    for head in head_names:
        cnum = num_classes[head]
        rows = []

        for clip_id, items in sorted(bucket[head].items()):
            if not items:
                continue

            labels_here = [int(x["y_true"]) for x in items]
            y_true = max(set(labels_here), key=labels_here.count)
            score = aggregate_clip_probs(items, cnum, mode=agg_mode, topk=topk)
            y_pred = int(np.argmax(score))

            row = {
                "sample_id": clip_id,
                "clip_id": clip_id,
                "split": split_name,
                "head": head,
                "level": "clip",
                "n_windows": len(items),
                "y_true": int(y_true),
                "y_pred": int(y_pred),
            }
            for c in range(cnum):
                row[f"prob_{c}"] = float(score[c])

            rows.append(row)
            all_rows.append(row)

        out_head = run_dir / f"{split_name}_{head}_clip_predictions.csv"
        pd.DataFrame(rows).to_csv(out_head, index=False, encoding="utf-8-sig")
        print(f"[saved] {out_head} rows={len(rows)}")

    out_all = run_dir / f"{split_name}_predictions.csv"
    pd.DataFrame(all_rows).to_csv(out_all, index=False, encoding="utf-8-sig")
    print(f"[saved] {out_all} rows={len(all_rows)} windows_seen={n_windows}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True,
                        help="Root of Compare_* project. Must contain src/, configs/, artifacts/ or results/.")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Run directory containing best.pt/last.pt/config.json.")
    parser.add_argument("--config", type=Path, default=None,
                        help="Config file. Default: run_dir/config.json.")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Checkpoint. Default: run_dir/best.pt then last.pt.")
    parser.add_argument("--splits", nargs="+", default=["test_clean", "test_masked"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    project_root = add_path(args.project_root)
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    from src.training.builders import build_clip_splits, build_loaders, build_model
    from src.utils.io import load_yaml
    from src.utils.logging import get_logger
    from src.utils.seed import set_seed

    cfg_path = args.config or first_existing([run_dir / "config.json"])
    if cfg_path is None:
        raise FileNotFoundError(f"No config found under {run_dir}. Pass --config explicitly.")

    if cfg_path.suffix.lower() == ".json":
        cfg = load_cfg(cfg_path)
    else:
        cfg = load_yaml(str(cfg_path))

    cfg.setdefault("paths", {})["save_root"] = str(run_dir)

    if args.batch_size is not None:
        cfg.setdefault("train", {})["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        cfg.setdefault("train", {})["num_workers"] = int(args.num_workers)

    ckpt_path = args.checkpoint or first_existing([run_dir / "best.pt", run_dir / "last.pt"])
    if ckpt_path is None:
        raise FileNotFoundError(f"checkpoint not found under {run_dir}")

    device = cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"

    print("project_root:", project_root)
    print("run_dir:", run_dir)
    print("config:", cfg_path)
    print("checkpoint:", ckpt_path)
    print("device:", device)

    set_seed(int(cfg.get("seed", 42)))
    log = get_logger("compare_export_predictions", log_file=run_dir / "export_predictions.log")

    # Rebuild clips/loaders exactly like training.
    train_clips, val_clips, test_eval_clips = build_clip_splits(cfg, run_dir, log)
    _, _, test_eval_loaders, _, _, _ = build_loaders(
        cfg, train_clips, val_clips, test_eval_clips, device, log
    )

    model, meta = build_model(cfg, device)
    print("[model meta]", meta)

    state = load_state_dict(ckpt_path, device)
    model.load_state_dict(state, strict=True)
    print("[loaded checkpoint]")

    for split_name in args.splits:
        if split_name not in test_eval_loaders:
            print(f"[skip split] {split_name}: available={list(test_eval_loaders.keys())}")
            continue
        export_split_predictions(model, test_eval_loaders[split_name], split_name, cfg, run_dir, device)

    print("[done] prediction CSV export complete")


if __name__ == "__main__":
    main()
