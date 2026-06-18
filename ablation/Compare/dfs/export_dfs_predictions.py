#!/usr/bin/env python3
"""Eval-only prediction export for DFS baseline.

Purpose:
  - Load an already trained DFS checkpoint (best.pt/last.pt)
  - Rebuild the fixed clean/masked test loaders
  - Save per-clip prediction probability CSVs for ROC/PR/AUROC/AUPRC

Expected output files under run_dir:
  test_clean_predictions.csv
  test_masked_predictions.csv
  test_clean_action_clip_predictions.csv
  test_masked_action_clip_predictions.csv
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
import yaml
from torch.utils.data import DataLoader


def load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def add_path(p: str | Path):
    p = Path(p).resolve()
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    return p


def first_existing(paths: list[Path | None]) -> Path | None:
    for p in paths:
        if p is not None and Path(p).exists():
            return Path(p)
    return None


def to_device(x: Any, device: str):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    return x


def get_any(batch: Any, names: list[str], default=None):
    if isinstance(batch, dict):
        for n in names:
            if n in batch:
                return batch[n]
    return default


def as_list(x, n: int | None = None):
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


def extract_batch(batch: Any, device: str):
    """Extract tensors/labels from common MemoryMultitaskDataset batch formats."""
    # Most likely project format: dict
    if isinstance(batch, dict):
        x_body = get_any(batch, ["body", "pose", "x_body", "x_pose", "pose_tensor", "body_tensor"])
        x_face = get_any(batch, ["face", "x_face", "facemesh", "face_tensor"])
        x_head_pose = get_any(batch, ["head_pose", "x_head_pose", "headpose", "head_pose_tensor"], None)

        labels = {
            "action": get_any(batch, ["y_action", "action", "label_action", "action_label"]),
            "gaze": get_any(batch, ["y_gaze_fine", "y_gaze", "gaze", "label_gaze", "gaze_label"]),
            "hands": get_any(batch, ["y_hands", "hands", "label_hands", "hands_label"]),
            "talk": get_any(batch, ["y_talk", "talk", "label_talk", "talk_label"]),
        }
        clip_ids = get_any(batch, ["clip_id", "clip_ids", "id", "ids", "sample_id", "sample_ids"], None)
        window_ids = get_any(batch, ["window_id", "window_ids", "index", "indices"], None)

        if x_body is None or x_face is None:
            raise KeyError(f"Cannot find body/face tensors in batch keys: {list(batch.keys())}")

        x_body = to_device(x_body, device)
        x_face = to_device(x_face, device)
        x_head_pose = to_device(x_head_pose, device) if x_head_pose is not None else None
        labels = {k: to_device(v, device) if v is not None else None for k, v in labels.items()}
        return x_body, x_face, x_head_pose, labels, clip_ids, window_ids

    # Fallback tuple patterns. If this fails, print batch structure for debugging.
    if isinstance(batch, (list, tuple)):
        if len(batch) >= 3:
            x_body, x_face = batch[0], batch[1]
            x_head_pose = None
            label_obj = batch[2]
            extra_start = 3
            if len(batch) >= 4 and torch.is_tensor(batch[2]) and torch.is_tensor(batch[3]):
                # Ambiguous, but keep simple. Common tuple usually not used here.
                pass
            labels = {}
            if isinstance(label_obj, dict):
                labels = {
                    "action": get_any(label_obj, ["y_action", "action", "label_action"]),
                    "gaze": get_any(label_obj, ["y_gaze_fine", "y_gaze", "gaze", "label_gaze"]),
                    "hands": get_any(label_obj, ["y_hands", "hands", "label_hands"]),
                    "talk": get_any(label_obj, ["y_talk", "talk", "label_talk"]),
                }
            else:
                raise TypeError("Tuple batch detected but label object is not dict; add a custom extractor.")
            clip_ids = batch[extra_start] if len(batch) > extra_start else None
            return to_device(x_body, device), to_device(x_face, device), x_head_pose, labels, clip_ids, None

    raise TypeError(f"Unsupported batch type: {type(batch)}")


def aggregate_clip_probs(items: list[dict], num_classes: int, mode: str = "topk_mean", topk: int = 3):
    """Aggregate window probabilities into one clip-level probability vector."""
    probs = np.stack([x["prob"] for x in items], axis=0)  # [W, C]
    if mode == "topk_mean":
        k = max(1, min(int(topk or 1), probs.shape[0]))
        vals = []
        for c in range(num_classes):
            col = np.sort(probs[:, c])[-k:]
            vals.append(float(np.mean(col)))
        score = np.asarray(vals, dtype=float)
    elif mode in ("mean", "avg", "average"):
        score = probs.mean(axis=0)
    elif mode == "max":
        score = probs.max(axis=0)
    else:
        score = probs.mean(axis=0)

    # Normalize again for safer ROC/PR input.
    s = float(score.sum())
    if np.isfinite(s) and s > 0:
        score = score / s
    else:
        score = np.ones(num_classes, dtype=float) / num_classes
    return score


def export_split_predictions(model, loader, split_name: str, cfg: dict, shared: dict, run_dir: Path, device: str):
    model.eval()
    head_names = list(shared["HEAD_NAMES"])
    num_classes = {
        "action": int(shared["NUM_ACTION_CLASSES"]),
        "gaze": int(shared["NUM_GAZE_ZONES"]),
        "hands": int(shared["NUM_HANDS_CLASSES"]),
        "talk": int(shared["NUM_TALK_CLASSES"]),
    }
    agg_mode = cfg.get("eval", {}).get("clip_agg_mode", "topk_mean")
    topk = int(cfg.get("eval", {}).get("clip_topk", 3) or 3)

    # head -> clip_id -> list(window dict)
    bucket: dict[str, dict[str, list[dict]]] = {h: {} for h in head_names}
    n_windows = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x_body, x_face, x_head_pose, labels, clip_ids, window_ids = extract_batch(batch, device)
            outputs = model(x_body, x_face, x_head_pose) if x_head_pose is not None else model(x_body, x_face)
            batch_size = x_body.shape[0]
            clip_id_list = as_list(clip_ids, batch_size)
            window_id_list = as_list(window_ids, batch_size)

            for head in head_names:
                if head not in outputs or labels.get(head) is None:
                    continue
                logits = outputs[head]
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                y = labels[head].detach().cpu().numpy().astype(int)
                cnum = num_classes[head]
                for i in range(batch_size):
                    yi = int(y[i])
                    if yi < 0 or yi >= cnum:
                        continue
                    cid = clip_id_list[i]
                    if cid is None:
                        cid = f"{split_name}_batch{batch_idx:06d}_idx{i:03d}"
                    cid = str(cid)
                    wid = window_id_list[i]
                    bucket[head].setdefault(cid, []).append({
                        "y_true": yi,
                        "prob": probs[i, :cnum].astype(float),
                        "window_id": wid,
                    })
            n_windows += int(batch_size)

    all_rows = []
    for head in head_names:
        rows = []
        cnum = num_classes[head]
        for clip_id, items in sorted(bucket[head].items()):
            if not items:
                continue
            # Use majority/first label. Labels should be identical within clip.
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--dfs-root", type=Path, default=Path.cwd(), help="Directory containing dfs_multitask.py and train_dfs_multitask.py")
    ap.add_argument("--run-dir", type=Path, required=True, help="Run directory containing best.pt/config.json")
    ap.add_argument("--config", type=Path, default=None, help="YAML/JSON config. Default: run_dir/config.json")
    ap.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint. Default: run_dir/best.pt then last.pt")
    ap.add_argument("--splits", nargs="+", default=["test_clean", "test_masked"])
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    dfs_root = add_path(args.dfs_root)
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    import train_dfs_multitask as td
    from dfs_multitask import DFSDMDMultitaskClassifier

    cfg_path = args.config or first_existing([run_dir / "config.json", dfs_root / "experiments" / "dfs_fixed_clean_masked_seed42.yaml"])
    if cfg_path is None:
        raise FileNotFoundError("No config found. Pass --config explicitly.")
    cfg = load_config(cfg_path)
    cfg.setdefault("paths", {})["save_root"] = str(run_dir)

    ckpt_path = args.checkpoint or first_existing([run_dir / "best.pt", run_dir / "last.pt"])
    if ckpt_path is None:
        raise FileNotFoundError(f"checkpoint not found under {run_dir}")

    classification_root = td._add_classification_root(cfg["paths"].get("classification_root", td.DEFAULT_CLASSIFICATION_ROOT))
    shared = td._import_shared_modules()
    shared["set_seed"](int(cfg.get("seed", 42)))
    device = cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    print("dfs_root:", dfs_root)
    print("run_dir:", run_dir)
    print("config:", cfg_path)
    print("checkpoint:", ckpt_path)
    print("classification_root:", classification_root)
    print("device:", device)

    # Build/load clips and fixed clean/masked split, but only preload test splits.
    log = shared["get_logger"]("dfs_export_predictions", log_file=run_dir / "export_predictions.log")
    clips = td.build_or_load_clips(cfg, shared, log)
    if (cfg.get("fixed_split") or {}).get("enabled", False):
        train_clips, val_clips, test_clip_sets, split_info = td.build_fixed_split_clips(cfg, clips, shared, log)
    else:
        split_path = Path(cfg["paths"].get("split_info_path", "")) if cfg["paths"].get("split_info_path") else run_dir / "split_info.json"
        split_info = shared["load_split_info"](split_path)
        train_clips, val_clips, test_clips = shared["filter_by_split_info"](clips, split_info)
        test_clip_sets = {"test": test_clips}

    face_cfg = cfg["face"]
    face_mode = face_cfg["mode"]
    use_head_pose = bool(cfg.get("baseline", {}).get("use_head_pose", False))
    common = dict(
        window_size=cfg["window"]["size"],
        window_stride=cfg["window"]["stride"],
        max_windows_per_clip=cfg["window"].get("max_per_clip"),
        pose_min_valid_frames=cfg["window"]["pose_min_valid_frames"],
        pose_min_valid_ratio=cfg["window"]["pose_min_valid_ratio"],
        pose_min_valid_joint_ratio=cfg["window"]["pose_min_valid_joint_ratio"],
        face_min_detected_ratio=cfg["window"]["face_min_detected_ratio"],
        joint_conf_thres=cfg["pose"]["joint_conf_thres"],
        face_mode=face_mode,
        face_use_z=face_cfg.get("use_z", True),
        face_use_detected_channel=face_cfg.get("use_detected_channel", True),
        face_use_det_score_channel=face_cfg.get("use_det_score_channel", True),
        face_bbox_det_thres=face_cfg.get("bbox_det_thres", 0.25),
        use_head_pose=use_head_pose,
        logger=log,
    )

    bs = int(args.batch_size or cfg["train"].get("batch_size", 128))
    pin = device != "cpu"
    test_loaders = {}
    for name in args.splits:
        if name not in test_clip_sets:
            print(f"[skip split] {name}: not in test_clip_sets={list(test_clip_sets)}")
            continue
        items = shared["preload_multitask_windows"](test_clip_sets[name], desc=f"preload {name}", **common)
        ds = shared["MemoryMultitaskDataset"](items)
        test_loaders[name] = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
        print(f"[loader] {name}: clips={len(test_clip_sets[name])} windows={len(items)}")

    # Infer channels exactly like training script.
    pose_in_ch = 2 + (2 if cfg["pose"].get("use_bone") else 0) + (2 if cfg["pose"].get("use_velocity") else 0) + (1 if cfg["pose"].get("use_conf_channel") else 0)
    if face_mode in ("facemesh", "facemesh_full"):
        face_in_ch = (3 if face_cfg.get("use_z", True) else 2) + (1 if face_cfg.get("use_detected_channel", True) else 0)
        num_face_regions = face_cfg.get("num_landmarks", 478) if face_mode == "facemesh_full" else face_cfg.get("num_regions", 10)
    else:
        face_in_ch = 2 + (1 if face_cfg.get("use_detected_channel", True) else 0) + (1 if face_cfg.get("use_det_score_channel", True) else 0)
        num_face_regions = face_cfg.get("num_regions", 5)

    bcfg = cfg["baseline"]
    model = DFSDMDMultitaskClassifier(
        pose_in_channels=pose_in_ch,
        face_in_channels=face_in_ch,
        num_pose_joints=cfg["pose"]["num_joints"],
        num_face_regions=num_face_regions,
        proj_dim=bcfg["proj_dim"],
        stream_dim=bcfg["stream_dim"],
        fusion_dim=bcfg["fusion_dim"],
        dropout=bcfg["dropout"],
        num_action=shared["NUM_ACTION_CLASSES"],
        num_gaze=shared["NUM_GAZE_ZONES"],
        num_hands=shared["NUM_HANDS_CLASSES"],
        num_talk=shared["NUM_TALK_CLASSES"],
        fusion_kind=bcfg.get("fusion_kind", "late_fusion"),
        use_head_pose=use_head_pose,
        head_pose_in_channels=2,
        num_head_pose_axes=3,
        score_fusion_body_weight=bcfg.get("score_fusion_body_weight", 1.0),
        score_fusion_face_weight=bcfg.get("score_fusion_face_weight", 1.0),
        score_fusion_head_pose_weight=bcfg.get("score_fusion_head_pose_weight", 0.5),
    ).to(device)

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ck.get("model_state_dict", ck)
    model.load_state_dict(state)
    print("[loaded checkpoint]")

    for split_name, loader in test_loaders.items():
        export_split_predictions(model, loader, split_name, cfg, shared, run_dir, device)

    print("[done] prediction CSV export complete")


if __name__ == "__main__":
    main()
