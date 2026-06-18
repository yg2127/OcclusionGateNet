from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from constants.gaze_zones import NUM_GAZE_ZONES
from src.data.clip_builder import NUM_ACTION_CLASSES, NUM_HANDS_CLASSES, NUM_TALK_CLASSES
from src.data.dataset import IGNORE_LABEL, MemoryMultitaskDataset, preload_multitask_windows
from src.training.aggregation import aggregate_clip_level
from src.training.builders import build_clip_splits, build_model
from src.training.loops import HEAD_NAMES
from src.utils.io import ensure_dir, load_yaml, save_json
from src.utils.logging import get_logger
from src.utils.seed import set_seed

TARGET_KEYS = {
    "action": "y_action",
    "gaze": "y_gaze_fine",
    "hands": "y_hands",
    "talk": "y_talk",
}

NUM_CLASSES = {
    "action": NUM_ACTION_CLASSES,
    "gaze": NUM_GAZE_ZONES,
    "hands": NUM_HANDS_CLASSES,
    "talk": NUM_TALK_CLASSES,
}


def _device_from_cfg(cfg: dict) -> str:
    dev = str(cfg.get("device", "cuda"))
    if dev == "cpu":
        return "cpu"
    return dev if torch.cuda.is_available() else "cpu"


def _common_preload_kwargs(cfg: dict, log):
    face_cfg = cfg["face"]
    return dict(
        window_size=cfg["window"]["size"],
        window_stride=cfg["window"]["stride"],
        max_windows_per_clip=cfg["window"].get("max_per_clip"),
        pose_min_valid_frames=cfg["window"]["pose_min_valid_frames"],
        pose_min_valid_ratio=cfg["window"]["pose_min_valid_ratio"],
        pose_min_valid_joint_ratio=cfg["window"]["pose_min_valid_joint_ratio"],
        face_min_detected_ratio=cfg["window"]["face_min_detected_ratio"],
        joint_conf_thres=cfg["pose"]["joint_conf_thres"],
        face_mode=face_cfg["mode"],
        face_use_z=face_cfg.get("use_z", True),
        face_use_detected_channel=face_cfg.get("use_detected_channel", True),
        face_use_det_score_channel=face_cfg.get("use_det_score_channel", True),
        face_bbox_det_thres=face_cfg.get("bbox_det_thres", 0.25),
        occ_cfg=cfg.get("occ", {}),
        logger=log,
    )


def _save_confusion(cm: np.ndarray, labels: list[str], path: Path) -> None:
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(path, encoding="utf-8-sig")


def _safe_metrics(targets: Iterable[int], preds: Iterable[int]) -> dict:
    t = np.asarray(list(targets), dtype=int)
    p = np.asarray(list(preds), dtype=int)
    mask = t != IGNORE_LABEL
    t = t[mask]
    p = p[mask]
    if len(t) == 0:
        return {"n": 0, "accuracy": 0.0, "precision_macro": 0.0, "recall_macro": 0.0, "f1_macro": 0.0}
    precision, recall, f1, _ = precision_recall_fscore_support(
        t, p, average="macro", zero_division=0
    )
    return {
        "n": int(len(t)),
        "accuracy": float(accuracy_score(t, p)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def _prediction_rows_from_clip_result(split_name: str, head: str, clip_res: dict) -> list[dict]:
    rows = []
    for clip_id, y_true, y_pred, prob in zip(
        clip_res["clip_ids"], clip_res["targets"], clip_res["preds"], clip_res["probs"]
    ):
        row = {
            "sample_id": str(clip_id),
            "clip_id": str(clip_id),
            "split": split_name,
            "head": head,
            "level": "clip",
            "y_true": int(y_true),
            "y_pred": int(y_pred),
        }
        prob = np.asarray(prob, dtype=float)
        for c, v in enumerate(prob):
            row[f"prob_{c}"] = float(v)
        rows.append(row)
    return rows


def _prediction_rows_from_window_state(split_name: str, head: str, state: dict) -> list[dict]:
    rows = []
    for idx, (clip_id, window_idx, y_true, y_pred, prob) in enumerate(
        zip(state["clip_ids"], state["window_idxs"], state["targets"], state["preds"], state["probs"])
    ):
        if int(y_true) == IGNORE_LABEL:
            continue
        row = {
            "sample_id": f"{clip_id}__w{int(window_idx)}",
            "clip_id": str(clip_id),
            "window_idx": int(window_idx),
            "split": split_name,
            "head": head,
            "level": "window",
            "y_true": int(y_true),
            "y_pred": int(y_pred),
        }
        prob = np.asarray(prob, dtype=float)
        for c, v in enumerate(prob):
            row[f"prob_{c}"] = float(v)
        rows.append(row)
    return rows


@torch.no_grad()
def evaluate_loader_and_save_predictions(
    *,
    model,
    loader,
    split_name: str,
    save_root: Path,
    device: str,
    cfg: dict,
    log,
    save_window_predictions: bool = True,
) -> dict:
    model.eval()
    head_state = {
        h: {"targets": [], "preds": [], "probs": [], "clip_ids": [], "window_idxs": []}
        for h in HEAD_NAMES
    }

    pbar = tqdm(loader, desc=f"eval {split_name}", ncols=120, ascii=True, mininterval=0.3)
    for batch in pbar:
        xb = batch["x_body"].to(device, non_blocking=True)
        xf = batch["x_face"].to(device, non_blocking=True)
        xocc = batch.get("x_occ", None)
        if xocc is not None:
            xocc = xocc.to(device, non_blocking=True)

        logits = model(xb, xf, x_occ=xocc)
        clip_ids = list(batch["clip_id"])
        window_idxs = batch["window_idx"].detach().cpu().numpy().tolist()

        for h in HEAD_NAMES:
            prob = torch.softmax(logits[h], dim=1)
            pred = prob.argmax(dim=1)
            target = batch[TARGET_KEYS[h]]

            head_state[h]["probs"].extend(prob.detach().cpu().numpy().astype(np.float32).tolist())
            head_state[h]["preds"].extend(pred.detach().cpu().numpy().astype(int).tolist())
            head_state[h]["targets"].extend(target.detach().cpu().numpy().astype(int).tolist())
            head_state[h]["clip_ids"].extend(clip_ids)
            head_state[h]["window_idxs"].extend(window_idxs)

    split_summary = {}
    combined_clip_rows = []
    combined_window_rows = []

    for h in HEAD_NAMES:
        s = head_state[h]
        window_metrics = _safe_metrics(s["targets"], s["preds"])
        probs_np = [np.asarray(p, dtype=np.float32) for p in s["probs"]]
        clip_res = aggregate_clip_level(
            targets=s["targets"],
            probs=probs_np,
            clip_ids=s["clip_ids"],
            agg_mode=cfg["eval"]["clip_agg_mode"],
            topk=cfg["eval"]["clip_topk"],
            ignore_label=IGNORE_LABEL,
        )
        clip_metrics = _safe_metrics(clip_res["targets"], clip_res["preds"])

        n_cls = NUM_CLASSES[h]
        cm = confusion_matrix(clip_res["targets"], clip_res["preds"], labels=list(range(n_cls)))
        _save_confusion(cm, [str(i) for i in range(n_cls)], save_root / f"{split_name}_{h}_clip_confusion.csv")

        clip_rows = _prediction_rows_from_clip_result(split_name, h, clip_res)
        clip_df = pd.DataFrame(clip_rows)
        clip_df.to_csv(save_root / f"{split_name}_{h}_clip_predictions.csv", index=False, encoding="utf-8-sig")
        combined_clip_rows.extend(clip_rows)

        if save_window_predictions:
            window_rows = _prediction_rows_from_window_state(split_name, h, s)
            pd.DataFrame(window_rows).to_csv(save_root / f"{split_name}_{h}_window_predictions.csv", index=False, encoding="utf-8-sig")
            combined_window_rows.extend(window_rows)

        split_summary[h] = {
            "window": window_metrics,
            "clip": clip_metrics,
            "clip_prediction_file": str(save_root / f"{split_name}_{h}_clip_predictions.csv"),
            "window_prediction_file": str(save_root / f"{split_name}_{h}_window_predictions.csv") if save_window_predictions else None,
        }

        log.info(
            f"[{split_name} {h}] "
            f"clip_f1={clip_metrics['f1_macro']:.4f} clip_acc={clip_metrics['accuracy']:.4f} "
            f"window_f1={window_metrics['f1_macro']:.4f} window_acc={window_metrics['accuracy']:.4f} "
            f"n_clip={clip_metrics['n']} n_window={window_metrics['n']}"
        )

    pd.DataFrame(combined_clip_rows).to_csv(save_root / f"{split_name}_predictions.csv", index=False, encoding="utf-8-sig")
    if save_window_predictions:
        pd.DataFrame(combined_window_rows).to_csv(save_root / f"{split_name}_window_predictions.csv", index=False, encoding="utf-8-sig")

    return split_summary


def _build_drop_summary(clean_summary: dict, masked_summary: dict) -> dict:
    rows = []
    out = {}
    for h in HEAD_NAMES:
        c = clean_summary[h]["clip"]
        m = masked_summary[h]["clip"]
        d = {
            "clean_clip_f1_macro": c["f1_macro"],
            "masked_clip_f1_macro": m["f1_macro"],
            "drop_clip_f1_macro": c["f1_macro"] - m["f1_macro"],
            "relative_drop_clip_f1_macro": (c["f1_macro"] - m["f1_macro"]) / c["f1_macro"] if c["f1_macro"] > 0 else 0.0,
            "pdi_clip_f1_percent": 100.0 * ((c["f1_macro"] - m["f1_macro"]) / c["f1_macro"]) if c["f1_macro"] > 0 else 0.0,
            "clean_clip_acc": c["accuracy"],
            "masked_clip_acc": m["accuracy"],
            "drop_clip_acc": c["accuracy"] - m["accuracy"],
            "relative_drop_clip_acc": (c["accuracy"] - m["accuracy"]) / c["accuracy"] if c["accuracy"] > 0 else 0.0,
            "pdi_clip_acc_percent": 100.0 * ((c["accuracy"] - m["accuracy"]) / c["accuracy"]) if c["accuracy"] > 0 else 0.0,
        }
        out[h] = d
        rows.append({"head": h, **d})
    return out, pd.DataFrame(rows)


def run_eval_for_config(
    config_path: str,
    checkpoint: str | None,
    splits: list[str],
    batch_size: int | None,
    num_workers: int | None,
    save_window_predictions: bool,
) -> None:
    cfg = load_yaml(config_path)
    set_seed(int(cfg.get("seed", 42)))

    save_root = ensure_dir(cfg["paths"]["save_root"])
    log = get_logger("v5_eval_only", log_file=save_root / "eval_predictions.log")
    log.info(f"config={config_path}")

    if batch_size is not None:
        cfg["train"]["batch_size"] = int(batch_size)
    if num_workers is not None:
        cfg["train"]["num_workers"] = int(num_workers)

    device = _device_from_cfg(cfg)
    log.info(f"device={device}")

    # test split만 사용한다. build_clip_splits는 manifest 재현용으로 train/val도 반환하지만 preload는 test만 수행한다.
    _train_clips, _val_clips, test_eval_clips = build_clip_splits(cfg, save_root, log)

    model, _model_meta = build_model(cfg, device)
    ckpt_path = Path(checkpoint) if checkpoint else save_root / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    log.info(f"loaded checkpoint={ckpt_path}")

    common = _common_preload_kwargs(cfg, log)
    summaries = {}
    for split_name, clips in test_eval_clips.items():
        if splits and split_name not in splits:
            continue
        items = preload_multitask_windows(clips, desc=f"preload {split_name}", **common)
        loader = DataLoader(
            MemoryMultitaskDataset(items),
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=False,
            num_workers=int(cfg["train"]["num_workers"]),
            pin_memory=(device != "cpu"),
        )
        summaries[split_name] = evaluate_loader_and_save_predictions(
            model=model,
            loader=loader,
            split_name=split_name,
            save_root=save_root,
            device=device,
            cfg=cfg,
            log=log,
            save_window_predictions=save_window_predictions,
        )

    save_json(summaries, save_root / "eval_predictions_summary.json")

    if "test_clean" in summaries and "test_masked" in summaries:
        drop_json, drop_df = _build_drop_summary(summaries["test_clean"], summaries["test_masked"])
        save_json(drop_json, save_root / "test_clean_vs_masked_drop_with_pdi.json")
        drop_df.to_csv(save_root / "test_clean_vs_masked_drop_with_pdi.csv", index=False, encoding="utf-8-sig")
        log.info(f"saved drop/PDI summary={save_root / 'test_clean_vs_masked_drop_with_pdi.csv'}")

    log.info("eval-only prediction export done")


def main():
    parser = argparse.ArgumentParser(description="Eval only: save per-sample probability CSV for ROC/PR and PDI analysis.")
    parser.add_argument("--config", action="append", required=True, help="Config path. Can be passed multiple times.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path. Default: paths.save_root/best.pt")
    parser.add_argument("--splits", nargs="*", default=["test_clean", "test_masked"], help="Splits to evaluate.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--no-window-predictions", action="store_true", help="Only save clip-level predictions.")
    args = parser.parse_args()

    if args.checkpoint and len(args.config) > 1:
        raise ValueError("--checkpoint can be used with one --config only. For multiple configs, use default save_root/best.pt.")

    for cfg in args.config:
        run_eval_for_config(
            config_path=cfg,
            checkpoint=args.checkpoint,
            splits=args.splits,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            save_window_predictions=not args.no_window_predictions,
        )


if __name__ == "__main__":
    main()
