# Auto-split from Pasted code(257).py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from constants.gaze_zones import NUM_GAZE_ZONES
from src.data.clip_builder import NUM_ACTION_CLASSES, NUM_HANDS_CLASSES, NUM_TALK_CLASSES
from src.training.loops import HEAD_NAMES, run_one_epoch


def _save_confusion(cm: np.ndarray, labels: list[str], path: Path) -> None:
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        path,
        encoding="utf-8-sig",
    )


def _num_classes_for_head(head_name: str) -> int:
    if head_name == "action":
        return NUM_ACTION_CLASSES
    if head_name == "gaze":
        return NUM_GAZE_ZONES
    if head_name == "hands":
        return NUM_HANDS_CLASSES
    if head_name == "talk":
        return NUM_TALK_CLASSES
    raise ValueError(f"Unknown head_name={head_name}")


def evaluate_test_split(
    *,
    name: str,
    model,
    loader,
    optimizer,
    criterion,
    device,
    cfg: dict,
    ablation_cfg: dict,
    save_root: Path,
    log,
) -> dict:
    """Evaluate one held-out test split and save per-head confusion matrices."""
    out = run_one_epoch(
        model,
        loader,
        optimizer,
        criterion,
        device,
        train=False,
        grad_clip_norm=None,
        agg_mode=cfg["eval"]["clip_agg_mode"],
        topk=cfg["eval"]["clip_topk"],
        ablation_cfg=ablation_cfg,
    )

    for h in HEAD_NAMES:
        th = out["heads"][h]

        if not th.get("clip_targets"):
            log.warning(f"[{name} {h}] no clip_targets; confusion matrix skipped")
            continue

        num = _num_classes_for_head(h)
        cm = confusion_matrix(
            th["clip_targets"],
            th["clip_preds"],
            labels=list(range(num)),
        )

        _save_confusion(
            cm,
            [str(i) for i in range(num)],
            save_root / f"{name}_{h}_clip_confusion.csv",
        )

        log.info(
            f"[{name} {h}] "
            f"c_f1={th.get('clip_f1_macro', 0.0):.4f} "
            f"c_acc={th.get('clip_acc', 0.0):.4f} "
            f"w_f1={th.get('window_f1_macro', 0.0):.4f} "
            f"w_acc={th.get('window_acc', 0.0):.4f}"
        )

    tgb = out.get("gaze_binary_on_distraction", {})
    if tgb:
        log.info(
            f"[{name} gaze_bin(dist)] "
            f"window acc={tgb.get('window_acc', 0):.4f} "
            f"f1={tgb.get('window_f1_macro', 0):.4f} "
            f"| clip acc={tgb.get('clip_acc', 0):.4f} "
            f"f1={tgb.get('clip_f1_macro', 0):.4f} "
            f"| n={tgb.get('n')} "
            f"(front={tgb.get('support_front')}, "
            f"not_front={tgb.get('support_not_front')})"
        )

        if tgb.get("clip_targets"):
            cm = confusion_matrix(
                tgb["clip_targets"],
                tgb["clip_preds"],
                labels=[0, 1],
            )
            _save_confusion(
                cm,
                ["not_front", "front"],
                save_root / f"{name}_gaze_binary_on_distraction.csv",
            )

    return out


def _head_metrics_for_summary(out: dict) -> dict:
    return {
        h: {
            "window_f1_macro": out["heads"][h].get("window_f1_macro", 0.0),
            "window_acc": out["heads"][h].get("window_acc", 0.0),
            "clip_f1_macro": out["heads"][h].get("clip_f1_macro", 0.0),
            "clip_acc": out["heads"][h].get("clip_acc", 0.0),
        }
        for h in HEAD_NAMES
    }


def _gaze_binary_summary(out: dict) -> dict | None:
    tgb = out.get("gaze_binary_on_distraction", {})
    if not tgb:
        return None
    return {
        "window_acc": tgb.get("window_acc", 0.0),
        "window_f1_macro": tgb.get("window_f1_macro", 0.0),
        "clip_acc": tgb.get("clip_acc", 0.0),
        "clip_f1_macro": tgb.get("clip_f1_macro", 0.0),
        "n_windows": tgb.get("n", 0),
        "support_front": tgb.get("support_front", 0),
        "support_not_front": tgb.get("support_not_front", 0),
    }


def build_clean_masked_drop_summary(clean_out: dict, masked_out: dict) -> dict:
    """Positive drop means masked performance is lower than clean performance."""
    drop = {}

    for h in HEAD_NAMES:
        ch = clean_out["heads"][h]
        mh = masked_out["heads"][h]

        clean_clip_f1 = ch.get("clip_f1_macro", 0.0)
        masked_clip_f1 = mh.get("clip_f1_macro", 0.0)
        clean_clip_acc = ch.get("clip_acc", 0.0)
        masked_clip_acc = mh.get("clip_acc", 0.0)
        clean_window_f1 = ch.get("window_f1_macro", 0.0)
        masked_window_f1 = mh.get("window_f1_macro", 0.0)
        clean_window_acc = ch.get("window_acc", 0.0)
        masked_window_acc = mh.get("window_acc", 0.0)

        drop[h] = {
            "clean_clip_f1_macro": clean_clip_f1,
            "masked_clip_f1_macro": masked_clip_f1,
            "drop_clip_f1_macro": clean_clip_f1 - masked_clip_f1,
            "relative_drop_clip_f1_macro": (
                (clean_clip_f1 - masked_clip_f1) / clean_clip_f1
                if clean_clip_f1 > 0
                else 0.0
            ),
            "clean_clip_acc": clean_clip_acc,
            "masked_clip_acc": masked_clip_acc,
            "drop_clip_acc": clean_clip_acc - masked_clip_acc,
            "relative_drop_clip_acc": (
                (clean_clip_acc - masked_clip_acc) / clean_clip_acc
                if clean_clip_acc > 0
                else 0.0
            ),
            "clean_window_f1_macro": clean_window_f1,
            "masked_window_f1_macro": masked_window_f1,
            "drop_window_f1_macro": clean_window_f1 - masked_window_f1,
            "relative_drop_window_f1_macro": (
                (clean_window_f1 - masked_window_f1) / clean_window_f1
                if clean_window_f1 > 0
                else 0.0
            ),
            "clean_window_acc": clean_window_acc,
            "masked_window_acc": masked_window_acc,
            "drop_window_acc": clean_window_acc - masked_window_acc,
            "relative_drop_window_acc": (
                (clean_window_acc - masked_window_acc) / clean_window_acc
                if clean_window_acc > 0
                else 0.0
            ),
        }

    return drop
