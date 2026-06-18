"""Window-level softmax → clip-level aggregation (per-head)."""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def aggregate_probs_topk_mean(prob_list: list[np.ndarray], topk: int = 3) -> np.ndarray:
    probs = np.stack(prob_list, axis=0)
    W, C = probs.shape
    k = min(topk, W)
    agg = np.zeros((C,), dtype=np.float32)
    for c in range(C):
        vals = probs[:, c]
        agg[c] = float(np.mean(np.sort(vals)[-k:]))
    return agg


def aggregate_clip_level(
    targets: Sequence[int],
    probs: Sequence[np.ndarray],
    clip_ids: Sequence[str],
    agg_mode: str = "topk_mean",
    topk: int = 3,
    ignore_label: int = -100,
) -> dict:
    """Window → clip 집계. target 이 ignore_label 인 샘플은 제외."""
    per_clip: dict[str, list[np.ndarray]] = defaultdict(list)
    target_of: dict[str, int] = {}
    for y, p, c in zip(targets, probs, clip_ids):
        if int(y) == ignore_label:
            continue
        per_clip[c].append(np.asarray(p, dtype=np.float32))
        if c in target_of:
            if target_of[c] != int(y):
                # 같은 clip_id 안에 라벨 서로 다르면 경고 (발생 시 버그)
                raise ValueError(f"다른 target 이 같은 clip_id에: {c}")
        else:
            target_of[c] = int(y)

    agg_targets, agg_preds, agg_probs, agg_ids = [], [], [], []
    for c, plist in per_clip.items():
        if agg_mode == "mean":
            final = np.mean(np.stack(plist, axis=0), axis=0)
        elif agg_mode == "topk_mean":
            final = aggregate_probs_topk_mean(plist, topk=topk)
        else:
            raise ValueError(agg_mode)
        agg_targets.append(target_of[c])
        agg_preds.append(int(np.argmax(final)))
        agg_probs.append(final)
        agg_ids.append(c)

    if not agg_targets:
        return {"targets": [], "preds": [], "probs": [], "clip_ids": [], "acc": 0.0, "f1_macro": 0.0}

    return {
        "targets": agg_targets,
        "preds": agg_preds,
        "probs": agg_probs,
        "clip_ids": agg_ids,
        "acc": float(accuracy_score(agg_targets, agg_preds)),
        "f1_macro": float(f1_score(agg_targets, agg_preds, average="macro", zero_division=0)),
    }
