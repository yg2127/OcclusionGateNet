"""Subject-disjoint split.

Default behavior in this version:
    Use a fixed gaze-aware subject split.

Why:
    Original has_gaze-stratified split only balanced whether a subject has gaze data.
    It did not ensure that gA/gB/gC gaze subjects were distributed across
    train/val/test. As a result, the previous test gaze split could be dominated
    by only gC subjects.

This file keeps the original stratified split code for fallback, but
split_single_fold() currently returns the fixed split first.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from src.data.clip_builder import ClipRecord
from src.utils.io import load_json, save_json


# ============================================================
# Fixed gaze-aware subject split
# ============================================================

FIXED_GAZE_AWARE_SUBJECT_SPLIT: dict[str, list[str]] = {
    "train": [
        "gA_1",
        "gA_2",
        "gA_3",
        "gB_6",
        "gB_7",
        "gB_9",
        "gC_11",
        "gC_15",
    ],
    "val": [
        "gA_4",
        "gB_8",
        "gC_14",
    ],
    "test": [
        "gA_5",
        "gB_10",
        "gC_12",
        "gC_13",
    ],
}

"""FIXED_GAZE_AWARE_SUBJECT_SPLIT: dict[str, list[str]] = {
    "train": [
        # gaze subjects
        "gA_1",
        "gA_2",
        "gA_3",
        "gB_6",
        "gB_7",
        "gB_9",
        "gC_11",
        "gC_15",

        # non-gaze subjects
        "gE_26",
        "gE_27",
        "gE_29",
        "gF_22",
        "gF_23",
        "gF_24",
        "gZ_31",
        "gZ_32",
        "gZ_34",
    ],
    "val": [
        # gaze subjects
        "gA_4",
        "gB_8",
        "gC_14",

        # non-gaze subject
        "gZ_35",
    ],
    "test": [
        # gaze subjects
        "gA_5",
        "gB_10",
        "gC_12",
        "gC_13",

        # non-gaze subjects
        "gE_30",
        "gF_25",
        "gZ_33",
        "gZ_36",
        "gZ_37",
    ],
}"""


def _subjects_with_gaze(clips: list[ClipRecord]) -> set[str]:
    return {c.subject_key for c in clips if c.source == "gaze"}


def _source_counter(clips: list[ClipRecord]) -> dict:
    return dict(Counter(c.source for c in clips))


def _subject_counter(clips: list[ClipRecord]) -> dict:
    return dict(Counter(c.subject_key for c in clips))


def split_fixed_subjects(
    clips: list[ClipRecord],
    fixed_subjects: dict[str, list[str]],
    logger=None,
) -> tuple[list[ClipRecord], list[ClipRecord], list[ClipRecord]]:
    """Split clips by a manually fixed subject split.

    This preserves subject-disjoint evaluation and ensures gaze subjects are
    distributed across train/val/test in a controlled way.
    """
    required_keys = {"train", "val", "test"}
    actual_keys = set(fixed_subjects.keys())

    if required_keys != actual_keys:
        raise ValueError(
            f"fixed_subjects must have exactly keys {required_keys}, "
            f"got {actual_keys}"
        )

    tr = set(fixed_subjects["train"])
    va = set(fixed_subjects["val"])
    te = set(fixed_subjects["test"])

    # Subject-disjoint assertion.
    assert tr.isdisjoint(va), f"train/val subject overlap: {tr & va}"
    assert tr.isdisjoint(te), f"train/test subject overlap: {tr & te}"
    assert va.isdisjoint(te), f"val/test subject overlap: {va & te}"

    all_fixed_subjects = tr | va | te
    all_clip_subjects = {c.subject_key for c in clips}

    missing_in_clips = sorted(all_fixed_subjects - all_clip_subjects)
    unassigned_subjects = sorted(all_clip_subjects - all_fixed_subjects)

    train = [c for c in clips if c.subject_key in tr]
    val = [c for c in clips if c.subject_key in va]
    test = [c for c in clips if c.subject_key in te]

    # Actual subject sets after filtering.
    tr_s = {c.subject_key for c in train}
    va_s = {c.subject_key for c in val}
    te_s = {c.subject_key for c in test}

    assert tr_s.isdisjoint(va_s), f"train/val leak: {tr_s & va_s}"
    assert tr_s.isdisjoint(te_s), f"train/test leak: {tr_s & te_s}"
    assert va_s.isdisjoint(te_s), f"val/test leak: {va_s & te_s}"

    if logger is not None:
        gaze_subjects = _subjects_with_gaze(clips)

        def _gaze_count(subjects: set[str]) -> int:
            return sum(1 for s in subjects if s in gaze_subjects)

        logger.info(
            "[fixed gaze-aware split] "
            f"train={len(train)} clips ({len(tr_s)} subj, gaze={_gaze_count(tr_s)}) | "
            f"val={len(val)} clips ({len(va_s)} subj, gaze={_gaze_count(va_s)}) | "
            f"test={len(test)} clips ({len(te_s)} subj, gaze={_gaze_count(te_s)})"
        )

        if missing_in_clips:
            logger.warning(
                f"[fixed split] subjects listed in fixed split but not found in clips: "
                f"{missing_in_clips}"
            )

        if unassigned_subjects:
            logger.warning(
                f"[fixed split] subjects found in clips but not assigned to any split: "
                f"{unassigned_subjects}"
            )

        for name, part in (("train", train), ("val", val), ("test", test)):
            subjects = sorted({c.subject_key for c in part})
            gaze_subj = sorted(set(subjects) & gaze_subjects)

            logger.info(f"  {name} subjects: {subjects}")
            logger.info(f"  {name} gaze subjects: {gaze_subj}")
            logger.info(f"  {name} source dist: {_source_counter(part)}")

    return train, val, test


# ============================================================
# Original stratified-group split fallback
# ============================================================

def _stratified_group_split(
    clips: list[ClipRecord],
    subject_has_gaze: dict[str, bool],
    test_ratio: float,
    seed: int,
) -> tuple[list[ClipRecord], list[ClipRecord]]:
    """Group-aware stratified split.

    Label y is whether each subject has gaze clips.
    """
    subjects = sorted({c.subject_key for c in clips})

    if len(subjects) == 0:
        raise ValueError("No subjects found in clips.")

    y_by_subj = np.array([int(subject_has_gaze.get(s, False)) for s in subjects])
    groups = np.array(subjects)

    n_splits = max(2, int(round(1.0 / test_ratio)))

    # StratifiedGroupKFold can fail if one class has too few groups.
    # Keep the original behavior but provide a clearer error.
    try:
        skf = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )

        # Treat each subject as one sample.
        X = np.arange(len(subjects)).reshape(-1, 1)
        train_subj_idx, test_subj_idx = next(iter(skf.split(X, y_by_subj, groups)))

    except Exception as e:
        raise RuntimeError(
            "Failed to run StratifiedGroupKFold. "
            f"n_subjects={len(subjects)}, n_splits={n_splits}, "
            f"y_counts={dict(Counter(y_by_subj.tolist()))}"
        ) from e

    train_subj = set(groups[train_subj_idx])
    test_subj = set(groups[test_subj_idx])

    train = [c for c in clips if c.subject_key in train_subj]
    test = [c for c in clips if c.subject_key in test_subj]

    return train, test


def split_single_fold(
    clips: list[ClipRecord],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    logger=None,
) -> tuple[list[ClipRecord], list[ClipRecord], list[ClipRecord]]:
    """Return train/val/test split.

    Minimal-change behavior:
        Always use the fixed gaze-aware subject split.

    The original has_gaze-stratified split logic is retained below as fallback
    code, but it is not reached because of the early return.
    """

    # ------------------------------------------------------------
    # Minimal modification:
    # Keep train.py/YAML/model untouched.
    # Override only the subject split here.
    # ------------------------------------------------------------
    return split_fixed_subjects(
        clips=clips,
        fixed_subjects=FIXED_GAZE_AWARE_SUBJECT_SPLIT,
        logger=logger,
    )

    # ------------------------------------------------------------
    # Original behavior below.
    # This code is intentionally preserved for reference/fallback.
    # Remove the early return above if you want to restore it.
    # ------------------------------------------------------------
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    gaze_subjects = _subjects_with_gaze(clips)
    subject_has_gaze = {
        s: (s in gaze_subjects)
        for s in {c.subject_key for c in clips}
    }

    rest, test = _stratified_group_split(
        clips=clips,
        subject_has_gaze=subject_has_gaze,
        test_ratio=test_ratio,
        seed=seed,
    )

    # Split validation from rest.
    # val ratio inside rest = val / (train + val)
    val_within_rest = val_ratio / (train_ratio + val_ratio)

    train, val = _stratified_group_split(
        clips=rest,
        subject_has_gaze=subject_has_gaze,
        test_ratio=val_within_rest,
        seed=seed + 1,
    )

    # Disjoint assertion.
    tr_s = {c.subject_key for c in train}
    va_s = {c.subject_key for c in val}
    te_s = {c.subject_key for c in test}

    assert tr_s.isdisjoint(va_s), f"train/val leak: {tr_s & va_s}"
    assert tr_s.isdisjoint(te_s), f"train/test leak: {tr_s & te_s}"
    assert va_s.isdisjoint(te_s), f"val/test leak: {va_s & te_s}"

    if logger is not None:
        def _gaze_count(subjects: set[str]) -> int:
            return sum(1 for s in subjects if subject_has_gaze.get(s))

        logger.info(
            f"split: train={len(train)} clips ({len(tr_s)} subj, gaze={_gaze_count(tr_s)}) | "
            f"val={len(val)} ({len(va_s)} subj, gaze={_gaze_count(va_s)}) | "
            f"test={len(test)} ({len(te_s)} subj, gaze={_gaze_count(te_s)})"
        )

        for name, part in (("train", train), ("val", val), ("test", test)):
            by_src = Counter(c.source for c in part)
            logger.info(f"  {name} source dist: {dict(by_src)}")

    return train, val, test


# ============================================================
# Save/load split info
# ============================================================

def save_split_info(
    train: list[ClipRecord],
    val: list[ClipRecord],
    test: list[ClipRecord],
    seed: int,
    path: str | Path,
) -> None:
    info = {
        "seed": seed,
        "split_mode": "fixed_gaze_aware",
        "subjects": {
            "train": sorted({c.subject_key for c in train}),
            "val": sorted({c.subject_key for c in val}),
            "test": sorted({c.subject_key for c in test}),
        },
        "clip_counts": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "source_counts": {
            name: dict(Counter(c.source for c in part))
            for name, part in (
                ("train", train),
                ("val", val),
                ("test", test),
            )
        },
        "gaze_subjects": {
            name: sorted(_subjects_with_gaze(part))
            for name, part in (
                ("train", train),
                ("val", val),
                ("test", test),
            )
        },
    }

    save_json(info, path)


def load_split_info(path: str | Path) -> dict:
    return load_json(path)


def filter_by_split_info(
    clips: list[ClipRecord],
    split_info: dict,
) -> tuple[list[ClipRecord], list[ClipRecord], list[ClipRecord]]:
    tr = set(split_info["subjects"]["train"])
    va = set(split_info["subjects"]["val"])
    te = set(split_info["subjects"]["test"])

    train = [c for c in clips if c.subject_key in tr]
    val = [c for c in clips if c.subject_key in va]
    test = [c for c in clips if c.subject_key in te]

    # Safety check.
    tr_s = {c.subject_key for c in train}
    va_s = {c.subject_key for c in val}
    te_s = {c.subject_key for c in test}

    assert tr_s.isdisjoint(va_s), f"train/val leak: {tr_s & va_s}"
    assert tr_s.isdisjoint(te_s), f"train/test leak: {tr_s & te_s}"
    assert va_s.isdisjoint(te_s), f"val/test leak: {va_s & te_s}"

    return train, val, test