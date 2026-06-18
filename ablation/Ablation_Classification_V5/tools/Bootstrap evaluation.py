#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bootstrap evaluation from saved *_clip_predictions.csv files.

Expected structure:
  ROOT/
    artifacts/
      model_A/
        test_clean_action_clip_predictions.csv
        test_masked_action_clip_predictions.csv
        test_clean_gaze_clip_predictions.csv
        test_masked_gaze_clip_predictions.csv
        ...
      model_B/
        ...

This script computes bootstrap metrics for each model/task:
  - clean_f1_macro
  - masked_f1_macro
  - drop_f1_macro
  - pdi_f1_macro
  - clean_acc
  - masked_acc
  - drop_acc
  - clean_f1_weighted
  - masked_f1_weighted
  - drop_f1_weighted

Important:
  Bootstrap indices are generated once per task/seed from the reference model,
  then reused for every model. This prevents unfair model-specific resampling.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


TASKS = ["action", "gaze", "hands", "talk"]


def make_pair_key(sample_id: str) -> str:
    """
    Convert clean/masked sample_id into a common key.

    Example:
      xxx__clean__dist__00000  -> xxx__PAIR__dist__00000
      xxx__masked__dist__00000 -> xxx__PAIR__dist__00000
    """
    return (
        str(sample_id)
        .replace("__clean__", "__PAIR__")
        .replace("__masked__", "__PAIR__")
    )


def find_model_dirs(artifacts_dir: Path, include: List[str] = None, exclude: List[str] = None) -> List[Path]:
    include = include or []
    exclude = exclude or []

    model_dirs = []
    for d in sorted(artifacts_dir.iterdir()):
        if not d.is_dir():
            continue

        name = d.name

        if include and not any(x in name for x in include):
            continue

        if exclude and any(x in name for x in exclude):
            continue

        has_any_pred = any(d.glob("test_clean_*_clip_predictions.csv"))
        if has_any_pred:
            model_dirs.append(d)

    return model_dirs


def load_task_pair(model_dir: Path, task: str) -> pd.DataFrame:
    """
    Load clean/masked prediction CSVs and align them by pair_key.

    Returns:
      columns:
        pair_key
        clean_sample_id
        masked_sample_id
        y_true_clean
        y_pred_clean
        y_true_masked
        y_pred_masked
    """
    clean_path = model_dir / f"test_clean_{task}_clip_predictions.csv"
    masked_path = model_dir / f"test_masked_{task}_clip_predictions.csv"

    if not clean_path.exists():
        raise FileNotFoundError(f"Missing clean prediction file: {clean_path}")
    if not masked_path.exists():
        raise FileNotFoundError(f"Missing masked prediction file: {masked_path}")

    usecols = ["sample_id", "y_true", "y_pred"]

    clean = pd.read_csv(clean_path, usecols=usecols)
    masked = pd.read_csv(masked_path, usecols=usecols)

    clean["pair_key"] = clean["sample_id"].map(make_pair_key)
    masked["pair_key"] = masked["sample_id"].map(make_pair_key)

    clean = clean.rename(
        columns={
            "sample_id": "clean_sample_id",
            "y_true": "y_true_clean",
            "y_pred": "y_pred_clean",
        }
    )

    masked = masked.rename(
        columns={
            "sample_id": "masked_sample_id",
            "y_true": "y_true_masked",
            "y_pred": "y_pred_masked",
        }
    )

    merged = clean.merge(
        masked,
        on="pair_key",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) == 0:
        raise RuntimeError(f"No common clean/masked samples after pair_key merge: {model_dir.name} / {task}")

    # 기본적으로 clean/masked의 true label은 같아야 정상.
    mismatch = (merged["y_true_clean"].values != merged["y_true_masked"].values).sum()
    if mismatch > 0:
        print(f"[WARN] {model_dir.name}/{task}: y_true mismatch count = {mismatch}")

    return merged.sort_values("pair_key").reset_index(drop=True)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def safe_pdi(clean_value: float, masked_value: float) -> float:
    if clean_value == 0:
        return np.nan
    return float((clean_value - masked_value) / clean_value * 100.0)


def eval_bootstrap_one(
    df: pd.DataFrame,
    boot_idx: np.ndarray,
    model_name: str,
    task: str,
    bootstrap_seed: int,
) -> Dict[str, float]:
    bdf = df.iloc[boot_idx]

    y_true_clean = bdf["y_true_clean"].to_numpy()
    y_pred_clean = bdf["y_pred_clean"].to_numpy()

    y_true_masked = bdf["y_true_masked"].to_numpy()
    y_pred_masked = bdf["y_pred_masked"].to_numpy()

    clean_m = compute_metrics(y_true_clean, y_pred_clean)
    masked_m = compute_metrics(y_true_masked, y_pred_masked)

    row = {
        "model": model_name,
        "task": task,
        "bootstrap_seed": int(bootstrap_seed),
        "n_samples": int(len(bdf)),

        "clean_acc": clean_m["acc"],
        "masked_acc": masked_m["acc"],
        "drop_acc": clean_m["acc"] - masked_m["acc"],

        "clean_f1_macro": clean_m["f1_macro"],
        "masked_f1_macro": masked_m["f1_macro"],
        "drop_f1_macro": clean_m["f1_macro"] - masked_m["f1_macro"],
        "pdi_f1_macro": safe_pdi(clean_m["f1_macro"], masked_m["f1_macro"]),

        "clean_f1_weighted": clean_m["f1_weighted"],
        "masked_f1_weighted": masked_m["f1_weighted"],
        "drop_f1_weighted": clean_m["f1_weighted"] - masked_m["f1_weighted"],
        "pdi_f1_weighted": safe_pdi(clean_m["f1_weighted"], masked_m["f1_weighted"]),
    }

    return row


def summarize_bootstrap(results_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "clean_acc",
        "masked_acc",
        "drop_acc",
        "clean_f1_macro",
        "masked_f1_macro",
        "drop_f1_macro",
        "pdi_f1_macro",
        "clean_f1_weighted",
        "masked_f1_weighted",
        "drop_f1_weighted",
        "pdi_f1_weighted",
    ]

    rows = []

    for (model, task), g in results_df.groupby(["model", "task"], sort=True):
        row = {
            "model": model,
            "task": task,
            "n_bootstrap": int(len(g)),
            "n_samples_mean": float(g["n_samples"].mean()),
        }

        for col in metric_cols:
            values = g[col].dropna().to_numpy()
            if len(values) == 0:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_std"] = np.nan
                row[f"{col}_ci95_low"] = np.nan
                row[f"{col}_ci95_high"] = np.nan
                continue

            row[f"{col}_mean"] = float(np.mean(values))
            row[f"{col}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{col}_ci95_low"] = float(np.percentile(values, 2.5))
            row[f"{col}_ci95_high"] = float(np.percentile(values, 97.5))

        rows.append(row)

    return pd.DataFrame(rows)


def make_reference_keys(
    model_task_data: Dict[str, Dict[str, pd.DataFrame]],
    reference_model: str,
    task: str,
) -> List[str]:
    """
    Reference model의 pair_key 순서를 기준으로 bootstrap population을 만든다.
    이후 모든 모델은 이 key 목록에 맞춰 reindex한다.
    """
    ref_df = model_task_data[reference_model][task]
    return ref_df["pair_key"].tolist()


def align_all_models_to_reference(
    model_task_data: Dict[str, Dict[str, pd.DataFrame]],
    reference_model: str,
    tasks: List[str],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    모든 모델을 reference_model의 pair_key와 동일한 sample set/order로 정렬한다.
    모델별 누락 sample이 있으면 공통 intersection만 사용한다.
    """
    aligned = {m: {} for m in model_task_data.keys()}

    for task in tasks:
        # 모든 모델이 가진 pair_key의 intersection
        common_keys = None

        for model_name, task_map in model_task_data.items():
            if task not in task_map:
                continue
            keys = set(task_map[task]["pair_key"].tolist())
            common_keys = keys if common_keys is None else (common_keys & keys)

        if not common_keys:
            raise RuntimeError(f"No common pair_key across models for task={task}")

        # reference 순서를 유지하되, 모든 모델에 존재하는 key만 사용
        ref_keys = make_reference_keys(model_task_data, reference_model, task)
        ordered_common_keys = [k for k in ref_keys if k in common_keys]

        print(f"[ALIGN] task={task:6s} common paired samples = {len(ordered_common_keys)}")

        for model_name, task_map in model_task_data.items():
            if task not in task_map:
                continue

            df = task_map[task].set_index("pair_key").loc[ordered_common_keys].reset_index()
            aligned[model_name][task] = df

    return aligned


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        default="/data/shared/scuppy/hyi/Classification_model_V5_transformer",
        help="Project root directory containing artifacts/",
    )
    parser.add_argument(
        "--artifacts_dir",
        type=str,
        default=None,
        help="Artifacts directory. Default: ROOT/artifacts",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Default: ROOT/bootstrap_results_seed42",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=TASKS,
        choices=TASKS,
    )
    parser.add_argument(
        "--num_bootstrap_seeds",
        type=int,
        default=5000,
        help="Number of bootstrap RNG seeds. Each seed creates one bootstrap sample.",
    )
    parser.add_argument(
        "--seed_start",
        type=int,
        default=0,
        help="First bootstrap seed.",
    )
    parser.add_argument(
        "--reference_model",
        type=str,
        default=None,
        help="Reference model directory name. Default: first detected model.",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=[],
        help="Only include model directories whose names contain one of these strings.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Exclude model directories whose names contain one of these strings.",
    )

    args = parser.parse_args()

    root = Path(args.root)
    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else root / "artifacts"
    out_dir = Path(args.out_dir) if args.out_dir else root / "bootstrap_results_seed42"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] root          = {root}")
    print(f"[INFO] artifacts_dir = {artifacts_dir}")
    print(f"[INFO] out_dir       = {out_dir}")
    print(f"[INFO] tasks         = {args.tasks}")
    print(f"[INFO] bootstrap seeds = {args.seed_start} ~ {args.seed_start + args.num_bootstrap_seeds - 1}")

    model_dirs = find_model_dirs(
        artifacts_dir=artifacts_dir,
        include=args.include,
        exclude=args.exclude,
    )

    if len(model_dirs) == 0:
        raise RuntimeError(f"No model directories found in {artifacts_dir}")

    model_names = [d.name for d in model_dirs]

    if args.reference_model is None:
        reference_model = model_names[0]
    else:
        reference_model = args.reference_model
        if reference_model not in model_names:
            raise RuntimeError(
                f"reference_model={reference_model} not found. Available: {model_names}"
            )

    print("[INFO] detected models:")
    for name in model_names:
        marker = "  <REF>" if name == reference_model else ""
        print(f"  - {name}{marker}")

    # Load all model/task prediction pairs
    model_task_data: Dict[str, Dict[str, pd.DataFrame]] = {}

    for model_dir in model_dirs:
        model_name = model_dir.name
        model_task_data[model_name] = {}

        for task in args.tasks:
            clean_path = model_dir / f"test_clean_{task}_clip_predictions.csv"
            masked_path = model_dir / f"test_masked_{task}_clip_predictions.csv"

            if not clean_path.exists() or not masked_path.exists():
                print(f"[WARN] skip missing task: model={model_name}, task={task}")
                continue

            df = load_task_pair(model_dir, task)
            model_task_data[model_name][task] = df

            print(
                f"[LOAD] model={model_name:45s} task={task:6s} "
                f"paired_n={len(df)}"
            )

    # Align all models to same task-wise sample set/order
    aligned_data = align_all_models_to_reference(
        model_task_data=model_task_data,
        reference_model=reference_model,
        tasks=args.tasks,
    )

    # Save alignment metadata
    metadata = {
        "root": str(root),
        "artifacts_dir": str(artifacts_dir),
        "out_dir": str(out_dir),
        "tasks": args.tasks,
        "num_bootstrap_seeds": args.num_bootstrap_seeds,
        "seed_start": args.seed_start,
        "reference_model": reference_model,
        "models": model_names,
        "note": "Bootstrap index is shared across all models for each task/seed.",
    }

    with open(out_dir / "bootstrap_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    all_rows = []

    bootstrap_seeds = list(range(args.seed_start, args.seed_start + args.num_bootstrap_seeds))

    for task in args.tasks:
        if task not in aligned_data[reference_model]:
            print(f"[WARN] task not in reference model. skip: {task}")
            continue

        n = len(aligned_data[reference_model][task])
        print(f"\n[BOOT] task={task}, n={n}")

        for i, boot_seed in enumerate(bootstrap_seeds):
            rng = np.random.default_rng(boot_seed)
            boot_idx = rng.choice(np.arange(n), size=n, replace=True)

            for model_name in model_names:
                if task not in aligned_data[model_name]:
                    continue

                df = aligned_data[model_name][task]

                row = eval_bootstrap_one(
                    df=df,
                    boot_idx=boot_idx,
                    model_name=model_name,
                    task=task,
                    bootstrap_seed=boot_seed,
                )
                all_rows.append(row)

            if (i + 1) % 500 == 0:
                print(f"  done {i + 1}/{len(bootstrap_seeds)} bootstrap seeds")

    results_df = pd.DataFrame(all_rows)

    raw_csv = out_dir / "bootstrap_raw_by_seed.csv"
    results_df.to_csv(raw_csv, index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] raw bootstrap results: {raw_csv}")

    summary_df = summarize_bootstrap(results_df)
    summary_csv = out_dir / "bootstrap_summary_by_model_task.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"[SAVE] summary: {summary_csv}")

    # 보기 편한 pivot: task별 clean/masked/drop macro F1 mean
    pivot_cols = [
        "clean_f1_macro_mean",
        "masked_f1_macro_mean",
        "drop_f1_macro_mean",
        "pdi_f1_macro_mean",
        "clean_f1_weighted_mean",
        "masked_f1_weighted_mean",
        "drop_f1_weighted_mean",
        "pdi_f1_weighted_mean",
    ]

    compact = summary_df[
        ["model", "task", "n_bootstrap", "n_samples_mean"] + pivot_cols
    ].copy()

    compact_csv = out_dir / "bootstrap_compact_mean.csv"
    compact.to_csv(compact_csv, index=False, encoding="utf-8-sig")
    print(f"[SAVE] compact mean: {compact_csv}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()