#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bootstrap clean/masked clip predictions for gaze045_light artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

TASKS = ["action", "gaze", "hands", "talk"]


def make_pair_key(sample_id: str) -> str:
    return str(sample_id).replace("__clean__", "__PAIR__").replace("__masked__", "__PAIR__")


def load_pair(model_dir: Path, task: str) -> pd.DataFrame:
    clean_path = model_dir / f"test_clean_{task}_clip_predictions.csv"
    masked_path = model_dir / f"test_masked_{task}_clip_predictions.csv"
    if not clean_path.exists() or not masked_path.exists():
        raise FileNotFoundError(f"Missing prediction pair for {model_dir.name}/{task}")
    cols = ["sample_id", "y_true", "y_pred"]
    clean = pd.read_csv(clean_path, usecols=cols)
    masked = pd.read_csv(masked_path, usecols=cols)
    clean["pair_key"] = clean["sample_id"].map(make_pair_key)
    masked["pair_key"] = masked["sample_id"].map(make_pair_key)
    clean = clean.rename(columns={"sample_id": "clean_sample_id", "y_true": "y_true_clean", "y_pred": "y_pred_clean"})
    masked = masked.rename(columns={"sample_id": "masked_sample_id", "y_true": "y_true_masked", "y_pred": "y_pred_masked"})
    df = clean.merge(masked, on="pair_key", how="inner", validate="one_to_one")
    return df.sort_values("pair_key").reset_index(drop=True)


def metrics(y_true, y_pred):
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def pdi(c, m):
    return np.nan if c == 0 else float((c - m) / c * 100.0)


def eval_one(df: pd.DataFrame, idx: np.ndarray, model: str, task: str, seed: int):
    b = df.iloc[idx]
    cm = metrics(b["y_true_clean"].to_numpy(), b["y_pred_clean"].to_numpy())
    mm = metrics(b["y_true_masked"].to_numpy(), b["y_pred_masked"].to_numpy())
    return {
        "model": model, "task": task, "bootstrap_seed": int(seed), "n_samples": int(len(b)),
        "clean_acc": cm["acc"], "masked_acc": mm["acc"], "drop_acc": cm["acc"] - mm["acc"],
        "clean_f1_macro": cm["f1_macro"], "masked_f1_macro": mm["f1_macro"],
        "drop_f1_macro": cm["f1_macro"] - mm["f1_macro"], "pdi_f1_macro": pdi(cm["f1_macro"], mm["f1_macro"]),
        "clean_f1_weighted": cm["f1_weighted"], "masked_f1_weighted": mm["f1_weighted"],
        "drop_f1_weighted": cm["f1_weighted"] - mm["f1_weighted"], "pdi_f1_weighted": pdi(cm["f1_weighted"], mm["f1_weighted"]),
    }


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in raw.columns if c not in {"model", "task", "bootstrap_seed", "n_samples"}]
    rows = []
    for (model, task), g in raw.groupby(["model", "task"], sort=True):
        row = {"model": model, "task": task, "n_bootstrap": len(g), "n_samples_mean": g["n_samples"].mean()}
        for c in metric_cols:
            v = g[c].dropna().to_numpy()
            row[f"{c}_mean"] = float(np.mean(v)) if len(v) else np.nan
            row[f"{c}_std"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
            row[f"{c}_ci95_low"] = float(np.percentile(v, 2.5)) if len(v) else np.nan
            row[f"{c}_ci95_high"] = float(np.percentile(v, 97.5)) if len(v) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/shared/scuppy/hyi/Ablation/Ablation_Classification_V5")
    ap.add_argument("--artifacts_dir", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--num_bootstrap_seeds", type=int, default=5000)
    ap.add_argument("--seed_start", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else root / "artifacts_gaze045_light"
    out_dir = Path(args.out_dir) if args.out_dir else root / "bootstrap_results_gaze045_light"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dirs = sorted([p for p in artifacts_dir.iterdir() if p.is_dir() and any(p.glob("test_clean_*_clip_predictions.csv"))])
    if not model_dirs:
        raise RuntimeError(f"No prediction artifacts found in {artifacts_dir}")

    data = {m.name: {} for m in model_dirs}
    for m in model_dirs:
        for task in args.tasks:
            try:
                data[m.name][task] = load_pair(m, task)
                print(f"[LOAD] {m.name} {task} n={len(data[m.name][task])}")
            except FileNotFoundError as e:
                print(f"[WARN] {e}")

    rows = []
    for task in args.tasks:
        models_with_task = [m.name for m in model_dirs if task in data[m.name]]
        if not models_with_task:
            continue
        common = None
        for model in models_with_task:
            keys = set(data[model][task]["pair_key"])
            common = keys if common is None else common & keys
        ref_order = data[models_with_task[0]][task]["pair_key"].tolist()
        ordered = [k for k in ref_order if k in common]
        aligned = {model: data[model][task].set_index("pair_key").loc[ordered].reset_index() for model in models_with_task}
        n = len(ordered)
        print(f"[BOOT] task={task} common_n={n} models={len(models_with_task)}")
        for i, seed in enumerate(range(args.seed_start, args.seed_start + args.num_bootstrap_seeds), 1):
            rng = np.random.default_rng(seed)
            idx = rng.choice(np.arange(n), size=n, replace=True)
            for model in models_with_task:
                rows.append(eval_one(aligned[model], idx, model, task, seed))
            if i % 500 == 0:
                print(f"  done {i}/{args.num_bootstrap_seeds}")

    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "bootstrap_raw_by_seed.csv", index=False, encoding="utf-8-sig")
    summary = summarize(raw)
    summary.to_csv(out_dir / "bootstrap_summary_by_model_task.csv", index=False, encoding="utf-8-sig")

    meta = vars(args) | {"artifacts_dir": str(artifacts_dir), "out_dir": str(out_dir)}
    with open(out_dir / "bootstrap_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {out_dir}")


if __name__ == "__main__":
    main()
