#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick task-wise ranking from bootstrap_summary_by_model_task.csv."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", default="/data/shared/scuppy/hyi/Ablation/Ablation_Classification_V5/bootstrap_results_gaze045_light/bootstrap_summary_by_model_task.csv")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()
    df = pd.read_csv(args.summary_csv)

    cols = [
        "model", "task",
        "clean_f1_macro_mean", "masked_f1_macro_mean", "drop_f1_macro_mean", "pdi_f1_macro_mean",
        "clean_f1_weighted_mean", "masked_f1_weighted_mean", "drop_f1_weighted_mean", "pdi_f1_weighted_mean",
    ]
    existing = [c for c in cols if c in df.columns]

    print("\n=== Task-wise ranking by masked_f1_macro_mean ===")
    for task in sorted(df["task"].unique()):
        sub = df[df["task"] == task].sort_values("masked_f1_macro_mean", ascending=False)
        print(f"\n[{task}]")
        print(sub[existing].head(args.top).to_string(index=False))

    print("\n=== Task-wise ranking by low PDI macro ===")
    for task in sorted(df["task"].unique()):
        sub = df[df["task"] == task].sort_values("pdi_f1_macro_mean", ascending=True)
        print(f"\n[{task}]")
        print(sub[existing].head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
