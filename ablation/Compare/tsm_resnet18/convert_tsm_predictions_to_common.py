#!/usr/bin/env python3
"""Convert DMD TSM-ResNet18 prediction CSVs into the common ROC/PR format.

Input files already exist in the TSM run directory:
  - test_clean_predictions.csv
  - test_masked_predictions.csv

Their original columns are wide per-head columns:
  action_target, action_pred, action_p0...
  gaze_target, gaze_pred, gaze_p0...
  hands_target, hands_pred, hands_p0...
  talk_target, talk_pred, talk_p0...

This script writes common-format CSVs:
  - test_clean_action_clip_predictions.csv
  - test_clean_gaze_clip_predictions.csv
  - ...
  - test_clean_predictions_common.csv
  - test_masked_predictions_common.csv

Common columns:
  sample_id, clip_id, split, head, level, y_true, y_pred, prob_0...
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


HEADS = ["action", "gaze", "hands", "talk"]


def find_prob_cols(df: pd.DataFrame, head: str) -> list[str]:
    cols = [c for c in df.columns if re.fullmatch(rf"{re.escape(head)}_p\d+", str(c))]
    cols = sorted(cols, key=lambda x: int(re.findall(r"\d+", x)[-1]))
    return cols


def convert_split(run_dir: Path, split: str, overwrite: bool = True) -> None:
    src = run_dir / f"{split}_predictions.csv"
    if not src.exists():
        raise FileNotFoundError(f"missing input: {src}")

    df = pd.read_csv(src)
    all_rows = []

    for head in HEADS:
        target_col = f"{head}_target"
        pred_col = f"{head}_pred"
        prob_cols = find_prob_cols(df, head)

        if target_col not in df.columns or pred_col not in df.columns or not prob_cols:
            print(f"[SKIP] {split} {head}: missing columns target/pred/prob")
            continue

        rows = []
        for idx, r in df.iterrows():
            y_true = r[target_col]
            y_pred = r[pred_col]

            if pd.isna(y_true) or pd.isna(y_pred):
                continue

            row = {
                "sample_id": r.get("clip_id", f"{split}_{idx:08d}"),
                "clip_id": r.get("clip_id", f"{split}_{idx:08d}"),
                "split": split,
                "head": head,
                "level": "clip",
                "y_true": int(y_true),
                "y_pred": int(y_pred),
            }

            for j, c in enumerate(prob_cols):
                row[f"prob_{j}"] = float(r[c])

            # Preserve useful metadata if present.
            for meta in ["subject_key", "source", "variant", "mask_region"]:
                if meta in df.columns:
                    row[meta] = r[meta]

            rows.append(row)
            all_rows.append(row)

        out_head = run_dir / f"{split}_{head}_clip_predictions.csv"
        if out_head.exists() and not overwrite:
            print(f"[SKIP exists] {out_head}")
        else:
            pd.DataFrame(rows).to_csv(out_head, index=False, encoding="utf-8-sig")
            print(f"[saved] {out_head} rows={len(rows)}")

    out_all = run_dir / f"{split}_predictions_common.csv"
    pd.DataFrame(all_rows).to_csv(out_all, index=False, encoding="utf-8-sig")
    print(f"[saved] {out_all} rows={len(all_rows)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True, help="TSM run directory containing test_clean_predictions.csv and test_masked_predictions.csv")
    ap.add_argument("--splits", nargs="+", default=["test_clean", "test_masked"])
    ap.add_argument("--no-overwrite", action="store_true")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    print("run_dir:", run_dir)

    for split in args.splits:
        convert_split(run_dir, split, overwrite=not args.no_overwrite)

    print("[done] converted TSM predictions to common ROC/PR format")


if __name__ == "__main__":
    main()
