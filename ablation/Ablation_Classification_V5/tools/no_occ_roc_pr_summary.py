"""Generate ROC/PR summary CSV for no_occ_original prediction exports."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

HEADS = ["action", "gaze", "hands", "talk"]
SPLITS = ["test_clean", "test_masked"]


def read_pred(path: Path, head: str | None = None):
    df = pd.read_csv(path)
    if head is not None and "head" in df.columns:
        df = df[df["head"].astype(str) == head].copy()
    prob_cols = [c for c in df.columns if re.match(r"^prob_?\d+$", c)]
    prob_cols = sorted(prob_cols, key=lambda x: int(re.findall(r"\d+", x)[-1]))
    if not prob_cols:
        raise ValueError(f"No prob_* cols in {path}")
    return df["y_true"].to_numpy(int), df[prob_cols].to_numpy(float)


def aucs(y_true, y_score):
    n_cls = y_score.shape[1]
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    if n_cls == 2:
        s = y_score[:, 1]
        return float(roc_auc_score(y_true, s)), float(average_precision_score(y_true, s))
    y_bin = label_binarize(y_true, classes=list(range(n_cls)))
    auroc = roc_auc_score(y_bin, y_score, average="macro", multi_class="ovr")
    auprc = average_precision_score(y_bin, y_score, average="macro")
    return float(auroc), float(auprc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=Path("/data/shared/scuppy/hyi/Classification_model_V5_transformer/artifacts/v5_no_occ_original_mediapipe_seed42"))
    args = ap.parse_args()
    run_dir = args.run_dir

    rows = []
    for split in SPLITS:
        for head in HEADS:
            p = run_dir / f"{split}_{head}_clip_predictions.csv"
            if not p.exists():
                p = run_dir / f"{split}_predictions.csv"
            if not p.exists():
                print("[SKIP missing]", split, head)
                continue
            y, score = read_pred(p, head=head)
            auroc, auprc = aucs(y, score)
            rows.append({"model": run_dir.name, "split": split, "head": head, "auroc": auroc, "auprc": auprc, "n": len(y), "file": str(p)})

    df = pd.DataFrame(rows)
    out = run_dir / "no_occ_roc_pr_auc_summary.csv"
    df.to_csv(out, index=False)
    print(df)
    print("saved:", out)

    if len(df) > 0:
        pivot = df.pivot_table(index=["model", "head"], columns="split", values=["auroc", "auprc"], aggfunc="mean")
        pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
        pivot = pivot.reset_index()
        if "auroc_test_clean" in pivot and "auroc_test_masked" in pivot:
            pivot["auroc_drop"] = pivot["auroc_test_clean"] - pivot["auroc_test_masked"]
            pivot["auroc_pdi_percent"] = 100 * pivot["auroc_drop"] / pivot["auroc_test_clean"].replace(0, np.nan)
        if "auprc_test_clean" in pivot and "auprc_test_masked" in pivot:
            pivot["auprc_drop"] = pivot["auprc_test_clean"] - pivot["auprc_test_masked"]
            pivot["auprc_pdi_percent"] = 100 * pivot["auprc_drop"] / pivot["auprc_test_clean"].replace(0, np.nan)
        out2 = run_dir / "no_occ_roc_pr_drop_summary.csv"
        pivot.to_csv(out2, index=False)
        print(pivot)
        print("saved:", out2)


if __name__ == "__main__":
    main()
