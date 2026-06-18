#!/usr/bin/env python3
"""
Post-process exported DMS prediction CSVs.

Inputs are per-clip prediction CSVs saved by:
  - V5: src.evaluation.eval_only_save_predictions
  - Compare: export_*_predictions.py

Outputs:
  - metrics_by_model_task_condition.csv
  - roc_pr_by_model_task_condition.csv
  - clean_masked_drop_by_model_task.csv
  - bootstrap_pairwise_<head>_<condition>.csv
  - bootstrap_samples_<head>_<condition>.csv.gz (optional)

CSV requirement:
  sample_id, y_true, y_pred, prob_0, prob_1, ...

Important:
  Bootstrap comparisons are paired: the same resampled sample_id set is applied
  to both models after taking their sample_id intersection.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

HEADS = ["action", "gaze", "hands", "talk"]
CONDITIONS = ["test_clean", "test_masked"]
IGNORE_LABEL = -100

DEFAULT_MODEL_ALIASES = {
    #"v5_no_occ_original_mediapipe_seed42_gaze045_light": "NoOcc",
    #"v5_task_gated_late_gaze045_light": "task_gated_late",
    #"v5_task_region_gated_late_gaze045_light": "task_region_gated_late",
    #"v5_task_region_scalar_gated_late_gaze045_light": "task_region_scalar_gated_late",
    #"v5_explicit_region_mask_gate_gaze045_light": "explicit_region_mask_gate",
    #"v5_explicit_region_scalar_mask_gate_gaze045_light": "explicit_region_scalar_mask_gate",
    #"v5_occ_attention_bias_gaze045_light": "attention_bias",
    #"v5_occ_token_region_transformer_gaze045_light": "occ_token_region_transformer",
    #"dfs": "dfs",
    #"dmd_original_seed42_gaze045_light": "dmd_original",
    #"driveact_seed42_gaze045_light": "DriveAct",
    #"pose_guided_seed42_gaze045_light": "pose_guided",
    #"skateformer_seed42_gaze045_light": "skateformer",
    #"spatiotemporal_seed42_gaze045_light": "spatiotemporal",
}

PROB_RE = re.compile(r"^prob_(\d+)$")


@dataclass(frozen=True)
class PredKey:
    model: str
    condition: str
    head: str
    path: Path


def infer_model_name(run_dir: Path) -> str:
    base = run_dir.name
    return DEFAULT_MODEL_ALIASES.get(base, base)


def discover_prediction_files(roots: list[Path]) -> list[PredKey]:
    out: list[PredKey] = []
    pattern = re.compile(r"^(test_clean|test_masked)_(action|gaze|hands|talk)_clip_predictions\.csv$")
    for root in roots:
        root = root.resolve()
        if not root.exists():
            continue
        for p in root.rglob("*_clip_predictions.csv"):
            m = pattern.match(p.name)
            if not m:
                continue
            condition, head = m.group(1), m.group(2)
            model = infer_model_name(p.parent)
            out.append(PredKey(model=model, condition=condition, head=head, path=p))
    # Deduplicate by model/condition/head. Prefer artifact path over copied/result path if duplicates appear.
    chosen: dict[tuple[str, str, str], PredKey] = {}
    for k in out:
        key = (k.model, k.condition, k.head)
        old = chosen.get(key)
        if old is None:
            chosen[key] = k
            continue
        # Prefer paths containing artifacts, then shorter path.
        old_score = ("artifacts" in str(old.path), -len(str(old.path)))
        new_score = ("artifacts" in str(k.path), -len(str(k.path)))
        if new_score > old_score:
            chosen[key] = k
    return sorted(chosen.values(), key=lambda x: (x.model, x.condition, x.head, str(x.path)))


def prob_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        m = PROB_RE.match(str(c))
        if m:
            cols.append((int(m.group(1)), c))
    return [c for _, c in sorted(cols)]


def load_prediction_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"sample_id", "y_true", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df = df.copy()
    df["sample_id"] = df["sample_id"].astype(str)
    df["y_true"] = df["y_true"].astype(int)
    df["y_pred"] = df["y_pred"].astype(int)
    df = df[df["y_true"] != IGNORE_LABEL].reset_index(drop=True)
    # If duplicate sample_id exists, keep first. Clip predictions should already be one row per clip/head.
    df = df.drop_duplicates(subset=["sample_id"], keep="first").reset_index(drop=True)
    return df


def metric_row(model: str, condition: str, head: str, df: pd.DataFrame) -> dict:
    y = df["y_true"].to_numpy(dtype=int)
    p = df["y_pred"].to_numpy(dtype=int)
    labels = sorted(set(y.tolist()) | set(p.tolist()))
    if len(y) == 0:
        return {"model": model, "condition": condition, "head": head, "n": 0}
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y, p, average="weighted", zero_division=0)
    return {
        "model": model,
        "condition": condition,
        "head": head,
        "n": int(len(y)),
        "n_classes_true": int(len(set(y.tolist()))),
        "accuracy": float(accuracy_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "precision_macro": float(prec_m),
        "recall_macro": float(rec_m),
        "f1_macro": float(f1_m),
        "precision_weighted": float(prec_w),
        "recall_weighted": float(rec_w),
        "f1_weighted": float(f1_w),
        "labels_present": json.dumps(labels, ensure_ascii=False),
    }


def roc_pr_row(model: str, condition: str, head: str, df: pd.DataFrame) -> dict:
    y = df["y_true"].to_numpy(dtype=int)
    prob_cols = prob_columns(df)
    base = {
        "model": model,
        "condition": condition,
        "head": head,
        "n": int(len(df)),
        "n_prob_cols": int(len(prob_cols)),
        "roc_auc_ovr_macro": np.nan,
        "roc_auc_ovr_weighted": np.nan,
        "auprc_macro": np.nan,
        "auprc_weighted": np.nan,
        "note": "",
    }
    if len(y) == 0 or not prob_cols:
        base["note"] = "empty_or_no_prob_cols"
        return base
    probs = df[prob_cols].to_numpy(dtype=float)
    n_classes = probs.shape[1]
    labels = np.arange(n_classes)
    present = np.unique(y)
    if len(present) < 2:
        base["note"] = "only_one_true_class_present"
        return base
    try:
        base["roc_auc_ovr_macro"] = float(roc_auc_score(y, probs, labels=labels, multi_class="ovr", average="macro"))
        base["roc_auc_ovr_weighted"] = float(roc_auc_score(y, probs, labels=labels, multi_class="ovr", average="weighted"))
    except Exception as e:
        base["note"] += f"roc_auc_error={type(e).__name__}: {e}; "

    try:
        y_bin = label_binarize(y, classes=labels)
        # average_precision_score can fail if some classes have no positives; compute per-class robustly.
        ap = []
        supports = []
        for c in range(n_classes):
            yt = y_bin[:, c]
            supports.append(int(yt.sum()))
            if yt.sum() == 0:
                continue
            ap.append(float(average_precision_score(yt, probs[:, c])))
        if ap:
            base["auprc_macro"] = float(np.mean(ap))
            weights = np.asarray([s for s in supports if s > 0], dtype=float)
            vals = []
            for c in range(n_classes):
                yt = y_bin[:, c]
                if yt.sum() == 0:
                    continue
                vals.append(float(average_precision_score(yt, probs[:, c])))
            if vals and weights.sum() > 0:
                base["auprc_weighted"] = float(np.average(vals, weights=weights))
    except Exception as e:
        base["note"] += f"auprc_error={type(e).__name__}: {e}; "
    return base


def compute_drop(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    idx = metrics_df.set_index(["model", "head", "condition"])
    for model in sorted(metrics_df["model"].unique()):
        for head in HEADS:
            key_c = (model, head, "test_clean")
            key_m = (model, head, "test_masked")
            if key_c not in idx.index or key_m not in idx.index:
                continue
            c = idx.loc[key_c]
            m = idx.loc[key_m]
            clean = float(c["f1_macro"])
            masked = float(m["f1_macro"])
            acc_c = float(c["accuracy"])
            acc_m = float(m["accuracy"])
            rows.append({
                "model": model,
                "head": head,
                "clean_f1_macro": clean,
                "masked_f1_macro": masked,
                "drop_f1_macro": clean - masked,
                "relative_drop_f1_macro": (clean - masked) / clean if clean > 0 else np.nan,
                "pdi_f1_percent": 100.0 * ((clean - masked) / clean) if clean > 0 else np.nan,
                "clean_accuracy": acc_c,
                "masked_accuracy": acc_m,
                "drop_accuracy": acc_c - acc_m,
                "relative_drop_accuracy": (acc_c - acc_m) / acc_c if acc_c > 0 else np.nan,
                "n_clean": int(c["n"]),
                "n_masked": int(m["n"]),
            })
    return pd.DataFrame(rows)


def f1_macro_from_arrays(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) == 0:
        return np.nan
    return float(f1_score(y, pred, average="macro", zero_division=0))


def accuracy_from_arrays(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) == 0:
        return np.nan
    return float(accuracy_score(y, pred))


def paired_bootstrap(
    dfs: dict[str, pd.DataFrame],
    proposed: str,
    baselines: list[str],
    condition: str,
    head: str,
    n_boot: int,
    seed: int,
    save_samples: bool,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    rng = np.random.default_rng(seed)
    rows = []
    sample_rows = []

    key_p = f"{proposed}::{condition}::{head}"
    if key_p not in dfs:
        raise KeyError(f"Proposed predictions not found: {key_p}")

    p_df0 = dfs[key_p]

    for baseline in baselines:
        key_b = f"{baseline}::{condition}::{head}"
        if key_b not in dfs:
            print(f"[bootstrap][skip] baseline missing: {key_b}")
            continue

        b_df0 = dfs[key_b]
        common = sorted(set(p_df0["sample_id"]).intersection(set(b_df0["sample_id"])))
        if len(common) < 5:
            print(f"[bootstrap][skip] too few common samples: {proposed} vs {baseline}, n={len(common)}")
            continue

        p_df = p_df0.set_index("sample_id").loc[common].reset_index()
        b_df = b_df0.set_index("sample_id").loc[common].reset_index()

        # Enforce identical ground truth on intersected ids.
        same_y = (p_df["y_true"].to_numpy(dtype=int) == b_df["y_true"].to_numpy(dtype=int))
        if not np.all(same_y):
            bad = int((~same_y).sum())
            raise ValueError(f"Ground-truth mismatch for {proposed} vs {baseline}, condition={condition}, head={head}, mismatches={bad}")

        y = p_df["y_true"].to_numpy(dtype=int)
        pred_p = p_df["y_pred"].to_numpy(dtype=int)
        pred_b = b_df["y_pred"].to_numpy(dtype=int)
        n = len(common)

        deltas_f1 = np.empty(n_boot, dtype=float)
        deltas_acc = np.empty(n_boot, dtype=float)
        prop_vals = np.empty(n_boot, dtype=float)
        base_vals = np.empty(n_boot, dtype=float)

        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            yp = y[idx]
            pp = pred_p[idx]
            pb = pred_b[idx]
            f1p = f1_macro_from_arrays(yp, pp)
            f1b = f1_macro_from_arrays(yp, pb)
            accp = accuracy_from_arrays(yp, pp)
            accb = accuracy_from_arrays(yp, pb)
            prop_vals[i] = f1p
            base_vals[i] = f1b
            deltas_f1[i] = f1p - f1b
            deltas_acc[i] = accp - accb
            if save_samples:
                sample_rows.append({
                    "proposed": proposed,
                    "baseline": baseline,
                    "condition": condition,
                    "head": head,
                    "iter": i,
                    "proposed_f1_macro": f1p,
                    "baseline_f1_macro": f1b,
                    "delta_f1_macro": f1p - f1b,
                    "proposed_accuracy": accp,
                    "baseline_accuracy": accb,
                    "delta_accuracy": accp - accb,
                })

        rows.append({
            "proposed": proposed,
            "baseline": baseline,
            "condition": condition,
            "head": head,
            "n_common": n,
            "n_boot": n_boot,
            "proposed_observed_f1_macro": f1_macro_from_arrays(y, pred_p),
            "baseline_observed_f1_macro": f1_macro_from_arrays(y, pred_b),
            "observed_delta_f1_macro": f1_macro_from_arrays(y, pred_p) - f1_macro_from_arrays(y, pred_b),
            "mean_delta_f1_macro": float(np.mean(deltas_f1)),
            "ci95_low_delta_f1_macro": float(np.quantile(deltas_f1, 0.025)),
            "ci95_high_delta_f1_macro": float(np.quantile(deltas_f1, 0.975)),
            "win_rate_delta_f1_gt0": float(np.mean(deltas_f1 > 0)),
            "p_like_delta_f1_le0": float(np.mean(deltas_f1 <= 0)),
            "mean_proposed_f1_macro": float(np.mean(prop_vals)),
            "mean_baseline_f1_macro": float(np.mean(base_vals)),
            "mean_delta_accuracy": float(np.mean(deltas_acc)),
            "ci95_low_delta_accuracy": float(np.quantile(deltas_acc, 0.025)),
            "ci95_high_delta_accuracy": float(np.quantile(deltas_acc, 0.975)),
        })

    return pd.DataFrame(rows), (pd.DataFrame(sample_rows) if save_samples else None)


def bootstrap_relative_drop(
    dfs: dict[str, pd.DataFrame],
    proposed: str,
    baselines: list[str],
    head: str,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1000003)
    rows = []

    p_clean_key = f"{proposed}::test_clean::{head}"
    p_mask_key = f"{proposed}::test_masked::{head}"
    if p_clean_key not in dfs or p_mask_key not in dfs:
        return pd.DataFrame()

    for baseline in baselines:
        b_clean_key = f"{baseline}::test_clean::{head}"
        b_mask_key = f"{baseline}::test_masked::{head}"
        if b_clean_key not in dfs or b_mask_key not in dfs:
            continue

        # Use the intersection of clean sample IDs and masked sample IDs separately.
        # Clean and masked ids may differ by variant suffix. This estimates each relative drop
        # using the same bootstrap iteration seed and size per condition.
        pc = dfs[p_clean_key]
        pm = dfs[p_mask_key]
        bc = dfs[b_clean_key]
        bm = dfs[b_mask_key]

        common_c = sorted(set(pc["sample_id"]).intersection(set(bc["sample_id"])))
        common_m = sorted(set(pm["sample_id"]).intersection(set(bm["sample_id"])))
        if len(common_c) < 5 or len(common_m) < 5:
            continue

        pc = pc.set_index("sample_id").loc[common_c].reset_index()
        bc = bc.set_index("sample_id").loc[common_c].reset_index()
        pm = pm.set_index("sample_id").loc[common_m].reset_index()
        bm = bm.set_index("sample_id").loc[common_m].reset_index()

        yc = pc["y_true"].to_numpy(dtype=int)
        ym = pm["y_true"].to_numpy(dtype=int)
        pc_pred = pc["y_pred"].to_numpy(dtype=int)
        pm_pred = pm["y_pred"].to_numpy(dtype=int)
        bc_pred = bc["y_pred"].to_numpy(dtype=int)
        bm_pred = bm["y_pred"].to_numpy(dtype=int)

        n_c = len(common_c)
        n_m = len(common_m)
        delta_rel = np.empty(n_boot, dtype=float)
        prop_rel = np.empty(n_boot, dtype=float)
        base_rel = np.empty(n_boot, dtype=float)

        for i in range(n_boot):
            ic = rng.integers(0, n_c, size=n_c)
            im = rng.integers(0, n_m, size=n_m)

            p_clean = f1_macro_from_arrays(yc[ic], pc_pred[ic])
            p_mask = f1_macro_from_arrays(ym[im], pm_pred[im])
            b_clean = f1_macro_from_arrays(yc[ic], bc_pred[ic])
            b_mask = f1_macro_from_arrays(ym[im], bm_pred[im])

            pr = (p_clean - p_mask) / p_clean if p_clean > 0 else np.nan
            br = (b_clean - b_mask) / b_clean if b_clean > 0 else np.nan
            prop_rel[i] = pr
            base_rel[i] = br
            # Positive means proposed has lower relative drop than baseline.
            delta_rel[i] = br - pr

        valid = np.isfinite(delta_rel)
        if not np.any(valid):
            continue
        d = delta_rel[valid]
        rows.append({
            "proposed": proposed,
            "baseline": baseline,
            "head": head,
            "n_common_clean": len(common_c),
            "n_common_masked": len(common_m),
            "n_boot": n_boot,
            "mean_delta_relative_drop_baseline_minus_proposed": float(np.mean(d)),
            "ci95_low": float(np.quantile(d, 0.025)),
            "ci95_high": float(np.quantile(d, 0.975)),
            "win_rate_lower_drop": float(np.mean(d > 0)),
            "p_like_delta_le0": float(np.mean(d <= 0)),
            "mean_proposed_relative_drop": float(np.nanmean(prop_rel)),
            "mean_baseline_relative_drop": float(np.nanmean(base_rel)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", type=Path, required=True,
                    help="Roots to search recursively, e.g. V5 root and Compare root.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--proposed", default="task_region_scalar_gated_late")
    ap.add_argument("--baselines", nargs="*", default=["NoOcc", "dmd_original", "DriveAct", "task_gated_late", "attention_bias", "spatiotemporal", "pose_guided"])
    ap.add_argument("--bootstrap-head", default="gaze", choices=HEADS)
    ap.add_argument("--bootstrap-condition", default="test_masked", choices=CONDITIONS)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-bootstrap-samples", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pred_files = discover_prediction_files(args.roots)
    if not pred_files:
        raise SystemExit("No prediction files found. Run export first.")

    manifest_rows = []
    dfs: dict[str, pd.DataFrame] = {}
    metric_rows = []
    roc_rows = []

    for k in pred_files:
        df = load_prediction_csv(k.path)
        dict_key = f"{k.model}::{k.condition}::{k.head}"
        dfs[dict_key] = df
        manifest_rows.append({"model": k.model, "condition": k.condition, "head": k.head, "path": str(k.path), "n": len(df)})
        metric_rows.append(metric_row(k.model, k.condition, k.head, df))
        roc_rows.append(roc_pr_row(k.model, k.condition, k.head, df))

    manifest_df = pd.DataFrame(manifest_rows).sort_values(["model", "condition", "head"])
    manifest_df.to_csv(args.out_dir / "prediction_manifest.csv", index=False, encoding="utf-8-sig")

    metrics_df = pd.DataFrame(metric_rows).sort_values(["head", "condition", "f1_macro"], ascending=[True, True, False])
    metrics_df.to_csv(args.out_dir / "metrics_by_model_task_condition.csv", index=False, encoding="utf-8-sig")

    roc_df = pd.DataFrame(roc_rows).sort_values(["head", "condition", "roc_auc_ovr_macro"], ascending=[True, True, False])
    roc_df.to_csv(args.out_dir / "roc_pr_by_model_task_condition.csv", index=False, encoding="utf-8-sig")

    drop_df = compute_drop(metrics_df)
    if not drop_df.empty:
        drop_df = drop_df.sort_values(["head", "masked_f1_macro"], ascending=[True, False])
        drop_df.to_csv(args.out_dir / "clean_masked_drop_by_model_task.csv", index=False, encoding="utf-8-sig")

    boot_df, samples_df = paired_bootstrap(
        dfs=dfs,
        proposed=args.proposed,
        baselines=args.baselines,
        condition=args.bootstrap_condition,
        head=args.bootstrap_head,
        n_boot=args.n_boot,
        seed=args.seed,
        save_samples=args.save_bootstrap_samples,
    )
    boot_path = args.out_dir / f"bootstrap_pairwise_{args.bootstrap_head}_{args.bootstrap_condition}.csv"
    boot_df.to_csv(boot_path, index=False, encoding="utf-8-sig")

    if samples_df is not None and not samples_df.empty:
        samples_df.to_csv(args.out_dir / f"bootstrap_samples_{args.bootstrap_head}_{args.bootstrap_condition}.csv.gz", index=False, compression="gzip")

    rel_df = bootstrap_relative_drop(
        dfs=dfs,
        proposed=args.proposed,
        baselines=args.baselines,
        head=args.bootstrap_head,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    rel_path = args.out_dir / f"bootstrap_relative_drop_{args.bootstrap_head}.csv"
    rel_df.to_csv(rel_path, index=False, encoding="utf-8-sig")

    print("[saved]", args.out_dir)
    print("  - prediction_manifest.csv")
    print("  - metrics_by_model_task_condition.csv")
    print("  - roc_pr_by_model_task_condition.csv")
    print("  - clean_masked_drop_by_model_task.csv")
    print("  -", boot_path.name)
    print("  -", rel_path.name)


if __name__ == "__main__":
    main()
