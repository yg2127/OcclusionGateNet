#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bootstrap / metric post-processing from exported DMS prediction CSVs.

This version handles both sample_id formats:

1) V5-style:
   gA_5_s6_...__clean__gaze__00000
   gA_5_s6_...__masked__gaze__00000

2) Compare-style:
   clean::clean::clean::gA_5_s6_...__gaze__00000
   masked::both_eyes::soft_noise::gA_5_s6_...__gaze__00000

Canonical rule:
  - If "::" exists, keep only the last "::" segment.
  - Remove "__clean__" and "__masked__".
  - Use this canonical_id for all cross-model and clean/masked matching.

Outputs:
  metrics_summary.csv
  roc_pr_summary.csv
  clean_masked_drop_summary.csv
  bootstrap_pairwise_<head>_<condition>.csv
  bootstrap_relative_drop_<head>.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


HEADS = ["action", "gaze", "hands", "talk"]
CONDITIONS = ["test_clean", "test_masked"]


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--pred-root", required=True, type=str)
    p.add_argument("--out-dir", required=True, type=str)

    p.add_argument("--proposed", required=True, type=str)
    p.add_argument("--baselines", nargs="+", required=True)

    p.add_argument("--head", default="gaze", choices=HEADS)
    p.add_argument("--condition", default="test_masked", choices=CONDITIONS)
    p.add_argument("--level", default="clip")

    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def pred_path(pred_root: Path, model: str, condition: str, head: str, level: str) -> Path:
    model_dir = pred_root / model

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    candidates = [
        model_dir / f"{condition}_{head}_{level}_predictions.csv",
        model_dir / f"{condition}_{head}_predictions.csv",
        model_dir / f"{condition}_predictions.csv",
    ]

    for p in candidates:
        if p.exists():
            return p

    hits = list(model_dir.rglob(f"{condition}_{head}_{level}_predictions.csv"))
    if hits:
        return hits[0]

    hits = list(model_dir.rglob(f"{condition}_{head}*predictions.csv"))
    if hits:
        return hits[0]

    raise FileNotFoundError(
        f"Prediction CSV not found: model={model}, condition={condition}, "
        f"head={head}, level={level}, dir={model_dir}"
    )


def make_canonical_id_one(x: str) -> str:
    """
    Normalize sample_id across V5 and Compare exports.

    V5:
      gA...__clean__gaze__00000
      gA...__masked__gaze__00000
      -> gA...__gaze__00000

    Compare:
      clean::clean::clean::gA...__gaze__00000
      masked::both_eyes::soft_noise::gA...__gaze__00000
      -> gA...__gaze__00000
    """
    s = str(x)

    if "::" in s:
        s = s.split("::")[-1]

    s = s.replace("__clean__", "__")
    s = s.replace("__masked__", "__")

    return s


def make_canonical_id(sample_id: pd.Series) -> pd.Series:
    return sample_id.astype(str).map(make_canonical_id_one)


def load_pred(pred_root: Path, model: str, condition: str, head: str, level: str) -> pd.DataFrame:
    path = pred_path(pred_root, model, condition, head, level)
    df = pd.read_csv(path)

    required = {"sample_id", "y_true", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    df = df.copy()

    df["sample_id"] = df["sample_id"].astype(str)
    df["canonical_id"] = make_canonical_id(df["sample_id"])

    df["y_true"] = df["y_true"].astype(int)
    df["y_pred"] = df["y_pred"].astype(int)

    df = df[df["y_true"] != -100].reset_index(drop=True)

    # Deduplicate by canonical_id, not raw sample_id.
    # This is safer after removing clean/masked or prefix strings.
    df = df.drop_duplicates("canonical_id", keep="first").reset_index(drop=True)

    return df


def f1_macro(df: pd.DataFrame, idx: np.ndarray | None = None) -> float:
    if idx is None:
        y = df["y_true"].to_numpy()
        p = df["y_pred"].to_numpy()
    else:
        y = df["y_true"].to_numpy()[idx]
        p = df["y_pred"].to_numpy()[idx]

    return float(f1_score(y, p, average="macro", zero_division=0))


def f1_weighted(df: pd.DataFrame) -> float:
    y = df["y_true"].to_numpy()
    p = df["y_pred"].to_numpy()
    return float(f1_score(y, p, average="weighted", zero_division=0))


def acc(df: pd.DataFrame, idx: np.ndarray | None = None) -> float:
    if idx is None:
        y = df["y_true"].to_numpy()
        p = df["y_pred"].to_numpy()
    else:
        y = df["y_true"].to_numpy()[idx]
        p = df["y_pred"].to_numpy()[idx]

    return float(accuracy_score(y, p))


def prob_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if str(c).startswith("prob_")]

    def key_fn(x: str):
        try:
            return int(str(x).split("_")[-1])
        except Exception:
            return 10**9

    return sorted(cols, key=key_fn)


def roc_pr(df: pd.DataFrame) -> tuple[float, float]:
    cols = prob_cols(df)
    if not cols:
        return np.nan, np.nan

    y = df["y_true"].to_numpy()
    probs = df[cols].to_numpy(dtype=float)
    n_classes = probs.shape[1]

    if len(np.unique(y)) < 2:
        return np.nan, np.nan

    try:
        if n_classes == 2:
            auroc = roc_auc_score(y, probs[:, 1])
            auprc = average_precision_score(y, probs[:, 1])
            return float(auroc), float(auprc)

        labels = np.arange(n_classes)

        auroc = roc_auc_score(
            y,
            probs,
            labels=labels,
            multi_class="ovr",
            average="macro",
        )

        y_bin = label_binarize(y, classes=labels)
        aps = []

        for k in range(n_classes):
            if y_bin[:, k].sum() == 0:
                continue
            aps.append(average_precision_score(y_bin[:, k], probs[:, k]))

        auprc = float(np.mean(aps)) if aps else np.nan
        return float(auroc), float(auprc)

    except Exception:
        return np.nan, np.nan


def align_pair(
    a: pd.DataFrame,
    b: pd.DataFrame,
    key_col: str = "canonical_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align two prediction DataFrames.

    We use canonical_id for both:
      - same-condition model comparison
      - clean/masked pairing

    Reason:
      Some Compare models use prefixes like
      masked::both_eyes::soft_noise::...
      while V5 models do not.
    """
    if key_col not in a.columns:
        raise ValueError(f"Missing key_col={key_col} in first DataFrame")
    if key_col not in b.columns:
        raise ValueError(f"Missing key_col={key_col} in second DataFrame")

    common = sorted(set(a[key_col]) & set(b[key_col]))

    if len(common) < 5:
        raise ValueError(f"Too few common samples using {key_col}: {len(common)}")

    a2 = a.set_index(key_col).loc[common].reset_index()
    b2 = b.set_index(key_col).loc[common].reset_index()

    ya = a2["y_true"].to_numpy()
    yb = b2["y_true"].to_numpy()

    if not np.all(ya == yb):
        bad = int(np.sum(ya != yb))
        raise ValueError(f"y_true mismatch after {key_col} alignment: {bad}")

    return a2, b2


def paired_bootstrap_condition(
    prop_df: pd.DataFrame,
    base_df: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> dict:
    """
    Paired bootstrap for same condition.
    Uses canonical_id.
    """
    p, b = align_pair(prop_df, base_df, key_col="canonical_id")

    n = len(p)
    rng = np.random.default_rng(seed)

    y = p["y_true"].to_numpy()
    pred_p = p["y_pred"].to_numpy()
    pred_b = b["y_pred"].to_numpy()

    prop_obs = float(f1_score(y, pred_p, average="macro", zero_division=0))
    base_obs = float(f1_score(y, pred_b, average="macro", zero_division=0))
    obs_delta = prop_obs - base_obs

    deltas_f1 = np.empty(n_boot, dtype=float)
    deltas_acc = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)

        f_prop = f1_score(y[idx], pred_p[idx], average="macro", zero_division=0)
        f_base = f1_score(y[idx], pred_b[idx], average="macro", zero_division=0)

        a_prop = accuracy_score(y[idx], pred_p[idx])
        a_base = accuracy_score(y[idx], pred_b[idx])

        deltas_f1[i] = f_prop - f_base
        deltas_acc[i] = a_prop - a_base

    return {
        "n_common": int(n),
        "proposed_observed_f1_macro": prop_obs,
        "baseline_observed_f1_macro": base_obs,
        "observed_delta_f1_macro": obs_delta,
        "mean_delta_f1_macro": float(np.mean(deltas_f1)),
        "std_delta_f1_macro": float(np.std(deltas_f1, ddof=1)),
        "ci95_low_delta_f1_macro": float(np.percentile(deltas_f1, 2.5)),
        "ci95_high_delta_f1_macro": float(np.percentile(deltas_f1, 97.5)),
        "win_rate_delta_f1_gt0": float(np.mean(deltas_f1 > 0)),
        "p_like_delta_f1_le0": float(np.mean(deltas_f1 <= 0)),
        "mean_delta_accuracy": float(np.mean(deltas_acc)),
        "ci95_low_delta_accuracy": float(np.percentile(deltas_acc, 2.5)),
        "ci95_high_delta_accuracy": float(np.percentile(deltas_acc, 97.5)),
    }


def observed_clean_masked_drop(
    clean_df: pd.DataFrame,
    masked_df: pd.DataFrame,
) -> tuple[float, float, float, int]:
    """
    Observed clean/masked relative drop.
    Uses canonical_id.
    """
    c, m = align_pair(clean_df, masked_df, key_col="canonical_id")

    cf = f1_macro(c)
    mf = f1_macro(m)
    drop = cf - mf
    rel = drop / cf if cf > 0 else np.nan

    return cf, mf, rel, len(c)


def bootstrap_relative_drop(
    prop_clean: pd.DataFrame,
    prop_masked: pd.DataFrame,
    base_clean: pd.DataFrame,
    base_masked: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> dict:
    """
    Bootstrap difference of relative drop.

    Positive delta means:
      baseline relative drop > proposed relative drop
      => proposed is more robust.

    Uses canonical_id across all four files.
    """

    pc, pm = align_pair(prop_clean, prop_masked, key_col="canonical_id")
    bc, bm = align_pair(base_clean, base_masked, key_col="canonical_id")

    common = sorted(set(pc["canonical_id"]) & set(bc["canonical_id"]))

    if len(common) < 5:
        raise ValueError(f"Too few common clean/masked canonical pairs: {len(common)}")

    pc = pc.set_index("canonical_id").loc[common].reset_index()
    pm = pm.set_index("canonical_id").loc[common].reset_index()
    bc = bc.set_index("canonical_id").loc[common].reset_index()
    bm = bm.set_index("canonical_id").loc[common].reset_index()

    y_pc = pc["y_true"].to_numpy()
    y_pm = pm["y_true"].to_numpy()
    y_bc = bc["y_true"].to_numpy()
    y_bm = bm["y_true"].to_numpy()

    if not (np.all(y_pc == y_pm) and np.all(y_pc == y_bc) and np.all(y_pc == y_bm)):
        raise ValueError("y_true mismatch among proposed/baseline clean/masked after canonical_id alignment")

    y = y_pc
    n = len(y)

    pc_pred = pc["y_pred"].to_numpy()
    pm_pred = pm["y_pred"].to_numpy()
    bc_pred = bc["y_pred"].to_numpy()
    bm_pred = bm["y_pred"].to_numpy()

    rng = np.random.default_rng(seed + 100003)

    deltas = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)

        p_clean_f1 = f1_score(y[idx], pc_pred[idx], average="macro", zero_division=0)
        p_mask_f1 = f1_score(y[idx], pm_pred[idx], average="macro", zero_division=0)

        b_clean_f1 = f1_score(y[idx], bc_pred[idx], average="macro", zero_division=0)
        b_mask_f1 = f1_score(y[idx], bm_pred[idx], average="macro", zero_division=0)

        prop_rel = (p_clean_f1 - p_mask_f1) / p_clean_f1 if p_clean_f1 > 0 else np.nan
        base_rel = (b_clean_f1 - b_mask_f1) / b_clean_f1 if b_clean_f1 > 0 else np.nan

        deltas[i] = base_rel - prop_rel

    valid = np.isfinite(deltas)
    d = deltas[valid]

    prop_clean_obs = f1_score(y, pc_pred, average="macro", zero_division=0)
    prop_mask_obs = f1_score(y, pm_pred, average="macro", zero_division=0)
    base_clean_obs = f1_score(y, bc_pred, average="macro", zero_division=0)
    base_mask_obs = f1_score(y, bm_pred, average="macro", zero_division=0)

    prop_rel_obs = (prop_clean_obs - prop_mask_obs) / prop_clean_obs if prop_clean_obs > 0 else np.nan
    base_rel_obs = (base_clean_obs - base_mask_obs) / base_clean_obs if base_clean_obs > 0 else np.nan

    return {
        "n_common_pairs": int(n),
        "proposed_observed_clean_f1_macro": float(prop_clean_obs),
        "proposed_observed_masked_f1_macro": float(prop_mask_obs),
        "baseline_observed_clean_f1_macro": float(base_clean_obs),
        "baseline_observed_masked_f1_macro": float(base_mask_obs),
        "proposed_observed_relative_drop": float(prop_rel_obs),
        "baseline_observed_relative_drop": float(base_rel_obs),
        "observed_delta_relative_drop_baseline_minus_proposed": float(base_rel_obs - prop_rel_obs),
        "mean_delta_relative_drop_baseline_minus_proposed": float(np.mean(d)),
        "std_delta_relative_drop_baseline_minus_proposed": float(np.std(d, ddof=1)),
        "ci95_low_delta_relative_drop_baseline_minus_proposed": float(np.percentile(d, 2.5)),
        "ci95_high_delta_relative_drop_baseline_minus_proposed": float(np.percentile(d, 97.5)),
        "win_rate_lower_drop": float(np.mean(d > 0)),
        "p_like_delta_le0": float(np.mean(d <= 0)),
    }


def main():
    args = parse_args()

    pred_root = Path(args.pred_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [args.proposed] + args.baselines

    cache = {}
    metric_rows = []
    roc_rows = []
    drop_rows = []

    print("=" * 80)
    print("[LOAD] prediction CSVs")
    print(f"pred_root = {pred_root}")
    print(f"out_dir   = {out_dir}")
    print("=" * 80)

    for model in models:
        for condition in CONDITIONS:
            for head in HEADS:
                try:
                    df = load_pred(pred_root, model, condition, head, args.level)
                except FileNotFoundError:
                    continue

                cache[(model, condition, head)] = df

                metric_rows.append({
                    "model": model,
                    "condition": condition,
                    "head": head,
                    "n": int(len(df)),
                    "accuracy": acc(df),
                    "f1_macro": f1_macro(df),
                    "f1_weighted": f1_weighted(df),
                    "n_prob_cols": len(prob_cols(df)),
                })

                auroc, auprc = roc_pr(df)
                roc_rows.append({
                    "model": model,
                    "condition": condition,
                    "head": head,
                    "n": int(len(df)),
                    "auroc_ovr_macro": auroc,
                    "auprc_macro": auprc,
                })

    metrics_df = pd.DataFrame(metric_rows)
    roc_df = pd.DataFrame(roc_rows)

    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values(
            ["head", "condition", "f1_macro"],
            ascending=[True, True, False],
        )

    if not roc_df.empty:
        roc_df = roc_df.sort_values(
            ["head", "condition", "auroc_ovr_macro"],
            ascending=[True, True, False],
        )

    metrics_path = out_dir / "metrics_summary.csv"
    roc_path = out_dir / "roc_pr_summary.csv"

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    roc_df.to_csv(roc_path, index=False, encoding="utf-8-sig")

    print(f"[SAVED] {metrics_path}")
    print(f"[SAVED] {roc_path}")

    # Clean/masked drop summary
    print("=" * 80)
    print("[DROP] clean/masked drop summary using canonical_id")
    print("=" * 80)

    for model in models:
        for head in HEADS:
            kc = (model, "test_clean", head)
            km = (model, "test_masked", head)

            if kc not in cache or km not in cache:
                continue

            try:
                cf, mf, rel, n_pair = observed_clean_masked_drop(cache[kc], cache[km])
            except ValueError as e:
                print(f"[DROP][SKIP] model={model}, head={head}: {e}")
                continue

            drop = cf - mf

            drop_rows.append({
                "model": model,
                "head": head,
                "clean_f1_macro": cf,
                "masked_f1_macro": mf,
                "drop_f1_macro": drop,
                "relative_drop_f1_macro": rel,
                "pdi_f1_percent": rel * 100 if np.isfinite(rel) else np.nan,
                "n_common_pairs": n_pair,
            })

    drop_df = pd.DataFrame(drop_rows)

    if not drop_df.empty:
        drop_df = drop_df.sort_values(["head", "masked_f1_macro"], ascending=[True, False])

    drop_path = out_dir / "clean_masked_drop_summary.csv"
    drop_df.to_csv(drop_path, index=False, encoding="utf-8-sig")

    print(f"[SAVED] {drop_path}")

    # Pairwise bootstrap for selected head/condition
    print("=" * 80)
    print("[BOOTSTRAP] pairwise condition comparison using canonical_id")
    print("=" * 80)

    prop_key = (args.proposed, args.condition, args.head)

    if prop_key not in cache:
        raise FileNotFoundError(f"Missing proposed prediction: {prop_key}")

    boot_rows = []

    for baseline in args.baselines:
        base_key = (baseline, args.condition, args.head)

        if base_key not in cache:
            print(f"[BOOT][SKIP] missing baseline prediction: {base_key}")
            continue

        try:
            r = paired_bootstrap_condition(
                cache[prop_key],
                cache[base_key],
                n_boot=args.n_boot,
                seed=args.seed,
            )
        except ValueError as e:
            print(f"[BOOT][SKIP] baseline={baseline}: {e}")
            continue

        boot_rows.append({
            "proposed": args.proposed,
            "baseline": baseline,
            "condition": args.condition,
            "head": args.head,
            "n_boot": args.n_boot,
            **r,
        })

    boot_df = pd.DataFrame(boot_rows)
    boot_path = out_dir / f"bootstrap_pairwise_{args.head}_{args.condition}.csv"
    boot_df.to_csv(boot_path, index=False, encoding="utf-8-sig")

    print(f"[SAVED] {boot_path}")

    # Relative drop bootstrap for selected head
    print("=" * 80)
    print("[BOOTSTRAP] relative drop using canonical_id")
    print("=" * 80)

    rel_rows = []

    pc_key = (args.proposed, "test_clean", args.head)
    pm_key = (args.proposed, "test_masked", args.head)

    if pc_key in cache and pm_key in cache:
        for baseline in args.baselines:
            bc_key = (baseline, "test_clean", args.head)
            bm_key = (baseline, "test_masked", args.head)

            if bc_key not in cache or bm_key not in cache:
                print(f"[REL][SKIP] missing clean/masked baseline prediction: {baseline}")
                continue

            try:
                r = bootstrap_relative_drop(
                    cache[pc_key],
                    cache[pm_key],
                    cache[bc_key],
                    cache[bm_key],
                    n_boot=args.n_boot,
                    seed=args.seed,
                )
            except ValueError as e:
                print(f"[REL][SKIP] baseline={baseline}: {e}")
                continue

            rel_rows.append({
                "proposed": args.proposed,
                "baseline": baseline,
                "head": args.head,
                "n_boot": args.n_boot,
                **r,
            })

    rel_df = pd.DataFrame(rel_rows)
    rel_path = out_dir / f"bootstrap_relative_drop_{args.head}.csv"
    rel_df.to_csv(rel_path, index=False, encoding="utf-8-sig")

    print(f"[SAVED] {rel_path}")

    print("=" * 80)
    print("[DONE]")
    print(f"saved to: {out_dir}")
    print("- metrics_summary.csv")
    print("- roc_pr_summary.csv")
    print("- clean_masked_drop_summary.csv")
    print(f"- {boot_path.name}")
    print(f"- {rel_path.name}")
    print("=" * 80)


if __name__ == "__main__":
    main()