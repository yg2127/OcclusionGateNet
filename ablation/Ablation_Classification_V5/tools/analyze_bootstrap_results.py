#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze bootstrap_raw_by_seed.csv.

Input:
  bootstrap_results_seed42/bootstrap_raw_by_seed.csv

Outputs:
  bootstrap_analysis/
    01_overall_model_summary.csv
    02_task_model_summary.csv
    03_best_model_by_task_masked_f1.csv
    04_best_model_by_task_pdi.csv
    05_model_pairwise_vs_best_gaze.csv
    plots/
      task_metric_boxplots_*.png
      gaze_masked_f1_distribution.png
      gaze_pdi_distribution.png
      weighted_overall_summary.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_ROOT = Path("/data/shared/scuppy/hyi/Classification_model_V5_transformer")
DEFAULT_IN = DEFAULT_ROOT / "bootstrap_results_seed42" / "bootstrap_raw_by_seed.csv"
DEFAULT_OUT = DEFAULT_ROOT / "bootstrap_results_seed42" / "bootstrap_analysis"


TASKS = ["action", "gaze", "hands", "talk"]


def ci_low(x):
    return np.percentile(x.dropna(), 2.5)


def ci_high(x):
    return np.percentile(x.dropna(), 97.5)


def summarize_by_model_task(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "clean_f1_macro",
        "masked_f1_macro",
        "drop_f1_macro",
        "pdi_f1_macro",
        "clean_f1_weighted",
        "masked_f1_weighted",
        "drop_f1_weighted",
        "pdi_f1_weighted",
        "clean_acc",
        "masked_acc",
        "drop_acc",
    ]

    agg_dict = {}

    for m in metrics:
        agg_dict[m] = ["mean", "std", ci_low, ci_high]

    summary = df.groupby(["model", "task"]).agg(agg_dict)
    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary.columns
    ]
    summary = summary.reset_index()

    # 보기 편하게 이름 변경
    summary = summary.rename(
        columns={
            "masked_f1_macro_ci_low": "masked_f1_macro_ci95_low",
            "masked_f1_macro_ci_high": "masked_f1_macro_ci95_high",
            "pdi_f1_macro_ci_low": "pdi_f1_macro_ci95_low",
            "pdi_f1_macro_ci_high": "pdi_f1_macro_ci95_high",
        }
    )

    return summary


def make_overall_summary(task_summary: pd.DataFrame) -> pd.DataFrame:
    """
    task별 평균을 다시 모델 단위로 평균.
    발표용 rough overview.
    """
    metric_cols = [
        c for c in task_summary.columns
        if c not in ["model", "task"]
    ]

    overall = (
        task_summary
        .groupby("model")[metric_cols]
        .mean(numeric_only=True)
        .reset_index()
    )

    # 종합 ranking용 컬럼
    # masked F1은 높을수록 좋고, PDI/drop은 낮을수록 좋음.
    overall["rank_masked_f1_macro"] = overall["masked_f1_macro_mean"].rank(
        ascending=False, method="min"
    )
    overall["rank_pdi_f1_macro"] = overall["pdi_f1_macro_mean"].rank(
        ascending=True, method="min"
    )
    overall["rank_drop_f1_macro"] = overall["drop_f1_macro_mean"].rank(
        ascending=True, method="min"
    )

    # 단순 robust score: masked_f1 - drop
    overall["robust_score_macro"] = (
        overall["masked_f1_macro_mean"] - overall["drop_f1_macro_mean"]
    )
    overall["rank_robust_score_macro"] = overall["robust_score_macro"].rank(
        ascending=False, method="min"
    )

    overall = overall.sort_values(
        ["rank_robust_score_macro", "rank_masked_f1_macro", "rank_pdi_f1_macro"]
    )

    return overall


def best_by_task(task_summary: pd.DataFrame):
    best_masked = (
        task_summary
        .sort_values(["task", "masked_f1_macro_mean"], ascending=[True, False])
        .groupby("task")
        .head(5)
        .reset_index(drop=True)
    )

    best_pdi = (
        task_summary
        .sort_values(["task", "pdi_f1_macro_mean"], ascending=[True, True])
        .groupby("task")
        .head(5)
        .reset_index(drop=True)
    )

    return best_masked, best_pdi


def print_console_overview(df: pd.DataFrame, task_summary: pd.DataFrame, overall: pd.DataFrame):
    print("\n" + "=" * 100)
    print("[BOOTSTRAP RAW INFO]")
    print("=" * 100)
    print(f"rows             : {len(df):,}")
    print(f"models           : {df['model'].nunique()}")
    print(f"tasks            : {sorted(df['task'].unique().tolist())}")
    print(f"bootstrap seeds  : {df['bootstrap_seed'].nunique()}")
    print(f"seed range       : {df['bootstrap_seed'].min()} ~ {df['bootstrap_seed'].max()}")

    print("\n" + "=" * 100)
    print("[OVERALL MODEL RANKING: robust_score_macro = masked_f1_macro - drop_f1_macro]")
    print("=" * 100)

    view_cols = [
        "model",
        "masked_f1_macro_mean",
        "drop_f1_macro_mean",
        "pdi_f1_macro_mean",
        "robust_score_macro",
        "rank_robust_score_macro",
    ]

    print(overall[view_cols].head(20).to_string(index=False))

    print("\n" + "=" * 100)
    print("[TASK-WISE TOP MODELS BY MASKED F1 MACRO]")
    print("=" * 100)

    for task in sorted(task_summary["task"].unique()):
        sub = (
            task_summary[task_summary["task"] == task]
            .sort_values("masked_f1_macro_mean", ascending=False)
            .head(5)
        )

        cols = [
            "model",
            "masked_f1_macro_mean",
            "masked_f1_macro_std",
            "drop_f1_macro_mean",
            "pdi_f1_macro_mean",
        ]

        print(f"\n[{task}]")
        print(sub[cols].to_string(index=False))

    print("\n" + "=" * 100)
    print("[TASK-WISE TOP MODELS BY LOW PDI]")
    print("=" * 100)

    for task in sorted(task_summary["task"].unique()):
        sub = (
            task_summary[task_summary["task"] == task]
            .sort_values("pdi_f1_macro_mean", ascending=True)
            .head(5)
        )

        cols = [
            "model",
            "pdi_f1_macro_mean",
            "pdi_f1_macro_std",
            "masked_f1_macro_mean",
            "drop_f1_macro_mean",
        ]

        print(f"\n[{task}]")
        print(sub[cols].to_string(index=False))


def plot_metric_boxplot(df: pd.DataFrame, out_dir: Path, task: str, metric: str, top_k: int = 12):
    """
    task별 metric 분포 boxplot.
    모델이 너무 많으면 median 기준 top_k만 그림.
    """
    sub = df[df["task"] == task].copy()

    if sub.empty:
        return

    if "pdi" in metric or "drop" in metric:
        # 낮을수록 좋은 metric
        order = (
            sub.groupby("model")[metric]
            .median()
            .sort_values(ascending=True)
            .head(top_k)
            .index
            .tolist()
        )
    else:
        # 높을수록 좋은 metric
        order = (
            sub.groupby("model")[metric]
            .median()
            .sort_values(ascending=False)
            .head(top_k)
            .index
            .tolist()
        )

    plot_data = [sub[sub["model"] == m][metric].dropna().values for m in order]

    plt.figure(figsize=(14, 6))
    plt.boxplot(plot_data, labels=order, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric)
    plt.title(f"{task} | Bootstrap distribution | {metric}")
    plt.tight_layout()

    out_path = out_dir / f"task_{task}_{metric}_boxplot.png"
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_gaze_distributions(df: pd.DataFrame, out_dir: Path, top_k: int = 8):
    """
    gaze task에서 masked F1과 PDI 분포를 별도 저장.
    """
    gaze = df[df["task"] == "gaze"].copy()
    if gaze.empty:
        return

    # masked F1 top models
    top_masked_models = (
        gaze.groupby("model")["masked_f1_macro"]
        .mean()
        .sort_values(ascending=False)
        .head(top_k)
        .index
        .tolist()
    )

    plt.figure(figsize=(12, 6))
    for model in top_masked_models:
        vals = gaze[gaze["model"] == model]["masked_f1_macro"].dropna().values
        plt.hist(vals, bins=40, alpha=0.35, label=model)

    plt.xlabel("gaze masked F1 macro")
    plt.ylabel("count")
    plt.title("Gaze masked F1 bootstrap distribution")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "gaze_masked_f1_distribution.png", dpi=200)
    plt.close()

    # PDI low models
    top_pdi_models = (
        gaze.groupby("model")["pdi_f1_macro"]
        .mean()
        .sort_values(ascending=True)
        .head(top_k)
        .index
        .tolist()
    )

    plt.figure(figsize=(12, 6))
    for model in top_pdi_models:
        vals = gaze[gaze["model"] == model]["pdi_f1_macro"].dropna().values
        plt.hist(vals, bins=40, alpha=0.35, label=model)

    plt.xlabel("gaze PDI F1 macro (%)")
    plt.ylabel("count")
    plt.title("Gaze PDI bootstrap distribution")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "gaze_pdi_distribution.png", dpi=200)
    plt.close()


def plot_overall_summary(overall: pd.DataFrame, out_dir: Path, top_k: int = 12):
    """
    종합 robust score 상위 모델 bar plot.
    """
    sub = overall.head(top_k).copy()
    sub = sub.sort_values("robust_score_macro", ascending=True)

    plt.figure(figsize=(12, 6))
    plt.barh(sub["model"], sub["robust_score_macro"])
    plt.xlabel("robust_score_macro = masked_f1_macro_mean - drop_f1_macro_mean")
    plt.title("Overall bootstrap robust score")
    plt.tight_layout()
    plt.savefig(out_dir / "weighted_overall_summary.png", dpi=200)
    plt.close()


def pairwise_vs_best_gaze(df: pd.DataFrame, out_dir: Path):
    """
    gaze 기준 best model과 나머지 모델의 bootstrap seed별 차이를 계산.
    같은 bootstrap_seed끼리 비교하므로 paired comparison에 가까움.
    """
    gaze = df[df["task"] == "gaze"].copy()
    if gaze.empty:
        return pd.DataFrame()

    best_model = (
        gaze.groupby("model")["masked_f1_macro"]
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )

    best = gaze[gaze["model"] == best_model][
        ["bootstrap_seed", "masked_f1_macro", "pdi_f1_macro", "drop_f1_macro"]
    ].rename(
        columns={
            "masked_f1_macro": "best_masked_f1_macro",
            "pdi_f1_macro": "best_pdi_f1_macro",
            "drop_f1_macro": "best_drop_f1_macro",
        }
    )

    rows = []

    for model in sorted(gaze["model"].unique()):
        if model == best_model:
            continue

        cur = gaze[gaze["model"] == model][
            ["bootstrap_seed", "masked_f1_macro", "pdi_f1_macro", "drop_f1_macro"]
        ]

        merged = best.merge(cur, on="bootstrap_seed", how="inner")

        diff_masked = merged["best_masked_f1_macro"] - merged["masked_f1_macro"]
        diff_pdi = merged["best_pdi_f1_macro"] - merged["pdi_f1_macro"]
        diff_drop = merged["best_drop_f1_macro"] - merged["drop_f1_macro"]

        rows.append(
            {
                "best_model": best_model,
                "compare_model": model,
                "n_common_bootstrap": len(merged),

                # 양수면 best_model의 masked F1이 더 높음
                "diff_masked_f1_mean": diff_masked.mean(),
                "diff_masked_f1_ci95_low": np.percentile(diff_masked, 2.5),
                "diff_masked_f1_ci95_high": np.percentile(diff_masked, 97.5),
                "best_higher_masked_f1_rate": float((diff_masked > 0).mean()),

                # PDI는 낮을수록 좋음. diff_pdi가 음수면 best_model의 PDI가 더 낮음.
                "diff_pdi_mean_best_minus_compare": diff_pdi.mean(),
                "best_lower_pdi_rate": float((diff_pdi < 0).mean()),

                # drop도 낮을수록 좋음. diff_drop이 음수면 best_model의 drop이 더 낮음.
                "diff_drop_mean_best_minus_compare": diff_drop.mean(),
                "best_lower_drop_rate": float((diff_drop < 0).mean()),
            }
        )

    result = pd.DataFrame(rows)
    result = result.sort_values("diff_masked_f1_mean", ascending=False)

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_csv",
        type=str,
        default=str(DEFAULT_IN),
        help="Path to bootstrap_raw_by_seed.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(DEFAULT_OUT),
        help="Output analysis directory",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=12,
        help="Number of top models to plot",
    )

    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    out_dir = Path(args.out_dir)
    plot_dir = out_dir / "plots"

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] input_csv = {input_csv}")
    print(f"[INFO] out_dir   = {out_dir}")

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)

    required_cols = [
        "model",
        "task",
        "bootstrap_seed",
        "clean_f1_macro",
        "masked_f1_macro",
        "drop_f1_macro",
        "pdi_f1_macro",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    task_summary = summarize_by_model_task(df)
    overall = make_overall_summary(task_summary)
    best_masked, best_pdi = best_by_task(task_summary)
    pairwise_gaze = pairwise_vs_best_gaze(df, out_dir)

    # Save CSVs
    task_summary.to_csv(out_dir / "02_task_model_summary.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(out_dir / "01_overall_model_summary.csv", index=False, encoding="utf-8-sig")
    best_masked.to_csv(out_dir / "03_best_model_by_task_masked_f1.csv", index=False, encoding="utf-8-sig")
    best_pdi.to_csv(out_dir / "04_best_model_by_task_pdi.csv", index=False, encoding="utf-8-sig")

    if not pairwise_gaze.empty:
        pairwise_gaze.to_csv(out_dir / "05_model_pairwise_vs_best_gaze.csv", index=False, encoding="utf-8-sig")

    # Console overview
    print_console_overview(df, task_summary, overall)

    # Plots
    for task in sorted(df["task"].unique()):
        plot_metric_boxplot(df, plot_dir, task, "masked_f1_macro", top_k=args.top_k)
        plot_metric_boxplot(df, plot_dir, task, "drop_f1_macro", top_k=args.top_k)
        plot_metric_boxplot(df, plot_dir, task, "pdi_f1_macro", top_k=args.top_k)

    plot_gaze_distributions(df, plot_dir, top_k=min(args.top_k, 8))
    plot_overall_summary(overall, plot_dir, top_k=args.top_k)

    print("\n[SAVED]")
    print(f"  {out_dir / '01_overall_model_summary.csv'}")
    print(f"  {out_dir / '02_task_model_summary.csv'}")
    print(f"  {out_dir / '03_best_model_by_task_masked_f1.csv'}")
    print(f"  {out_dir / '04_best_model_by_task_pdi.csv'}")
    print(f"  {out_dir / '05_model_pairwise_vs_best_gaze.csv'}")
    print(f"  plots -> {plot_dir}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()