from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

HEADS = ("action", "gaze", "hands", "talk")


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def summarize_run(root: Path) -> list[dict]:
    summary = _load_json(root / "summary.json")
    cfg = _load_json(root / "config.json")
    drop = _load_json(root / "test_clean_vs_masked_drop.json")

    row_base = {
        "run": root.name,
        "train_variants": "+".join(_safe_get(cfg, "data", "train_variants", default=[])),
        "val_variants": "+".join(_safe_get(cfg, "data", "val_variants", default=[])),
        "test_variants": "+".join(_safe_get(cfg, "data", "test_variants", default=[])),
        "best_epoch": summary.get("best_epoch"),
        "best_score": summary.get("best_score"),
        "n_train_windows": summary.get("n_train_windows"),
        "n_val_windows": summary.get("n_val_windows"),
    }

    rows = []
    test_splits = summary.get("test_splits", {})
    for head in HEADS:
        r = dict(row_base)
        r["head"] = head
        for split_name in ("test_clean", "test_masked"):
            metrics = _safe_get(test_splits, split_name, "per_head", head, default={}) or {}
            prefix = split_name.replace("test_", "")
            r[f"{prefix}_clip_f1_macro"] = metrics.get("clip_f1_macro")
            r[f"{prefix}_clip_acc"] = metrics.get("clip_acc")
            r[f"{prefix}_window_f1_macro"] = metrics.get("window_f1_macro")
            r[f"{prefix}_window_acc"] = metrics.get("window_acc")

        d = drop.get(head, {}) if isinstance(drop, dict) else {}
        for key, value in d.items():
            r[key] = value
        rows.append(r)

    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="artifact run directories")
    ap.add_argument("--out", default="results/requested_summary.csv")
    args = ap.parse_args()

    all_rows = []
    for root_s in args.roots:
        all_rows.extend(summarize_run(Path(root_s)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[saved] {out} rows={len(df)}")


if __name__ == "__main__":
    main()
