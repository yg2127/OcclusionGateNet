from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import pandas as pd

WEIGHTS = {'action': 0.70, 'gaze': 0.15, 'hands': 0.10, 'talk': 0.05}
HEADS = ['action', 'gaze', 'hands', 'talk']


def _get_head_metric(split: dict[str, Any], head: str, key: str) -> float | None:
    # summary schema: test_splits -> test_clean/test_masked -> per_head -> head -> clip_f1_macro, clip_acc, ...
    try:
        return float(split['per_head'][head][key])
    except Exception:
        return None


def read_summary(summary_path: Path) -> dict[str, Any] | None:
    if not summary_path.exists():
        return None
    try:
        s = json.loads(summary_path.read_text(encoding='utf-8'))
    except Exception:
        return None

    cfg = {}
    cfg_path = summary_path.parent / 'config.json'
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
        except Exception:
            cfg = {}

    fusion = cfg.get('model', {}).get('fusion', {}).get('kind', summary_path.parent.name)
    occ_enabled = bool(cfg.get('occ', {}).get('enabled', False))
    seed = cfg.get('seed')

    test_splits = s.get('test_splits', {})
    clean = test_splits.get('test_clean') or test_splits.get('clean') or {}
    masked = test_splits.get('test_masked') or test_splits.get('masked') or {}

    row = {
        'run_dir': str(summary_path.parent),
        'fusion': fusion,
        'occ_enabled': occ_enabled,
        'seed': seed,
    }

    weighted_clean = 0.0
    weighted_masked = 0.0
    weighted_drop = 0.0
    has_all = True

    for h in HEADS:
        c = _get_head_metric(clean, h, 'clip_f1_macro')
        m = _get_head_metric(masked, h, 'clip_f1_macro')
        if c is None or m is None:
            has_all = False
            row[f'{h}_clean_f1'] = None
            row[f'{h}_masked_f1'] = None
            row[f'{h}_drop'] = None
            continue
        row[f'{h}_clean_f1'] = c
        row[f'{h}_masked_f1'] = m
        row[f'{h}_drop'] = c - m
        weighted_clean += WEIGHTS[h] * c
        weighted_masked += WEIGHTS[h] * m
        weighted_drop += WEIGHTS[h] * (c - m)

    row['weighted_clean_f1'] = weighted_clean if has_all else None
    row['weighted_masked_f1'] = weighted_masked if has_all else None
    row['weighted_drop'] = weighted_drop if has_all else None

    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/data/shared/scuppy/hyi/Compare/Compare_Pose-guided Multi-task')
    ap.add_argument('--tag', default='poguise_dms')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    root = Path(args.root)
    artifact_root = root / 'artifacts'
    rows = []
    for p in sorted(artifact_root.glob(f'{args.tag}_*/summary.json')):
        row = read_summary(p)
        if row is not None:
            rows.append(row)

    if not rows:
        raise SystemExit(f'No summary.json found under {artifact_root}/{args.tag}_*/summary.json')

    df = pd.DataFrame(rows).sort_values(['fusion', 'seed'])
    out_dir = Path(args.out) if args.out else root / 'analysis' / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'poguise_dms_seed_results_raw.csv', index=False, encoding='utf-8-sig')

    metric_cols = [
        'weighted_masked_f1', 'weighted_drop',
        'action_masked_f1', 'action_drop',
        'gaze_masked_f1', 'gaze_drop',
        'hands_masked_f1', 'hands_drop',
        'talk_masked_f1', 'talk_drop',
    ]
    agg_rows = []
    for fusion, g in df.groupby('fusion', dropna=False):
        row = {'fusion': fusion, 'n': len(g)}
        for col in metric_cols:
            vals = [float(x) for x in g[col].dropna().tolist()]
            if vals:
                row[f'{col}_mean'] = mean(vals)
                row[f'{col}_std'] = pstdev(vals) if len(vals) > 1 else 0.0
            else:
                row[f'{col}_mean'] = None
                row[f'{col}_std'] = None
        agg_rows.append(row)

    agg = pd.DataFrame(agg_rows).sort_values('weighted_masked_f1_mean', ascending=False)
    agg.to_csv(out_dir / 'poguise_dms_seed_results_mean_std.csv', index=False, encoding='utf-8-sig')

    print('[SAVED]', out_dir / 'poguise_dms_seed_results_raw.csv')
    print('[SAVED]', out_dir / 'poguise_dms_seed_results_mean_std.csv')
    print(agg[['fusion', 'n', 'weighted_masked_f1_mean', 'weighted_drop_mean', 'gaze_masked_f1_mean', 'gaze_drop_mean']].to_string(index=False))


if __name__ == '__main__':
    main()
