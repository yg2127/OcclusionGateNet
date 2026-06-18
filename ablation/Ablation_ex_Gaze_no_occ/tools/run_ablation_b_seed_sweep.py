from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

DEFAULT_ROOT = Path('/data/shared/scuppy/hyi/Ablation/AblationB')
DEFAULT_SEEDS = [42, 43, 44]

# Ablation B: only fusion location changes under same backbone/crop/split/window.
# no_occ_original uses occ.enabled=false and concat/identity-style fusion as the no-OCC baseline in this codebase.
DEFAULT_EXPERIMENTS = [
    {'name': 'no_occ_original', 'fusion_kind': 'concat', 'occ_enabled': False},
    {'name': 'concat_condition', 'fusion_kind': 'concat_condition', 'occ_enabled': True},
    {'name': 'task_gated_late', 'fusion_kind': 'task_gated_late', 'occ_enabled': True},
    {'name': 'task_gated_late_no_gaze_occ', 'fusion_kind': 'task_gated_late_no_gaze_occ', 'occ_enabled': True},
    {'name': 'task_region_gated_late_no_gaze_occ', 'fusion_kind': 'task_region_gated_late_no_gaze_occ', 'occ_enabled': True},
    {'name': 'task_region_scalar_gated_late_no_gaze_occ', 'fusion_kind': 'task_region_scalar_gated_late_no_gaze_occ', 'occ_enabled': True},
    {'name': 'explicit_region_mask_gate', 'fusion_kind': 'explicit_region_mask_gate', 'occ_enabled': True},
    {'name': 'occ_attention_bias', 'fusion_kind': 'occ_attention_bias', 'occ_enabled': True},
    {'name': 'task_region_scalar_gated_late', 'fusion_kind': 'task_region_scalar_gated_late', 'occ_enabled': True},
]


def deep_set(d: dict[str, Any], keys: list[str], value: Any) -> None:
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True, width=120)


def make_config(
    base_cfg: dict[str, Any],
    *,
    root: Path,
    seed: int,
    exp_name: str,
    fusion_kind: str,
    occ_enabled: bool,
    tag: str,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)

    save_root = root / 'artifacts' / f'{tag}_{exp_name}_seed{seed}'
    results_root = root / 'results'

    cfg['seed'] = int(seed)
    deep_set(cfg, ['paths', 'save_root'], str(save_root))
    deep_set(cfg, ['paths', 'results_root'], str(results_root))
    deep_set(cfg, ['paths', 'frame_shifts'], str(root / 'constants' / 'frame_shifts.json'))

    # Keep the supplied fixed manifest path unless the user overrides it in base yaml.
    deep_set(cfg, ['model', 'fusion', 'kind'], fusion_kind)
    deep_set(cfg, ['occ', 'enabled'], bool(occ_enabled))

    # For no-OCC baseline, model occ_dim becomes 0 and dataset x_occ is zeros.
    # For OCC-aware methods, keep dim/map/default values from base config.
    deep_set(cfg, ['train', 'resume'], False)
    deep_set(cfg, ['train', 'resume_path'], None)
    deep_set(cfg, ['notify', 'enabled'], False)
    deep_set(cfg, ['notify', 'tag'], f'{tag}_{exp_name}_seed{seed}')

    return cfg


def run_one(
    *,
    root: Path,
    python_bin: str,
    config_path: Path,
    log_path: Path,
    continue_on_error: bool,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['PYTHONPATH'] = str(root) + (':' + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')

    cmd = [python_bin, '-m', 'src.training.train', '--config', str(config_path)]
    print('[RUN]', ' '.join(cmd))
    print('[LOG]', log_path)

    with log_path.open('w', encoding='utf-8') as log_f:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if proc.returncode != 0:
        print(f'[FAILED] rc={proc.returncode} config={config_path}')
        if not continue_on_error:
            raise SystemExit(proc.returncode)
    else:
        print(f'[OK] config={config_path}')

    return int(proc.returncode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(DEFAULT_ROOT), help='Project root on the server')
    ap.add_argument('--base-config', default='configs/ablation_b_base.yaml')
    ap.add_argument('--python', default='/data/shared/envs/scuppy/bin/python')
    ap.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    ap.add_argument('--tag', default='ablation_b')
    ap.add_argument('--only', nargs='*', default=None, help='Subset experiment names to run')
    ap.add_argument('--skip-existing', action='store_true')
    ap.add_argument('--continue-on-error', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    root = Path(args.root)
    base_config_path = Path(args.base_config)
    if not base_config_path.is_absolute():
        base_config_path = root / base_config_path

    base_cfg = load_yaml(base_config_path)
    selected = DEFAULT_EXPERIMENTS
    if args.only:
        wanted = set(args.only)
        selected = [e for e in DEFAULT_EXPERIMENTS if e['name'] in wanted]
        missing = wanted - {e['name'] for e in selected}
        if missing:
            raise ValueError(f'Unknown --only names: {sorted(missing)}')

    generated_dir = root / 'configs' / 'generated_ablation_b'
    log_dir = root / 'logs' / 'ablation_b'
    plan = []

    for exp in selected:
        for seed in args.seeds:
            cfg = make_config(
                base_cfg,
                root=root,
                seed=seed,
                exp_name=exp['name'],
                fusion_kind=exp['fusion_kind'],
                occ_enabled=exp['occ_enabled'],
                tag=args.tag,
            )
            config_path = generated_dir / f'{args.tag}_{exp["name"]}_seed{seed}.yaml'
            save_root = Path(cfg['paths']['save_root'])
            log_path = log_dir / f'{args.tag}_{exp["name"]}_seed{seed}.log'
            save_yaml(cfg, config_path)
            plan.append((exp['name'], seed, config_path, save_root, log_path))

    print('=' * 80)
    print('[Ablation B seed sweep]')
    print('root        :', root)
    print('base config :', base_config_path)
    print('seeds       :', args.seeds)
    print('experiments :', [e['name'] for e in selected])
    print('num runs    :', len(plan))
    print('=' * 80)

    failures = []
    for name, seed, config_path, save_root, log_path in plan:
        summary = save_root / 'summary.json'
        if args.skip_existing and summary.exists():
            print(f'[SKIP existing] {name} seed={seed} summary={summary}')
            continue
        print(f'\n[START] {name} seed={seed}')
        print('config  :', config_path)
        print('save_root:', save_root)
        if args.dry_run:
            continue
        rc = run_one(
            root=root,
            python_bin=args.python,
            config_path=config_path,
            log_path=log_path,
            continue_on_error=args.continue_on_error,
        )
        if rc != 0:
            failures.append({'name': name, 'seed': seed, 'returncode': rc, 'config': str(config_path), 'log': str(log_path)})

    if failures:
        fail_path = log_dir / f'{args.tag}_failures.json'
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        fail_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding='utf-8')
        print('[DONE with failures]', fail_path)
        if not args.continue_on_error:
            raise SystemExit(1)
    else:
        print('[DONE] all requested runs completed or skipped')


if __name__ == '__main__':
    main()
