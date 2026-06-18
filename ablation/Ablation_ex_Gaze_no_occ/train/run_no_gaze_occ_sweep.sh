#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/shared/scuppy/hyi/Ablation/AblationB}
PYTHON=${PYTHON:-/data/shared/envs/scuppy/bin/python}
BASE_CONFIG=${BASE_CONFIG:-configs/ablation_b_base.yaml}
SEEDS=${SEEDS:-"42 43 44"}
TAG=${TAG:-ablation_b_no_gaze_occ}

cd "$ROOT"

PYTHONPATH=. "$PYTHON" tools/run_ablation_b_seed_sweep.py \
  --root "$ROOT" \
  --base-config "$BASE_CONFIG" \
  --seeds $SEEDS \
  --tag "$TAG" \
  --only task_gated_late_no_gaze_occ task_region_gated_late_no_gaze_occ task_region_scalar_gated_late_no_gaze_occ \
  --skip-existing \
  --continue-on-error
