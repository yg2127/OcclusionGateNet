#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/shared/scuppy/hyi/Ablation/AblationB}
PYTHON=${PYTHON:-/data/shared/envs/scuppy/bin/python}
TAG=${TAG:-ablation_b_mediapipe_task_gated_no_gaze_occ}
SEEDS=${SEEDS:-"42 43 44"}
BASE_CONFIG=${BASE_CONFIG:-configs/ablation_b_task_gated_no_gaze_occ_mediapipe_base.yaml}

cd "$ROOT"

echo "[TaskGated No-Gaze-OCC] root=$ROOT"
echo "[TaskGated No-Gaze-OCC] python=$PYTHON"
echo "[TaskGated No-Gaze-OCC] tag=$TAG"
echo "[TaskGated No-Gaze-OCC] seeds=$SEEDS"
echo "[TaskGated No-Gaze-OCC] base_config=$BASE_CONFIG"
echo "[TaskGated No-Gaze-OCC] fusion=task_gated_late_no_gaze_occ"

PYTHONPATH=. "$PYTHON" tools/run_ablation_b_seed_sweep.py \
  --root "$ROOT" \
  --base-config "$BASE_CONFIG" \
  --python "$PYTHON" \
  --seeds $SEEDS \
  --tag "$TAG" \
  --only task_gated_late_no_gaze_occ \
  --skip-existing \
  --continue-on-error
