#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/shared/scuppy/hyi/Ablation/AblationB}
PYTHON=${PYTHON:-/data/shared/envs/scuppy/bin/python}
TAG=${TAG:-ablation_b_mediapipe}
SEEDS=${SEEDS:-"43 44"}

cd "$ROOT"

echo "[AblationB MediaPipe Remaining] root=$ROOT"
echo "[AblationB MediaPipe Remaining] python=$PYTHON"
echo "[AblationB MediaPipe Remaining] tag=$TAG"
echo "[AblationB MediaPipe Remaining] seeds=$SEEDS"
echo "[AblationB MediaPipe Remaining] skip: no_occ_original"
echo "[AblationB MediaPipe Remaining] run:"
echo "  - concat_condition"
echo "  - task_gated_late"
echo "  - explicit_region_mask_gate"
echo "  - occ_attention_bias"
echo "  - task_region_gated_late"
echo "  - task_region_scalar_gated_late"

PYTHONPATH=. "$PYTHON" tools/run_ablation_b_seed_sweep.py \
  --root "$ROOT" \
  --base-config configs/ablation_b_base.yaml \
  --python "$PYTHON" \
  --seeds $SEEDS \
  --tag "$TAG" \
  --only concat_condition task_gated_late explicit_region_mask_gate occ_attention_bias task_region_gated_late task_region_scalar_gated_late \
  --skip-existing \
  --continue-on-error