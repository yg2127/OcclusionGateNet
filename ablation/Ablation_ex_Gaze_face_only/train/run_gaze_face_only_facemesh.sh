#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/shared/scuppy/hyi/Ablation/Ablation_ex_Gaze_face_only}
PYTHON=${PYTHON:-/data/shared/envs/scuppy/bin/python}
TAG=${TAG:-gaze_face_only_facemesh}
SEEDS=${SEEDS:-"42 43 44"}

cd "$ROOT"

echo "[Gaze Face-Only] root=$ROOT"
echo "[Gaze Face-Only] python=$PYTHON"
echo "[Gaze Face-Only] tag=$TAG"
echo "[Gaze Face-Only] seeds=$SEEDS"
echo "[Gaze Face-Only] fusion=task_gated_late_gaze_face_only"
echo "[Gaze Face-Only] objective=gaze-only loss; action/hands/talk loss weights are 0"

PYTHONPATH=. "$PYTHON" tools/run_ablation_b_seed_sweep.py \
  --root "$ROOT" \
  --base-config configs/gaze_face_only_facemesh_base.yaml \
  --python "$PYTHON" \
  --seeds $SEEDS \
  --tag "$TAG" \
  --only task_gated_late_gaze_face_only \
  --skip-existing \
  --continue-on-error
