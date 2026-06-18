#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/shared/scuppy/hyi/Ablation/AblationB}
PYTHON=${PYTHON:-/data/shared/envs/scuppy/bin/python}
TAG=${TAG:-ablation_b}

cd "$ROOT"

echo "[AblationB] root=$ROOT"
echo "[AblationB] python=$PYTHON"
echo "[AblationB] tag=$TAG"

PYTHONPATH=. "$PYTHON" tools/run_ablation_b_seed_sweep.py \
  --root "$ROOT" \
  --base-config configs/ablation_b_base.yaml \
  --python "$PYTHON" \
  --seeds 42 43 44 \
  --tag "$TAG" \
  --skip-existing \
  --continue-on-error
