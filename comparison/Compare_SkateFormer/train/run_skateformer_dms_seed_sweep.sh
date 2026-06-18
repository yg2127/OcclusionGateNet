#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/shared/scuppy/hyi/Compare/Compare_SkateFormer"
PYTHON_BIN="/data/shared/envs/scuppy/bin/python"

cd "$ROOT"
"$PYTHON_BIN" tools/run_skateformer_dms_seed_sweep.py \
  --root "$ROOT" \
  --base-config configs/skateformer_dms_base.yaml \
  --python "$PYTHON_BIN" \
  --tag skateformer_dms \
  --seeds 42 #43 44

"$PYTHON_BIN" tools/summarize_skateformer_dms_results.py \
  --root "$ROOT" \
  --tag skateformer_dms
