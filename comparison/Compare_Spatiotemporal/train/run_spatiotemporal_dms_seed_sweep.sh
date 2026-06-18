#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/shared/scuppy/hyi/Compare/Compare_Spatiotemporal"
PYTHON_BIN="/data/shared/envs/scuppy/bin/python"

cd "$ROOT"
"$PYTHON_BIN" tools/run_spatiotemporal_dms_seed_sweep.py \
  --root "$ROOT" \
  --base-config configs/spatiotemporal_dms_base.yaml \
  --python "$PYTHON_BIN" \
  --tag spatiotemporal_dms \
  --seeds 42 #43 44

"$PYTHON_BIN" tools/summarize_spatiotemporal_dms_results.py \
  --root "$ROOT" \
  --tag spatiotemporal_dms
