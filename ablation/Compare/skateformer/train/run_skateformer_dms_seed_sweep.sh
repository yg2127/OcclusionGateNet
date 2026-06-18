#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/shared/scuppy/hyi/Ablation/Compare/skateformer"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$ROOT"
"$PYTHON_BIN" tools/run_skateformer_dms_seed_sweep.py   --root "$ROOT"   --base-config configs/skateformer_dms_base.yaml   --python "$PYTHON_BIN"   --tag skateformer   --seeds 42

"$PYTHON_BIN" tools/summarize_skateformer_dms_results.py   --root "$ROOT"   --tag skateformer
