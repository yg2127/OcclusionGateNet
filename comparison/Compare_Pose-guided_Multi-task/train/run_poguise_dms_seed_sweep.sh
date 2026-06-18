#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/shared/scuppy/hyi/Compare/Compare_Pose-guided Multi-task}"
PYTHON_BIN="${PYTHON_BIN:-/data/shared/envs/scuppy/bin/python}"
SEEDS="${SEEDS:-42 43 44}"

cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

"$PYTHON_BIN" tools/run_poguise_dms_seed_sweep.py \
  --root "$ROOT" \
  --python "$PYTHON_BIN" \
  --seeds $SEEDS \
  --tag poguise_dms \
  --continue-on-error

"$PYTHON_BIN" tools/summarize_poguise_dms_results.py \
  --root "$ROOT" \
  --tag poguise_dms \
  --out "$ROOT/analysis/poguise_dms"
