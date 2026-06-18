#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/shared/scuppy/hyi/Ablation/Compare/skateformer"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="${RUN_DIR:-/data/shared/scuppy/hyi/Ablation/Compare/skateformer/artifacts_gaze045_light/skateformer_seed42_gaze045_light}"
CONFIG="${CONFIG:-$ROOT/configs/skateformer_seed42_gaze045_light.yaml}"
NUM_WORKERS="${NUM_WORKERS:-2}"

cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
"$PYTHON_BIN" "$ROOT/export_compare_predictions.py"   --project-root "$ROOT"   --run-dir "$RUN_DIR"   --config "$CONFIG"   --splits test_clean test_masked   --num-workers "$NUM_WORKERS"
