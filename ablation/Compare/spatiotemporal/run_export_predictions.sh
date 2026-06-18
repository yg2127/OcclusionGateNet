#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/shared/scuppy/hyi/Ablation/Compare/spatiotemporal"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="$ROOT/artifacts_gaze045_light/spatiotemporal_seed42_gaze045_light"
CONFIG="$ROOT/configs/spatiotemporal_seed42_gaze045_light.yaml"

cd "$ROOT"
mkdir -p "$RUN_DIR"
"$PYTHON_BIN" export_compare_predictions.py \
  --project-root "$ROOT" \
  --config "$CONFIG" \
  --run-dir "$RUN_DIR" \
  --checkpoint "$RUN_DIR/best.pt" \
  2>&1 | tee "$RUN_DIR/export_predictions.log"
