#!/usr/bin/env bash
set -euo pipefail

DRIVEACT_ROOT="${DRIVEACT_ROOT:-/data/shared/scuppy/baselines/driveact}"
RUN_DIR="${RUN_DIR:-${DRIVEACT_ROOT}/runs/driveact_fixed_clean_masked_seed42}"
PYTHON="${PYTHON:-/data/shared/envs/scuppy/bin/python}"
CONFIG="${CONFIG:-${RUN_DIR}/config.json}"
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/best.pt}"

cd "$DRIVEACT_ROOT"

echo "[INFO] DRIVEACT_ROOT=$DRIVEACT_ROOT"
echo "[INFO] RUN_DIR=$RUN_DIR"
echo "[INFO] PYTHON=$PYTHON"
echo "[INFO] CONFIG=$CONFIG"
echo "[INFO] CHECKPOINT=$CHECKPOINT"

"$PYTHON" export_driveact_predictions.py \
  --driveact-root "$DRIVEACT_ROOT" \
  --run-dir "$RUN_DIR" \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --splits test_clean test_masked \
  --num-workers 2

find "$RUN_DIR" -maxdepth 1 -name '*predictions.csv' | sort
