#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
RUN_DIR="${RUN_DIR:-}"
CONFIG="${CONFIG:-}"
CHECKPOINT="${CHECKPOINT:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
NUM_WORKERS="${NUM_WORKERS:-2}"

if [ -z "$RUN_DIR" ]; then
  echo "[ERROR] RUN_DIR is required"
  echo "Example:"
  echo "  PROJECT_ROOT=/data/shared/scuppy/hyi/Compare/Compare_SkateFormer \\"
  echo "  RUN_DIR=/data/shared/scuppy/hyi/Compare/Compare_SkateFormer/artifacts/skateformer_dms_skateformer_face_seed42 \\"
  echo "  PYTHON=/home/hyi/Code/.venv/bin/python \\"
  echo "  bash run_compare_export_predictions.sh"
  exit 1
fi

CMD=(
  "$PYTHON" "$PROJECT_ROOT/export_compare_predictions.py"
  --project-root "$PROJECT_ROOT"
  --run-dir "$RUN_DIR"
  --splits test_clean test_masked
  --num-workers "$NUM_WORKERS"
)

if [ -n "$CONFIG" ]; then
  CMD+=(--config "$CONFIG")
fi

if [ -n "$CHECKPOINT" ]; then
  CMD+=(--checkpoint "$CHECKPOINT")
fi

if [ -n "$BATCH_SIZE" ]; then
  CMD+=(--batch-size "$BATCH_SIZE")
fi

echo "[RUN] ${CMD[*]}"
"${CMD[@]}"
