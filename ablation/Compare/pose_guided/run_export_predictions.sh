#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python}"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
RUN_DIR="${RUN_DIR:-/data/shared/scuppy/hyi/Ablation/Compare/pose_guided/artifacts_gaze045_light/pose_guided_seed42_gaze045_light}"
CONFIG="${CONFIG:-configs/pose_guided_seed42_gaze045_light.yaml}"
NUM_WORKERS="${NUM_WORKERS:-2}"

"$PYTHON_BIN" export_compare_predictions.py   --project-root "$PROJECT_ROOT"   --run-dir "$RUN_DIR"   --config "$CONFIG"   --splits test_clean test_masked   --num-workers "$NUM_WORKERS"
