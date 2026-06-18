#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
RUN_DIR="/data/shared/scuppy/hyi/Ablation/Compare/driveact/artifacts_gaze045_light/driveact_seed42_gaze045_light"

python export_driveact_predictions.py   --driveact-root .   --run-dir "${RUN_DIR}"   --config configs/driveact_seed42_gaze045_light.yaml   --splits test_clean test_masked   --num-workers 2
