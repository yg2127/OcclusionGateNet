#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python train_driveact_multitask.py   --config configs/driveact_seed42_gaze045_light.yaml   2>&1 | tee train_driveact_seed42_gaze045_light.log
