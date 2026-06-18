#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python train_dmd_original_multitask.py   --config configs/dmd_original_seed42_gaze045_light.yaml   2>&1 | tee train_dmd_original_seed42_gaze045_light.log
