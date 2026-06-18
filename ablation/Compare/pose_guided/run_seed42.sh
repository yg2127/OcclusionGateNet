#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="configs/pose_guided_seed42_gaze045_light.yaml"
LOG="train_pose_guided_seed42_gaze045_light.log"

"$PYTHON_BIN" train_pose_guided_multitask.py   --config "$CONFIG"   2>&1 | tee "$LOG"
