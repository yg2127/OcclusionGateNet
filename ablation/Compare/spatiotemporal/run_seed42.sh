#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/shared/scuppy/hyi/Ablation/Compare/spatiotemporal"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$ROOT"
"$PYTHON_BIN" -m src.training.train   --config configs/spatiotemporal_seed42_gaze045_light.yaml   2>&1 | tee train_spatiotemporal_seed42_gaze045_light.log
