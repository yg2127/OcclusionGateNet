#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/shared/scuppy/hyi/Ablation/Compare/skateformer"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="configs/skateformer_seed42_gaze045_light.yaml"
LOG="train_skateformer_seed42_gaze045_light.log"

cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
"$PYTHON_BIN" -m src.training.train --config "$CONFIG" 2>&1 | tee "$LOG"
