#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/shared/scuppy/hyi/Ablation/Ablation_ex_1.2}"
PYTHON="${PYTHON:-/data/shared/envs/scuppy/bin/python}"
CONFIG="${CONFIG:-configs/clean_mask_train_clean_mask_test.yaml}"

cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

if [ ! -f constants/frame_shifts.json ]; then
  echo "[prebuild] constants/frame_shifts.json not found; generating..."
  "$PYTHON" scripts/build_frame_shifts.py
fi

echo "[run] ROOT=$ROOT"
echo "[run] PYTHON=$PYTHON"
echo "[run] CONFIG=$CONFIG"
"$PYTHON" -m src.training.train --config "$CONFIG"
