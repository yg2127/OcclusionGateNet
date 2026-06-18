#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/shared/scuppy/hyi/Ablation/Ablation_ex_478_only_gaze}"
PYTHON="${PYTHON:-/data/shared/envs/scuppy/bin/python}"
CONFIG="${CONFIG:-configs/clean_only_train_clean_test.yaml}"

cd "$ROOT"

if [ ! -f "$CONFIG" ]; then
  echo "[ERROR] config not found: $CONFIG"
  exit 1
fi

echo "============================================================"
echo "[RUN SINGLE YAML]"
echo "ROOT   : $ROOT"
echo "PYTHON : $PYTHON"
echo "CONFIG : $CONFIG"
echo "TIME   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"$PYTHON" -m src.training.train --config "$CONFIG"
