#!/usr/bin/env bash
set -euo pipefail

# Same execution style as AblationB/train/*.sh
ROOT=${ROOT:-/data/shared/scuppy/hyi/Ablation/Ablation_ex_Gaze_clean_to_clean}
PYTHON=${PYTHON:-/data/shared/envs/scuppy/bin/python}
CONFIG=${CONFIG:-configs/gaze_clean_to_clean.yaml}

cd "$ROOT"

echo "[gaze-clean-to-clean] root=$ROOT"
echo "[gaze-clean-to-clean] python=$PYTHON"
echo "[gaze-clean-to-clean] config=$CONFIG"

PYTHONPATH=. "$PYTHON" tools/train_gaze_clean_to_clean.py \
  --config "$CONFIG"
