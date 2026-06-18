#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/shared/scuppy/hyi/Ablation/Ablation_Classification_V5}
PYTHON=${PYTHON:-/data/shared/envs/scuppy/bin/python}

cd "$ROOT"
mkdir -p logs

# Generate gaze/action-balanced light configs if they do not exist.
if [ ! -f "configs_gaze045_light/config_list.txt" ]; then
  "$PYTHON" tools/make_gaze045_light_configs.py --root "$ROOT"
fi

mapfile -t CONFIGS < configs_gaze045_light/config_list.txt

for CFG in "${CONFIGS[@]}"; do
  NAME="$(basename "$CFG" .yaml)"
  LOG="logs/${NAME}_train_$(date +%Y%m%d_%H%M%S).log"

  echo "============================================================"
  echo "[TRAIN]  $CFG"
  echo "[PYTHON] $PYTHON"
  echo "[LOG]    $LOG"
  echo "============================================================"

  "$PYTHON" -m src.training.train --config "$CFG" 2>&1 | tee "$LOG"

  echo "============================================================"
  echo "[DONE] $CFG"
  echo "============================================================"
done

echo "[ALL DONE] gaze045 light V5 training finished."
