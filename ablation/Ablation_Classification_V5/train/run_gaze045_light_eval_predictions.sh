#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/shared/scuppy/hyi/Ablation/Ablation_Classification_V5}
PYTHON=${PYTHON:-/data/shared/envs/scuppy/bin/python}

cd "$ROOT"
mkdir -p logs

if [ ! -f "configs_gaze045_light/config_list.txt" ]; then
  "$PYTHON" tools/make_gaze045_light_configs.py --root "$ROOT"
fi

CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
  mapfile -t CONFIGS < configs_gaze045_light/config_list.txt
fi

for CFG in "${CONFIGS[@]}"; do
  NAME="$(basename "$CFG" .yaml)"
  LOG="logs/${NAME}_eval_predictions_$(date +%Y%m%d_%H%M%S).log"

  echo "============================================================"
  echo "[EVAL PRED] $CFG"
  echo "[PYTHON]    $PYTHON"
  echo "[LOG]       $LOG"
  echo "============================================================"

  "$PYTHON" -m src.evaluation.eval_only_save_predictions \
    --config "$CFG" \
    --splits test_clean test_masked \
    --no-window-predictions \
    2>&1 | tee "$LOG"

  echo "[DONE] $CFG"
  echo
done

echo "[ALL DONE] gaze045 light prediction export finished."
