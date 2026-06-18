#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash train/run_eval_predictions.sh
#   bash train/run_eval_predictions.sh configs/v5_task_gated_late.yaml configs/v5_occ_attention_bias.yaml

cd /data/shared/scuppy/hyi/Classification_model_V5_transformer

# 네가 현재 쓰는 환경에 맞춰 하나만 선택.
# 1) 기존 코드에 적혀 있던 공용 env
PYTHON=${PYTHON:-/data/shared/envs/scuppy/bin/python}
# 2) ~/Code/.venv를 쓰려면 실행할 때 이렇게:
#    PYTHON=/home/hyi/Code/.venv/bin/python bash train/run_eval_predictions.sh

CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
  CONFIGS=(
    #"configs/v5_concat.yaml"
    #"configs/v5_concat_condition.yaml"
    "configs/v5_task_gated_late.yaml"
    "configs/v5_task_region_gated_late.yaml"
    "configs/v5_task_region_scalar_gated_late.yaml"
    "configs/v5_explicit_region_mask_gate.yaml"
    "configs/v5_explicit_region_scalar_mask_gate.yaml"
    "configs/v5_occ_token_region_transformer.yaml"
    "configs/v5_occ_attention_bias.yaml"
  )
fi

mkdir -p logs

for CFG in "${CONFIGS[@]}"; do
  NAME="$(basename "$CFG" .yaml)"
  LOG="logs/${NAME}_eval_predictions_$(date +%Y%m%d_%H%M%S).log"

  echo "============================================================"
  echo "[EVAL ONLY] $CFG"
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

echo "[ALL DONE] eval-only prediction export finished."
