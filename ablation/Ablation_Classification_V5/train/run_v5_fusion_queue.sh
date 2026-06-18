#!/usr/bin/env bash
set -euo pipefail

cd /data/shared/scuppy/hyi/Classification_model_V5_transformer

PYTHON=/data/shared/envs/scuppy/bin/python

# 혹시 아직 import mismatch가 남아 있으면 자동 수정
if grep -q "src.evaluation.v5_eval" src/training/runner.py; then
  sed -i 's/from src\.evaluation\.v5_eval import (/from src.evaluation.v1_eval import (/g' src/training/runner.py
fi

CONFIGS=(
  "configs/v5_concat.yaml"
  "configs/v5_concat_condition.yaml"
  #"configs/v5_task_gated_late.yaml"
  #"configs/v5_task_region_gated_late.yaml"
  #"configs/v5_task_region_scalar_gated_late.yaml"
  #"configs/v5_explicit_region_mask_gate.yaml"
  #"configs/v5_explicit_region_scalar_mask_gate.yaml"
  #"configs/v5_occ_token_region_transformer.yaml"
  #"configs/v5_occ_attention_bias.yaml"
)

mkdir -p logs

for CFG in "${CONFIGS[@]}"; do
  NAME="$(basename "$CFG" .yaml)"
  LOG="logs/${NAME}_$(date +%Y%m%d_%H%M%S).log"

  echo "============================================================"
  echo "[START] $CFG"
  echo "[LOG]   $LOG"
  echo "============================================================"

  "$PYTHON" -m src.training.train --config "$CFG" 2>&1 | tee "$LOG"

  echo "============================================================"
  echo "[DONE] $CFG"
  echo "============================================================"
done

echo "[ALL DONE] V5 fusion queue finished."