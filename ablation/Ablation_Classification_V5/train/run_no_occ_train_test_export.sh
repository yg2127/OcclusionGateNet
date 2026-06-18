#!/usr/bin/env bash
set -euo pipefail

# no_occ_original only: train -> test -> export ROC/PR-ready prediction CSVs.
# Run from project root:
#   bash train/run_no_occ_train_test_export.sh
# Optional:
#   PYTHON=/home/hyi/Code/.venv/bin/python bash train/run_no_occ_train_test_export.sh
#   CONFIG=configs/v5_no_occ_original_mediapipe_seed42.yaml bash train/run_no_occ_train_test_export.sh

PROJECT_ROOT="${PROJECT_ROOT:-/data/shared/scuppy/hyi/Classification_model_V5_transformer}"
CONFIG="${CONFIG:-configs/v5_no_occ_original_mediapipe_seed42.yaml}"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "/data/shared/envs/scuppy/bin/python" ]]; then
    PYTHON="/data/shared/envs/scuppy/bin/python"
  elif [[ -x "/home/hyi/Code/.venv/bin/python" ]]; then
    PYTHON="/home/hyi/Code/.venv/bin/python"
  else
    PYTHON="python"
  fi
fi

cd "$PROJECT_ROOT"

echo "[INFO] project root: $PROJECT_ROOT"
echo "[INFO] python      : $PYTHON"
echo "[INFO] config      : $CONFIG"

echo "============================================================"
echo "[1/3] Train no_occ_original"
echo "============================================================"
"$PYTHON" -m src.training.train --config "$CONFIG"

SAVE_ROOT=$("$PYTHON" - <<PY
import yaml
from pathlib import Path
cfg = yaml.safe_load(open("$CONFIG", "r", encoding="utf-8"))
print(cfg["paths"]["save_root"])
PY
)

CKPT="$SAVE_ROOT/best.pt"
if [[ ! -f "$CKPT" ]]; then
  echo "[ERROR] best.pt not found: $CKPT" >&2
  echo "[HINT] Check training log: $SAVE_ROOT/train.log" >&2
  exit 1
fi

echo "============================================================"
echo "[2/3] Eval-only export predictions"
echo "============================================================"
"$PYTHON" -m src.evaluation.eval_only_save_predictions \
  --config "$CONFIG" \
  --checkpoint "$CKPT" \
  --splits test_clean test_masked \
  --no-window-predictions

echo "============================================================"
echo "[3/3] Check generated files"
echo "============================================================"
find "$SAVE_ROOT" -maxdepth 1 -type f \( -name '*predictions.csv' -o -name '*with_pdi.csv' -o -name 'summary.json' \) | sort

echo "[DONE] no_occ_original train/test/export completed."
echo "[SAVE_ROOT] $SAVE_ROOT"
