#!/usr/bin/env bash
set -euo pipefail

# no_occ_original eval/export only. Use after training or when best.pt already exists.
# Run from project root:
#   bash train/run_no_occ_eval_export_only.sh

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
SAVE_ROOT=$("$PYTHON" - <<PY
import yaml
cfg = yaml.safe_load(open("$CONFIG", "r", encoding="utf-8"))
print(cfg["paths"]["save_root"])
PY
)
CKPT="${CHECKPOINT:-$SAVE_ROOT/best.pt}"

if [[ ! -f "$CKPT" ]]; then
  echo "[ERROR] checkpoint not found: $CKPT" >&2
  exit 1
fi

echo "[INFO] project root: $PROJECT_ROOT"
echo "[INFO] python      : $PYTHON"
echo "[INFO] config      : $CONFIG"
echo "[INFO] checkpoint  : $CKPT"

"$PYTHON" -m src.evaluation.eval_only_save_predictions \
  --config "$CONFIG" \
  --checkpoint "$CKPT" \
  --splits test_clean test_masked \
  --no-window-predictions

find "$SAVE_ROOT" -maxdepth 1 -type f -name '*predictions.csv' | sort
