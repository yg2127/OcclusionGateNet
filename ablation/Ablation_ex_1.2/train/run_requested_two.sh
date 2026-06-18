#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/shared/scuppy/hyi/Ablation/Ablation_ex_1.2}"
PYTHON="${PYTHON:-/data/shared/envs/scuppy/bin/python}"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

if [ ! -f constants/frame_shifts.json ]; then
  echo "[prebuild] constants/frame_shifts.json not found; generating..."
  "$PYTHON" scripts/build_frame_shifts.py
fi

run_cfg() {
  local cfg="$1"
  echo
  echo "================================================================================"
  echo "[run] $cfg"
  echo "================================================================================"
  "$PYTHON" -m src.training.train --config "$cfg"
}

# 1) clean으로만 train/val → clean test
run_cfg configs/clean_only_train_clean_test.yaml

# 2) clean+masked로 train/val → clean test + masked test + clean-vs-masked drop 저장
run_cfg configs/clean_mask_train_clean_mask_test.yaml

"$PYTHON" tools/summarize_requested_results.py \
  --roots \
  artifacts/clean_only_train_clean_test_seed42 \
  artifacts/clean_mask_train_clean_mask_test_seed42 \
  --out results/requested_two_summary.csv
