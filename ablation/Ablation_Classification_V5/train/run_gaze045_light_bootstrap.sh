#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/shared/scuppy/hyi/Ablation/Ablation_Classification_V5}
PYTHON=${PYTHON:-/data/shared/envs/scuppy/bin/python}
N_BOOT=${N_BOOT:-5000}

cd "$ROOT"

"$PYTHON" tools/bootstrap_gaze045_light_from_predictions.py \
  --root "$ROOT" \
  --num_bootstrap_seeds "$N_BOOT"

"$PYTHON" tools/quick_compare_gaze045_light.py \
  --summary_csv "$ROOT/bootstrap_results_gaze045_light/bootstrap_summary_by_model_task.csv" \
  | tee "$ROOT/bootstrap_results_gaze045_light/quick_compare.txt"

echo "[ALL DONE] bootstrap + quick comparison finished."
