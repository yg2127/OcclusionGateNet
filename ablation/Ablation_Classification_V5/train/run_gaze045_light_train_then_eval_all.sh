#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/shared/scuppy/hyi/Ablation/Ablation_Classification_V5}
PYTHON=${PYTHON:-/data/shared/envs/scuppy/bin/python}

cd "$ROOT"

bash train/run_gaze045_light_all.sh
bash train/run_gaze045_light_eval_predictions.sh

echo "[ALL DONE] train + eval finished."
