#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/shared/scuppy/hyi/Ablation/AblationB}
PYTHON=${PYTHON:-/data/shared/envs/scuppy/bin/python}
CONFIG=${CONFIG:-configs/ablation_b_base.yaml}

cd "$ROOT"
echo "[train-single] config=$CONFIG"
PYTHONPATH=. "$PYTHON" -m src.training.train --config "$CONFIG"
