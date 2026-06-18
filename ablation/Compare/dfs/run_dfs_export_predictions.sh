#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   DFS_ROOT=/data/shared/scuppy/baselines/dfs \
#   RUN_DIR=/data/shared/scuppy/baselines/dfs/runs/dfs_fixed_clean_masked_seed42 \
#   PYTHON=/data/shared/envs/scuppy/bin/python \
#   bash run_dfs_export_predictions.sh

PYTHON=${PYTHON:-python}
DFS_ROOT=${DFS_ROOT:-/data/shared/scuppy/baselines/dfs}
RUN_DIR=${RUN_DIR:-${DFS_ROOT}/runs/dfs_fixed_clean_masked_seed42}
CONFIG=${CONFIG:-${RUN_DIR}/config.json}
CHECKPOINT=${CHECKPOINT:-${RUN_DIR}/best.pt}

cd "$DFS_ROOT"

"$PYTHON" "$DFS_ROOT/export_dfs_predictions.py" \
  --dfs-root "$DFS_ROOT" \
  --run-dir "$RUN_DIR" \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --splits test_clean test_masked \
  --num-workers 2
