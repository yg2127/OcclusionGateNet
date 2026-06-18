#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/data/shared/envs/scuppy/bin/python}"
DFS_ROOT="/data/shared/scuppy/hyi/Ablation/Compare/dfs"
RUN_DIR="/data/shared/scuppy/hyi/Ablation/Compare/dfs/runs/dfs_fixed_clean_masked_seed42_loss045"
CONFIG="/data/shared/scuppy/hyi/Ablation/Compare/dfs/experiments/dfs_fixed_clean_masked_seed42.yaml"
EXPORT_SCRIPT="/data/shared/scuppy/hyi/Ablation/Compare/dfs/export_dfs_predictions.py"
CHECKPOINT="${RUN_DIR}/best.pt"
NUM_WORKERS="${NUM_WORKERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-}"

echo "============================================================"
echo "[DFS ONLY] export predictions"
echo "DFS_ROOT      = ${DFS_ROOT}"
echo "RUN_DIR       = ${RUN_DIR}"
echo "CONFIG        = ${CONFIG}"
echo "EXPORT_SCRIPT = ${EXPORT_SCRIPT}"
echo "CHECKPOINT    = ${CHECKPOINT}"
echo "PYTHON        = ${PYTHON}"
echo "============================================================"

if [ ! -d "${DFS_ROOT}" ]; then
  echo "[ERROR] DFS_ROOT not found: ${DFS_ROOT}"
  exit 1
fi

if [ ! -d "${RUN_DIR}" ]; then
  echo "[ERROR] RUN_DIR not found: ${RUN_DIR}"
  exit 1
fi

if [ ! -f "${CONFIG}" ]; then
  echo "[ERROR] CONFIG not found: ${CONFIG}"
  exit 1
fi

if [ ! -f "${EXPORT_SCRIPT}" ]; then
  echo "[ERROR] EXPORT_SCRIPT not found: ${EXPORT_SCRIPT}"
  exit 1
fi

if [ ! -f "${CHECKPOINT}" ]; then
  echo "[ERROR] CHECKPOINT not found: ${CHECKPOINT}"
  exit 1
fi

cd "${DFS_ROOT}"
export PYTHONPATH="${DFS_ROOT}:${PYTHONPATH:-}"

CMD=(
  "${PYTHON}" "${EXPORT_SCRIPT}"
  --dfs-root "${DFS_ROOT}"
  --run-dir "${RUN_DIR}"
  --config "${CONFIG}"
  --checkpoint "${CHECKPOINT}"
  --splits test_clean test_masked
  --num-workers "${NUM_WORKERS}"
)

if [ -n "${BATCH_SIZE}" ]; then
  CMD+=(--batch-size "${BATCH_SIZE}")
fi

echo "[RUN] ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "${RUN_DIR}/export_predictions.log"

echo
echo "============================================================"
echo "[CHECK] generated prediction files"
echo "============================================================"

find "${RUN_DIR}" -type f -name "*predictions.csv" | sort

echo
echo "[DONE] dfs prediction export complete."