#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# HGNET_Classification 4545 three variants prediction export
#
# Variants:
#   1) task_gated_late
#   2) task_region_scalar_gated_late
#   3) explicit_region_scalar_mask_gate
#
# Export:
#   test_clean / test_masked
#   clip-level predictions
#
# Usage:
#   bash /data/shared/scuppy/hyi/Ablation/bootstrap_toolkit/export_hgnet_4545_three_predictions.sh
#
# Optional:
#   PYTHON=/path/to/python NUM_WORKERS=2 bash ...
# ============================================================

PYTHON="${PYTHON:-/data/shared/envs/scuppy/bin/python}"

HGNET_ROOT="/data/shared/scuppy/hyi/Ablation/HGNET_Classification"
CLASSIFIER_ROOT="${HGNET_ROOT}/classifier"
CONFIG_DIR="${CLASSIFIER_ROOT}/configs/hgnet_4545"
RESULT_ROOT="${HGNET_ROOT}/results_gaze045_light"

NUM_WORKERS="${NUM_WORKERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-}"

echo "============================================================"
echo "[HGNET 4545 THREE] export predictions"
echo "HGNET_ROOT      = ${HGNET_ROOT}"
echo "CLASSIFIER_ROOT = ${CLASSIFIER_ROOT}"
echo "CONFIG_DIR      = ${CONFIG_DIR}"
echo "RESULT_ROOT     = ${RESULT_ROOT}"
echo "PYTHON          = ${PYTHON}"
echo "NUM_WORKERS     = ${NUM_WORKERS}"
echo "BATCH_SIZE      = ${BATCH_SIZE:-<config default>}"
echo "============================================================"

if [ ! -d "${HGNET_ROOT}" ]; then
  echo "[ERROR] HGNET_ROOT not found: ${HGNET_ROOT}"
  exit 1
fi

if [ ! -d "${CLASSIFIER_ROOT}" ]; then
  echo "[ERROR] CLASSIFIER_ROOT not found: ${CLASSIFIER_ROOT}"
  exit 1
fi

if [ ! -d "${CONFIG_DIR}" ]; then
  echo "[ERROR] CONFIG_DIR not found: ${CONFIG_DIR}"
  exit 1
fi

if [ ! -f "${CLASSIFIER_ROOT}/src/evaluation/eval_only_save_predictions.py" ]; then
  echo "[ERROR] eval_only_save_predictions.py not found:"
  echo "        ${CLASSIFIER_ROOT}/src/evaluation/eval_only_save_predictions.py"
  exit 1
fi

cd "${CLASSIFIER_ROOT}"
export PYTHONPATH="${CLASSIFIER_ROOT}:${CLASSIFIER_ROOT}/configs:${PYTHONPATH:-}"

mkdir -p "${CLASSIFIER_ROOT}/logs"

CONFIGS=(
  "${CONFIG_DIR}/model4_occgateRAW_taskGatedLate_seed42_loss045.yaml"
  "${CONFIG_DIR}/model4_occgateRAW_taskRegionScalarGatedLate_seed42_loss045.yaml"
  "${CONFIG_DIR}/model4_occgateRAW_explicitRegionScalarMaskGate_seed42_loss045.yaml"
)

run_name_from_config () {
  local cfg="$1"
  basename "$cfg" .yaml
}

for CONFIG in "${CONFIGS[@]}"; do
  RUN_NAME="$(run_name_from_config "${CONFIG}")"
  RUN_DIR="${RESULT_ROOT}/${RUN_NAME}"
  CHECKPOINT="${RUN_DIR}/best.pt"
  LOG="${CLASSIFIER_ROOT}/logs/${RUN_NAME}_export_predictions_$(date +%Y%m%d_%H%M%S).log"

  echo
  echo "============================================================"
  echo "[EXPORT] ${RUN_NAME}"
  echo "CONFIG     = ${CONFIG}"
  echo "RUN_DIR    = ${RUN_DIR}"
  echo "CHECKPOINT = ${CHECKPOINT}"
  echo "LOG        = ${LOG}"
  echo "============================================================"

  if [ ! -f "${CONFIG}" ]; then
    echo "[ERROR] CONFIG not found: ${CONFIG}"
    exit 1
  fi

  if [ ! -d "${RUN_DIR}" ]; then
    echo "[ERROR] RUN_DIR not found: ${RUN_DIR}"
    exit 1
  fi

  if [ ! -f "${CHECKPOINT}" ]; then
    echo "[ERROR] CHECKPOINT not found: ${CHECKPOINT}"
    exit 1
  fi

  CMD=(
    "${PYTHON}" -m src.evaluation.eval_only_save_predictions
    --config "${CONFIG}"
    --checkpoint "${CHECKPOINT}"
    --splits test_clean test_masked
    --no-window-predictions
    --num-workers "${NUM_WORKERS}"
  )

  if [ -n "${BATCH_SIZE}" ]; then
    CMD+=(--batch-size "${BATCH_SIZE}")
  fi

  echo "[RUN] ${CMD[*]}"
  "${CMD[@]}" 2>&1 | tee "${LOG}"

  echo
  echo "------------------------------------------------------------"
  echo "[CHECK] generated prediction files for ${RUN_NAME}"
  echo "------------------------------------------------------------"
  find "${RUN_DIR}" -maxdepth 1 -type f -name "*predictions.csv" | sort

  echo "[DONE] ${RUN_NAME}"
done

echo
echo "============================================================"
echo "[SUMMARY] all generated prediction files"
echo "============================================================"

for CONFIG in "${CONFIGS[@]}"; do
  RUN_NAME="$(run_name_from_config "${CONFIG}")"
  RUN_DIR="${RESULT_ROOT}/${RUN_NAME}"
  COUNT="$(find "${RUN_DIR}" -maxdepth 1 -type f -name "*predictions.csv" | wc -l)"
  echo "${RUN_NAME}: ${COUNT}"
done

echo
echo "[ALL DONE] HGNET 4545 three prediction export complete."