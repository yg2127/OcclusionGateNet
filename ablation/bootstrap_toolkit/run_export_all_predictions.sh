#!/usr/bin/env bash
set -euo pipefail

# Export per-sample test predictions from already trained best.pt checkpoints.
# This does NOT train. It only rebuilds test_clean/test_masked loaders and saves:
#   test_clean_<head>_clip_predictions.csv
#   test_masked_<head>_clip_predictions.csv
#   test_clean_predictions.csv
#   test_masked_predictions.csv
# These CSVs contain y_true, y_pred, prob_* and are enough for bootstrap/AUROC/AUPRC.

PYTHON="${PYTHON:-/data/shared/envs/scuppy/bin/python}"
V5_ROOT="${V5_ROOT:-/data/shared/scuppy/hyi/Ablation/Ablation_Classification_V5}"
COMPARE_ROOT="${COMPARE_ROOT:-/data/shared/scuppy/hyi/Ablation/Compare}"
NUM_WORKERS="${NUM_WORKERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-}"
ONLY="${ONLY:-all}"  # all | v5 | compare

log_section() {
  echo
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

run_v5() {
  log_section "[V5] export predictions"
  cd "$V5_ROOT"
  export PYTHONPATH="$V5_ROOT:${PYTHONPATH:-}"

  mkdir -p logs

  # Prefer gaze045_light configs because this is the 0.45/0.45 loss experiment.
  if [ -d "configs_gaze045_light" ]; then
    mapfile -t CFGS < <(find configs_gaze045_light -maxdepth 1 -name '*.yaml' | sort)
  else
    mapfile -t CFGS < <(find configs -maxdepth 1 -name '*.yaml' | sort)
  fi

  if [ "${#CFGS[@]}" -eq 0 ]; then
    echo "[V5][ERROR] no config yaml found under $V5_ROOT/configs_gaze045_light or configs"
    exit 1
  fi

  for CFG in "${CFGS[@]}"; do
    NAME="$(basename "$CFG" .yaml)"
    # save_root/best.pt is used by default inside eval_only_save_predictions.py.
    echo "[V5] $NAME"
    CMD=("$PYTHON" -m src.evaluation.eval_only_save_predictions
      --config "$CFG"
      --splits test_clean test_masked
      --no-window-predictions
      --num-workers "$NUM_WORKERS")
    if [ -n "$BATCH_SIZE" ]; then
      CMD+=(--batch-size "$BATCH_SIZE")
    fi
    echo "[RUN] ${CMD[*]}"
    "${CMD[@]}" 2>&1 | tee "logs/${NAME}_export_predictions.log"
  done
}

run_compare_one() {
  local NAME="$1"
  local ROOT="$COMPARE_ROOT/$NAME"
  local CONFIG="$ROOT/configs/${NAME}_seed42_gaze045_light.yaml"
  local RUN_DIR="$ROOT/artifacts_gaze045_light/${NAME}_seed42_gaze045_light"
  local EXPORT_SCRIPT=""

  case "$NAME" in
    dfs)
      CONFIG="$ROOT/experiments/dfs_fixed_clean_masked_seed42.yaml"
      RUN_DIR="$ROOT/artifacts_gaze045_light/dfs_seed42_gaze045_light"
      EXPORT_SCRIPT="$ROOT/export_dfs_predictions.py"
      ;;
    dmd_original)
      CONFIG="$ROOT/configs/dmd_original_seed42_gaze045_light.yaml"
      RUN_DIR="$ROOT/artifacts_gaze045_light/dmd_original_seed42_gaze045_light"
      EXPORT_SCRIPT="$ROOT/export_dmd_original_predictions.py"
      ;;
    driveact)
      CONFIG="$ROOT/configs/driveact_seed42_gaze045_light.yaml"
      RUN_DIR="$ROOT/artifacts_gaze045_light/driveact_seed42_gaze045_light"
      EXPORT_SCRIPT="$ROOT/export_driveact_predictions.py"
      ;;
    pose_guided)
      CONFIG="$ROOT/configs/pose_guided_seed42_gaze045_light.yaml"
      RUN_DIR="$ROOT/artifacts_gaze045_light/pose_guided_seed42_gaze045_light"
      EXPORT_SCRIPT="$ROOT/export_compare_predictions.py"
      ;;
    skateformer)
      CONFIG="$ROOT/configs/skateformer_seed42_gaze045_light.yaml"
      RUN_DIR="$ROOT/artifacts_gaze045_light/skateformer_seed42_gaze045_light"
      EXPORT_SCRIPT="$ROOT/export_compare_predictions.py"
      ;;
    spatiotemporal)
      CONFIG="$ROOT/configs/spatiotemporal_seed42_gaze045_light.yaml"
      RUN_DIR="$ROOT/artifacts_gaze045_light/spatiotemporal_seed42_gaze045_light"
      EXPORT_SCRIPT="$ROOT/export_compare_predictions.py"
      ;;
    *)
      echo "[COMPARE][SKIP] unknown project: $NAME"
      return 0
      ;;
  esac

  if [ ! -d "$ROOT" ]; then
    echo "[COMPARE][SKIP] root missing: $ROOT"
    return 0
  fi
  if [ ! -f "$CONFIG" ]; then
    echo "[COMPARE][SKIP] config missing: $CONFIG"
    return 0
  fi
  if [ ! -f "$RUN_DIR/best.pt" ]; then
    echo "[COMPARE][SKIP] best.pt missing: $RUN_DIR/best.pt"
    return 0
  fi
  if [ ! -f "$EXPORT_SCRIPT" ]; then
    echo "[COMPARE][SKIP] export script missing: $EXPORT_SCRIPT"
    return 0
  fi

  log_section "[COMPARE] $NAME"
  cd "$ROOT"
  export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

  mkdir -p "$RUN_DIR"

  if [ "$NAME" = "dfs" ]; then
    CMD=("$PYTHON" "$EXPORT_SCRIPT"
      --dfs-root "$ROOT"
      --run-dir "$RUN_DIR"
      --config "$CONFIG"
      --checkpoint "$RUN_DIR/best.pt"
      --splits test_clean test_masked
      --num-workers "$NUM_WORKERS")
  elif [ "$NAME" = "dmd_original" ]; then
    CMD=("$PYTHON" "$EXPORT_SCRIPT"
      --dmd-root "$ROOT"
      --run-dir "$RUN_DIR"
      --config "$CONFIG"
      --checkpoint "$RUN_DIR/best.pt"
      --splits test_clean test_masked
      --num-workers "$NUM_WORKERS")
  elif [ "$NAME" = "driveact" ]; then
    CMD=("$PYTHON" "$EXPORT_SCRIPT"
      --driveact-root "$ROOT"
      --run-dir "$RUN_DIR"
      --config "$CONFIG"
      --checkpoint "$RUN_DIR/best.pt"
      --splits test_clean test_masked
      --num-workers "$NUM_WORKERS")
  else
    CMD=("$PYTHON" "$EXPORT_SCRIPT"
      --project-root "$ROOT"
      --run-dir "$RUN_DIR"
      --config "$CONFIG"
      --checkpoint "$RUN_DIR/best.pt"
      --splits test_clean test_masked
      --num-workers "$NUM_WORKERS")
  fi

  if [ -n "$BATCH_SIZE" ]; then
    CMD+=(--batch-size "$BATCH_SIZE")
  fi

  echo "[RUN] ${CMD[*]}"
  "${CMD[@]}" 2>&1 | tee "$RUN_DIR/export_predictions.log"
}

run_compare() {
  log_section "[COMPARE] export predictions"
  for NAME in dfs dmd_original driveact pose_guided skateformer spatiotemporal; do
    run_compare_one "$NAME"
  done
}

case "$ONLY" in
  all)
    run_v5
    run_compare
    ;;
  v5)
    run_v5
    ;;
  compare)
    run_compare
    ;;
  *)
    echo "[ERROR] ONLY must be one of: all, v5, compare"
    exit 1
    ;;
esac

echo
find "$V5_ROOT" "$COMPARE_ROOT" -path '*_clip_predictions.csv' | sort | tail -100

echo "[DONE] prediction export complete."
