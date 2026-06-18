#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Correctly collect DMS prediction CSVs into one central folder.
#
# Target:
#   /data/shared/scuppy/hyi/Ablation/bootstrap_4545/predictions/<model_name>/
#
# Sources:
#   1) Ablation_Classification_V5
#   2) Compare baselines
#   3) HGNET_Classification new 3 variants
#
# HGNET new variants:
#   - hgnet_task_gated_late
#   - hgnet_task_region_scalar_gated_late
#   - hgnet_explicit_region_scalar_mask_gate
#
# Expected per model:
#   test_clean_predictions.csv
#   test_masked_predictions.csv
#   test_clean_action_clip_predictions.csv
#   test_masked_action_clip_predictions.csv
#   test_clean_gaze_clip_predictions.csv
#   test_masked_gaze_clip_predictions.csv
#   test_clean_hands_clip_predictions.csv
#   test_masked_hands_clip_predictions.csv
#   test_clean_talk_clip_predictions.csv
#   test_masked_talk_clip_predictions.csv
# ============================================================

V5_ROOT="/data/shared/scuppy/hyi/Ablation/Ablation_Classification_V5"
COMPARE_ROOT="/data/shared/scuppy/hyi/Ablation/Compare"
HGNET_ROOT="/data/shared/scuppy/hyi/Ablation/HGNET_Classification"
HGNET_RESULT_ROOT="${HGNET_ROOT}/results_gaze045_light"

OUT_ROOT="/data/shared/scuppy/hyi/Ablation/bootstrap_4545/predictions"

mkdir -p "${OUT_ROOT}"

echo "============================================================"
echo "[COLLECT] prediction csv files"
echo "V5_ROOT           = ${V5_ROOT}"
echo "COMPARE_ROOT      = ${COMPARE_ROOT}"
echo "HGNET_ROOT        = ${HGNET_ROOT}"
echo "HGNET_RESULT_ROOT = ${HGNET_RESULT_ROOT}"
echo "OUT_ROOT          = ${OUT_ROOT}"
echo "============================================================"


# ------------------------------------------------------------
# Copy exactly from one run directory.
# ------------------------------------------------------------
copy_from_run_dir () {
  local model_name="$1"
  local run_dir="$2"

  local out_dir="${OUT_ROOT}/${model_name}"

  echo
  echo "------------------------------------------------------------"
  echo "[MODEL] ${model_name}"
  echo "[RUN DIR] ${run_dir}"
  echo "[OUT DIR] ${out_dir}"
  echo "------------------------------------------------------------"

  if [ ! -d "${run_dir}" ]; then
    echo "[ERROR] run dir not found: ${run_dir}"
    return 1
  fi

  rm -rf "${out_dir}"
  mkdir -p "${out_dir}"

  mapfile -t files < <(
    find "${run_dir}" -maxdepth 1 -type f \
      \( -name "test_clean_predictions.csv" \
         -o -name "test_masked_predictions.csv" \
         -o -name "test_clean_*_clip_predictions.csv" \
         -o -name "test_masked_*_clip_predictions.csv" \
         -o -name "test_clean_*_window_predictions.csv" \
         -o -name "test_masked_*_window_predictions.csv" \) \
      | sort
  )

  if [ "${#files[@]}" -eq 0 ]; then
    echo "[ERROR] no prediction csv found in ${run_dir}"
    echo "[DEBUG] files in run_dir:"
    find "${run_dir}" -maxdepth 1 -type f | sort || true
    return 1
  fi

  for src in "${files[@]}"; do
    cp "${src}" "${out_dir}/"
    echo "[COPY] $(basename "${src}")"
  done

  local count
  count="$(find "${out_dir}" -type f -name "*predictions.csv" | wc -l)"
  echo "[DONE] ${model_name}: copied ${count} files"

  if [ "${count}" -lt 8 ]; then
    echo "[WARN] ${model_name}: expected around 10 prediction files, got ${count}"
  fi
}


# ------------------------------------------------------------
# Discover V5 run directory by model-specific matcher.
# We search only directories that actually contain:
#   test_masked_gaze_clip_predictions.csv
# ------------------------------------------------------------
discover_v5_run_dir () {
  local model_name="$1"
  local matcher="$2"

  mapfile -t candidates < <(
    find "${V5_ROOT}" -type f -name "test_masked_gaze_clip_predictions.csv" \
      | xargs -r -n1 dirname \
      | sort -u
  )

  local matched=()

  for d in "${candidates[@]}"; do
    local base
    base="$(basename "${d}")"
    local low
    low="$(echo "${base}" | tr '[:upper:]' '[:lower:]')"

    case "${matcher}" in
      noocc)
        if [[ "${low}" =~ no[_-]?occ ]] || [[ "${low}" =~ no_occ ]] || [[ "${low}" =~ noocc ]]; then
          matched+=("${d}")
        fi
        ;;
      task_gated_late)
        if [[ "${low}" == *"task_gated_late"* ]]; then
          matched+=("${d}")
        fi
        ;;
      task_region_gated_late)
        if [[ "${low}" == *"task_region_gated_late"* && "${low}" != *"scalar"* ]]; then
          matched+=("${d}")
        fi
        ;;
      task_region_scalar_gated_late)
        if [[ "${low}" == *"task_region_scalar_gated_late"* ]] || [[ "${low}" == *"task_scalar_region_gated_late"* ]]; then
          matched+=("${d}")
        fi
        ;;
      explicit_region_mask_gate)
        if [[ "${low}" == *"explicit_region_mask_gate"* && "${low}" != *"scalar"* ]]; then
          matched+=("${d}")
        fi
        ;;
      explicit_region_scalar_mask_gate)
        if [[ "${low}" == *"explicit_region_scalar_mask_gate"* ]]; then
          matched+=("${d}")
        fi
        ;;
      attention_bias)
        if [[ "${low}" == *"attention_bias"* ]] || [[ "${low}" == *"occ_attention_bias"* ]]; then
          matched+=("${d}")
        fi
        ;;
      occ_token_region_transformer)
        if [[ "${low}" == *"occ_token_region_transformer"* ]]; then
          matched+=("${d}")
        fi
        ;;
      *)
        echo "[ERROR] unknown matcher: ${matcher}"
        return 1
        ;;
    esac
  done

  if [ "${#matched[@]}" -eq 0 ]; then
    echo "[ERROR] V5 run dir not found for ${model_name} matcher=${matcher}" >&2
    echo "[DEBUG] available V5 run dirs containing test_masked_gaze_clip_predictions.csv:" >&2
    printf '  %s\n' "${candidates[@]}" >&2
    return 1
  fi

  if [ "${#matched[@]}" -gt 1 ]; then
    echo "[WARN] multiple V5 run dirs matched for ${model_name}. Choosing best-scored candidate:" >&2
    printf '  %s\n' "${matched[@]}" >&2
  fi

  # Prefer gaze045/loss045-looking paths if multiple.
  printf '%s\n' "${matched[@]}" \
    | awk '
      {
        score=0
        if ($0 ~ /gaze045/) score+=10
        if ($0 ~ /loss045/) score+=10
        if ($0 ~ /artifacts_gaze045_light/) score+=5
        print score "\t" length($0) "\t" $0
      }
    ' \
    | sort -k1,1nr -k2,2n \
    | head -n 1 \
    | cut -f3-
}


copy_v5_model () {
  local model_name="$1"
  local matcher="$2"

  local run_dir
  run_dir="$(discover_v5_run_dir "${model_name}" "${matcher}")"
  copy_from_run_dir "${model_name}" "${run_dir}"
}


copy_compare_model () {
  local model_name="$1"
  local run_dir="$2"

  copy_from_run_dir "${model_name}" "${run_dir}"
}


copy_hgnet_model () {
  local model_name="$1"
  local run_name="$2"

  local run_dir="${HGNET_RESULT_ROOT}/${run_name}"
  copy_from_run_dir "${model_name}" "${run_dir}"
}


echo
echo "============================================================"
echo "[V5] collect ablation prediction files"
echo "============================================================"

copy_v5_model "NoOcc" "noocc"
copy_v5_model "task_gated_late" "task_gated_late"
copy_v5_model "task_region_gated_late" "task_region_gated_late"
copy_v5_model "task_region_scalar_gated_late" "task_region_scalar_gated_late"
copy_v5_model "explicit_region_mask_gate" "explicit_region_mask_gate"
copy_v5_model "explicit_region_scalar_mask_gate" "explicit_region_scalar_mask_gate"
copy_v5_model "attention_bias" "attention_bias"
copy_v5_model "occ_token_region_transformer" "occ_token_region_transformer"


echo
echo "============================================================"
echo "[COMPARE] collect baseline prediction files"
echo "============================================================"

copy_compare_model "dfs" \
  "${COMPARE_ROOT}/dfs/runs/dfs_fixed_clean_masked_seed42_loss045"

copy_compare_model "dmd_original" \
  "${COMPARE_ROOT}/dmd_original/artifacts_gaze045_light/dmd_original_seed42_gaze045_light"

copy_compare_model "DriveAct" \
  "${COMPARE_ROOT}/driveact/artifacts_gaze045_light/driveact_seed42_gaze045_light"

copy_compare_model "pose_guided" \
  "${COMPARE_ROOT}/pose_guided/artifacts_gaze045_light/pose_guided_seed42_gaze045_light"

copy_compare_model "skateformer" \
  "${COMPARE_ROOT}/skateformer/artifacts_gaze045_light/skateformer_seed42_gaze045_light"

copy_compare_model "spatiotemporal" \
  "${COMPARE_ROOT}/spatiotemporal/artifacts_gaze045_light/spatiotemporal_seed42_gaze045_light"


echo
echo "============================================================"
echo "[HGNET] collect new 3 prediction files"
echo "============================================================"

copy_hgnet_model "hgnet_task_gated_late" \
  "model4_occgateRAW_taskGatedLate_seed42_loss045"

copy_hgnet_model "hgnet_task_region_scalar_gated_late" \
  "model4_occgateRAW_taskRegionScalarGatedLate_seed42_loss045"

copy_hgnet_model "hgnet_explicit_region_scalar_mask_gate" \
  "model4_occgateRAW_explicitRegionScalarMaskGate_seed42_loss045"


echo
echo "============================================================"
echo "[SUMMARY] collected files"
echo "============================================================"
find "${OUT_ROOT}" -type f -name "*predictions.csv" | sort

echo
echo "============================================================"
echo "[COUNT BY MODEL]"
echo "============================================================"
for d in "${OUT_ROOT}"/*; do
  [ -d "$d" ] || continue
  model="$(basename "$d")"
  count="$(find "$d" -type f -name "*predictions.csv" | wc -l)"
  echo "${model}: ${count}"
done


echo
echo "============================================================"
echo "[MD5 CHECK | V5 test_masked_gaze_clip_predictions.csv]"
echo "If these are all identical, collection is still wrong."
echo "============================================================"

for m in \
  NoOcc \
  task_gated_late \
  task_region_gated_late \
  task_region_scalar_gated_late \
  explicit_region_mask_gate \
  explicit_region_scalar_mask_gate \
  attention_bias \
  occ_token_region_transformer
do
  f="${OUT_ROOT}/${m}/test_masked_gaze_clip_predictions.csv"
  if [ -f "${f}" ]; then
    md5sum "${f}"
  else
    echo "[MISSING] ${f}"
  fi
done


echo
echo "============================================================"
echo "[MD5 CHECK | HGNET new 3 test_masked_gaze_clip_predictions.csv]"
echo "If these are all identical, collection may be wrong."
echo "============================================================"

for m in \
  hgnet_task_gated_late \
  hgnet_task_region_scalar_gated_late \
  hgnet_explicit_region_scalar_mask_gate
do
  f="${OUT_ROOT}/${m}/test_masked_gaze_clip_predictions.csv"
  if [ -f "${f}" ]; then
    md5sum "${f}"
  else
    echo "[MISSING] ${f}"
  fi
done


echo
echo "============================================================"
echo "[QUICK METRIC CHECK | gaze masked macro F1]"
echo "This should roughly match your 4545 / HGNET summary values."
echo "============================================================"

python - <<'PY'
from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score

root = Path("/data/shared/scuppy/hyi/Ablation/bootstrap_4545/predictions")
models = [
    "NoOcc",
    "task_gated_late",
    "task_region_gated_late",
    "task_region_scalar_gated_late",
    "explicit_region_mask_gate",
    "explicit_region_scalar_mask_gate",
    "attention_bias",
    "occ_token_region_transformer",
    "dfs",
    "dmd_original",
    "DriveAct",
    "pose_guided",
    "skateformer",
    "spatiotemporal",
    "hgnet_task_gated_late",
    "hgnet_task_region_scalar_gated_late",
    "hgnet_explicit_region_scalar_mask_gate",
]

rows = []
for m in models:
    p = root / m / "test_masked_gaze_clip_predictions.csv"
    if not p.exists():
        rows.append((m, "MISSING", ""))
        continue
    df = pd.read_csv(p)
    f1 = f1_score(df["y_true"], df["y_pred"], average="macro", zero_division=0)
    rows.append((m, f"{f1:.4f}", len(df)))

print(f"{'model':45s} {'gaze_masked_f1':>15s} {'n':>8s}")
for m, f1, n in rows:
    print(f"{m:45s} {str(f1):>15s} {str(n):>8s}")
PY

echo
echo "[DONE] Prediction collection finished."