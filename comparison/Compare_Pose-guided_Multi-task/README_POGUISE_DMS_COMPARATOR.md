# PO-GUISE-inspired DMS Comparator

이 패키지는 기존 Ablation 묶음이 아니라, 새 비교군 하나만 돌리기 위한 최소 패키지입니다.

## 비교군

`pose_guided_token_selection`

논문 *Pose-guided Multi-task Video Transformer for Driver Action Recognition*의 핵심인 **pose/class semantic token 기반 token selection**과 **dropped token merge** 아이디어를 현재 DMS feature pipeline에 맞게 축소 적용했습니다.

이 버전은 원 논문의 RGB/NIR VideoMAEv2 전체 재현이 아닙니다. 현재 프로젝트의 기존 입력 형태인 YOLO pose skeleton feature와 FaceMesh/region feature를 그대로 사용합니다.

## 남긴 것

학습과 최종 결과 산출에 필요한 파일만 남겼습니다.

```text
configs/poguise_dms_base.yaml
constants/
src/data/
src/evaluation/
src/experiment/
src/models/backbones/
src/models/fusion/pose_guided_token_selection.py
src/models/fusion/factory.py
src/models/fusion/task_feature_fusion.py
src/models/fusion/concat_joint.py
src/models/temporal/
src/training/
src/utils/
tools/run_poguise_dms_seed_sweep.py
tools/summarize_poguise_dms_results.py
train/run_poguise_dms_seed_sweep.sh
```

## 제거한 것

기존 Ablation용 yaml, queue script, reliability/OCC fusion 비교군, 문서 파일, pycache는 제거했습니다.

## 유지한 데이터/평가 프로토콜

```yaml
paths.fixed_items_json: /data/shared/scuppy/hyi/fixed_splits/dms_clean_masked_fixed_items_v1.json

data.use_fixed_items_manifest: true
data.train_variants: [clean, masked]
data.val_variants: [clean, masked]
data.test_variants: [clean, masked]

window.size: 48
window.stride: 24
face.encoder: region_pool
face.region_scheme: dms_10
```

즉, 기존 실험과 같은 fixed clean/masked split을 사용하고, 최종적으로 `test_clean`, `test_masked`, `clean - masked drop`을 냅니다.

## OCC/reliability 사용 여부

사용하지 않습니다.

```yaml
occ:
  enabled: false

model:
  fusion:
    kind: pose_guided_token_selection
```

`occ` 관련 경로 일부는 기존 loader/config 호환을 위해 yaml에 남아 있지만, `occ.enabled: false`이므로 모델 입력으로 들어가지 않습니다.

## 실행

```bash
cd /data/shared/scuppy/hyi
unzip POGUISE_DMS_Comparator_slim.zip -d PoseGuidedDMSComparator
cd /data/shared/scuppy/hyi/Compare/Compare_Pose-guided Multi-task

bash train/run_poguise_dms_seed_sweep.sh
```

직접 실행:

```bash
PYTHONPATH=$PWD /data/shared/envs/scuppy/bin/python tools/run_poguise_dms_seed_sweep.py \
  --root $PWD \
  --python /data/shared/envs/scuppy/bin/python \
  --seeds 42 43 44 \
  --tag poguise_dms \
  --continue-on-error

PYTHONPATH=$PWD /data/shared/envs/scuppy/bin/python tools/summarize_poguise_dms_results.py \
  --root $PWD \
  --tag poguise_dms \
  --out $PWD/analysis/poguise_dms
```

## 결과 파일

```text
artifacts/poguise_dms_pose_guided_token_selection_seed42/summary.json
artifacts/poguise_dms_pose_guided_token_selection_seed43/summary.json
artifacts/poguise_dms_pose_guided_token_selection_seed44/summary.json

analysis/poguise_dms/poguise_dms_seed_results_raw.csv
analysis/poguise_dms/poguise_dms_seed_results_mean_std.csv
```

기존 표에는 `fusion = pose_guided_token_selection` 행만 추가하면 됩니다.
