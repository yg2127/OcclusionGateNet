# DMS 4545 Loss Prediction Export + Bootstrap Toolkit

## 목적

이미 학습된 `best.pt`를 다시 test해서, 부트스트래핑과 AUROC/AUPRC 계산이 가능한 per-sample prediction CSV를 저장한다.

대상 경로:

```bash
/data/shared/scuppy/hyi/Ablation/Ablation_Classification_V5
/data/shared/scuppy/hyi/Ablation/Compare
```

## 1) Prediction export

```bash
cd /data/shared/scuppy/hyi
mkdir -p bootstrap_4545
cp /path/to/dms_bootstrap_toolkit/*.sh bootstrap_4545/
cp /path/to/dms_bootstrap_toolkit/*.py bootstrap_4545/
cd bootstrap_4545

PYTHON=/data/shared/envs/scuppy/bin/python \
V5_ROOT=/data/shared/scuppy/hyi/Ablation/Ablation_Classification_V5 \
COMPARE_ROOT=/data/shared/scuppy/hyi/Ablation/Compare \
NUM_WORKERS=2 \
bash run_export_all_predictions.sh
```

V5만 돌릴 때:

```bash
ONLY=v5 bash run_export_all_predictions.sh
```

비교군만 돌릴 때:

```bash
ONLY=compare bash run_export_all_predictions.sh
```

## 2) Bootstrap + AUROC/AUPRC 후처리

```bash
PYTHON=/data/shared/envs/scuppy/bin/python \
python evaluate_bootstrap_from_predictions.py \
  --roots /data/shared/scuppy/hyi/Ablation/Ablation_Classification_V5 \
          /data/shared/scuppy/hyi/Ablation/Compare \
  --out-dir /data/shared/scuppy/hyi/Ablation/bootstrap_4545_results \
  --proposed task_region_scalar_gated_late \
  --baselines NoOcc dmd_original DriveAct task_gated_late attention_bias spatiotemporal pose_guided \
  --bootstrap-head gaze \
  --bootstrap-condition test_masked \
  --n-boot 5000 \
  --seed 42
```

## 출력 파일

`--out-dir` 아래에 생성된다.

- `prediction_manifest.csv`: 어떤 prediction 파일을 읽었는지 확인용
- `metrics_by_model_task_condition.csv`: condition별 accuracy, macro F1, weighted F1 등
- `roc_pr_by_model_task_condition.csv`: AUROC-OVR macro/weighted, AUPRC macro/weighted
- `clean_masked_drop_by_model_task.csv`: Clean/Masked drop, relative drop, PDI
- `bootstrap_pairwise_gaze_test_masked.csv`: Gaze masked F1 pairwise bootstrap 결과
- `bootstrap_relative_drop_gaze.csv`: Gaze relative drop pairwise bootstrap 결과

## 핵심 해석 기준

`bootstrap_pairwise_gaze_test_masked.csv`에서:

- `observed_delta_f1_macro > 0`: proposed가 baseline보다 실제 test에서 좋음
- `ci95_low_delta_f1_macro > 0`: bootstrap 95% CI가 양수라서 안정적 우위
- `win_rate_delta_f1_gt0`: bootstrap 반복 중 proposed가 이긴 비율
- `p_like_delta_f1_le0`: 0에 가까울수록 우위가 강함

`bootstrap_relative_drop_gaze.csv`에서:

- `mean_delta_relative_drop_baseline_minus_proposed > 0`: proposed의 relative drop이 baseline보다 작음
- `win_rate_lower_drop`: proposed가 더 낮은 drop을 보인 bootstrap 비율

## 주의

- 최종 F1 표만으로는 bootstrap이 불가능하다. 반드시 `*_clip_predictions.csv`가 필요하다.
- 모델 간 비교는 반드시 같은 `sample_id` intersection에서 paired bootstrap으로 수행한다.
- `best.pt`는 각 run directory 아래에 있어야 한다.
- AUROC/AUPRC는 `prob_*` column을 사용한다. 따라서 prediction export 단계에서 probability가 저장되어야 한다.
