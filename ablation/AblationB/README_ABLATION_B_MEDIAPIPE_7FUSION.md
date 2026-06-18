# Ablation B - MediaPipe OCC 7-fusion seed sweep

이 패키지는 Ablation B를 **MediaPipe crop 기반 OCC npz map**으로 실행하기 위한 버전입니다.

Root 기준:

```bash
/data/shared/scuppy/hyi/Ablation/AblationB
```

## 핵심 조건

- 목적: OCC fusion 위치 비교
- OCC map: `/data/shared/Occlusion_subset_dataset/region_occlusion_video_dataset_v3_original_fixedmask_occ_pred_mediapipe/face_npz_to_occ_npz.json`
- 바꾸는 것: `model.fusion.kind`
- 고정: backbone, split/window, DMS face input path, hyperparameters

## 포함된 7개 실험

1. `no_occ_original` -> `fusion.kind=concat`, `occ.enabled=false`
2. `concat_condition`
3. `task_gated_late`
4. `explicit_region_mask_gate`
5. `occ_attention_bias`
6. `task_region_gated_late`
7. `task_region_scalar_gated_late`

기본 seed는 42, 43, 44입니다.

## 실행

```bash
cd /data/shared/scuppy/hyi/Ablation/AblationB
bash train/run_ablation_b_mediapipe_seed_sweep.sh
```

43, 44만 실행하려면:

```bash
cd /data/shared/scuppy/hyi/Ablation/AblationB
SEEDS="43 44" bash train/run_ablation_b_mediapipe_seed_sweep.sh
```

일부만 실행하려면:

```bash
PYTHONPATH=. /data/shared/envs/scuppy/bin/python tools/run_ablation_b_seed_sweep.py \
  --root /data/shared/scuppy/hyi/Ablation/AblationB \
  --base-config configs/ablation_b_base.yaml \
  --seeds 43 44 \
  --tag ablation_b_mediapipe \
  --only task_gated_late task_region_gated_late task_region_scalar_gated_late \
  --skip-existing \
  --continue-on-error
```

## 요약

```bash
PYTHONPATH=. /data/shared/envs/scuppy/bin/python tools/summarize_ablation_b_results.py \
  --root /data/shared/scuppy/hyi/Ablation/AblationB \
  --tag ablation_b_mediapipe
```

생성 결과:

```text
analysis/ablation_b_mediapipe/ablation_b_seed_results_raw.csv
analysis/ablation_b_mediapipe/ablation_b_seed_results_mean_std.csv
```
