# Ablation B: OCC fusion 위치 seed sweep

이 폴더는 `/data/shared/scuppy/hyi/Ablation/AblationB`를 프로젝트 root로 두고,
동일한 fixed clean/masked split, 동일한 Mesh OCC vector 조건에서 `model.fusion.kind`만 바꿔 학습하는 Ablation B용 코드입니다.

## 실험 목적

질문:

> 같은 backbone/crop/split/window 조건에서 OCC 정보를 어디에 넣는 것이 masked robustness에 가장 유리한가?

고정 조건:

- Backbone/face encoder: V4/V5 계열 `region_pool`, `region_reduce=mean`
- OCC crop: Mesh crop OCC map
- Split/window: `fixed_splits/dms_clean_masked_fixed_items_v1.json`
- Train/val: clean + masked
- Test: `test_clean`, `test_masked` 분리 평가

변경 조건:

- `model.fusion.kind`
- `seed`

## Sweep 대상

기본 sweep은 6개 fusion × 3 seeds입니다.

| name | fusion.kind | occ.enabled | 의미 |
|---|---|---:|---|
| `no_occ_original` | `concat` | false | no-OCC baseline |
| `concat_condition` | `concat_condition` | true | OCC feature-level condition |
| `task_gated_late` | `task_gated_late` | true | task별 pose-face reliability gate |
| `explicit_region_mask_gate` | `explicit_region_mask_gate` | true | region gate × OCC visibility mask |
| `occ_attention_bias` | `occ_attention_bias` | true | OCC attention bias |
| `task_region_scalar_gated_late` | `task_region_scalar_gated_late` | true | region gate + pose/face scalar gate |

기본 seed:

```txt
42, 43, 44
```

## 설치 / 배치

압축을 다음 위치에 풀면 됩니다.

```bash
mkdir -p /data/shared/scuppy/hyi/Ablation
cd /data/shared/scuppy/hyi/Ablation
unzip AblationB_seed_sweep.zip
cd AblationB
```

최종 root가 반드시 아래처럼 되도록 맞추면 됩니다.

```txt
/data/shared/scuppy/hyi/Ablation/AblationB
```

## 전체 sweep 실행

```bash
cd /data/shared/scuppy/hyi/Ablation/AblationB
bash train/run_ablation_b_seed_sweep.sh
```

내부적으로 실행되는 명령:

```bash
PYTHONPATH=. /data/shared/envs/scuppy/bin/python tools/run_ablation_b_seed_sweep.py \
  --root /data/shared/scuppy/hyi/Ablation/AblationB \
  --base-config configs/ablation_b_base.yaml \
  --python /data/shared/envs/scuppy/bin/python \
  --seeds 42 43 44 \
  --tag ablation_b \
  --skip-existing \
  --continue-on-error
```

## 일부만 실행

예: `task_gated_late`, `concat_condition`만 실행

```bash
PYTHONPATH=. /data/shared/envs/scuppy/bin/python tools/run_ablation_b_seed_sweep.py \
  --root /data/shared/scuppy/hyi/Ablation/AblationB \
  --base-config configs/ablation_b_base.yaml \
  --seeds 42 43 44 \
  --tag ablation_b \
  --only task_gated_late concat_condition \
  --skip-existing \
  --continue-on-error
```

## 생성되는 경로

Generated configs:

```txt
configs/generated_ablation_b/
```

Logs:

```txt
logs/ablation_b/
```

Artifacts:

```txt
artifacts/ablation_b_<fusion>_seed<seed>/
```

예:

```txt
artifacts/ablation_b_task_gated_late_seed42/
artifacts/ablation_b_task_gated_late_seed43/
artifacts/ablation_b_task_gated_late_seed44/
```

각 artifact 안에 `summary.json`, `test_clean_vs_masked_drop.csv`, `train.log` 등이 저장됩니다.

## 결과 요약

모든 실행 후 평균/표준편차 테이블 생성:

```bash
cd /data/shared/scuppy/hyi/Ablation/AblationB
PYTHONPATH=. /data/shared/envs/scuppy/bin/python tools/summarize_ablation_b_results.py \
  --root /data/shared/scuppy/hyi/Ablation/AblationB \
  --tag ablation_b
```

출력:

```txt
analysis/ablation_b/ablation_b_seed_results_raw.csv
analysis/ablation_b/ablation_b_seed_results_mean_std.csv
```

## 해석 기준

Ablation B의 핵심 비교는 다음입니다.

1. `no_occ_original` vs `task_gated_late`
   - OCC-aware reliability gate가 no-OCC보다 masked robustness를 줄이는지 확인
2. `concat_condition` vs `task_gated_late`
   - OCC를 단순 feature로 붙이는 것보다 task-level gate가 좋은지 확인
3. `explicit_region_mask_gate` vs `task_gated_late`
   - region을 직접 억제하는 방식이 gaze에서 불안정한지 확인
4. `occ_attention_bias` vs `task_gated_late`
   - attention-level intervention과 task-level reliability gate 비교
5. `task_region_scalar_gated_late` vs `task_gated_late`
   - V5 core region+scalar 구조와 단순 task gate 비교

보고서 표현은 다음처럼 쓰는 것이 안전합니다.

> 동일 fixed split 조건에서 seed를 달리해 반복 학습한 결과, OCC 정보를 task별 pose-face reliability gate에 사용하는 방식이 가장 안정적인 경향을 보였다.

통계 검정을 하지 않았다면 “통계적으로 유의하다”는 표현은 피하는 것이 좋습니다.
