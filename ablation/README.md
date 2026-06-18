# Ablation Study — occlusion-robust DMS

Model4/Model5(occgateRAW + ORFormer/HGNet 복원 + occ CNN)의 설계 선택을 검증하는 ablation 모음.
모두 본 저장소 `classifier/` 의 "V5 구조"를 fork 한 실험 스냅샷이다.

> 체크포인트(`artifacts*/`), per-clip 예측 덤프(`predictions/`, `*_clip_predictions.csv`), per-epoch confusion CSV,
> train 로그는 git 미추적. `results*/<run>/summary.json`·`metrics.csv`·`analysis/.../*_mean_std.csv`·
> bootstrap 요약 CSV·headline figure 만 포함. 출처: hyi (2026-05-24~06-02).

## 핵심: `AblationB/` — occ 정보 주입 위치 비교
"같은 backbone/crop/split/window 조건에서 occlusion 정보를 **어디에** 넣어야 masked robustness 가 좋은가?"
(6 fusion × 3 seeds). 결과: `analysis/ablation_b_mediapipe/ablation_b_seed_results_mean_std.csv`.

| fusion kind | occ | 요지 |
|---|---|---|
| `concat`(no_occ) | ✗ | 베이스라인 |
| `concat_condition` | ✓ | occ feature concat |
| `task_gated_late` | ✓ | task별 pose/face 신뢰도 게이트 — **gaze drop 최소** |
| `explicit_region_mask_gate` | ✓ | region게이트 × occ 가시성 마스크 |
| `occ_attention_bias` | ✓ | occ 를 attention bias 로 |
| `task_region_scalar_gated_late` | ✓ | region+scalar 게이트 |

## 보조 ablation (대부분 gaze head)
| 폴더 | 질문 |
|---|---|
| `Ablation_ex_478_only_gaze/` | 478 full-face 만으로 gaze (face 입력 품질 상한) |
| `Ablation_ex_Gaze_clean_to_clean/` | clean→clean (FaceMesh upper-bound) |
| `Ablation_ex_Gaze_face_only/` | pose 융합 제거(face만) — 병목 분리 |
| `Ablation_ex_Gaze_no_occ/` | gaze 에만 occ 라우팅 제거 (occ 가 gaze 를 불안정하게 하나?) |
| `Ablation_ex_1.2/` | 478노드 메모리 최적화 학습 |
| `Ablation_Classification_V5/` | no-occ 베이스라인 + gaze045 가중치 변형 |
| `HGNET_Classification/` | Model4/Model5 본체(occgateRAW). ※ 내부 `classifier/`·`landmark/` 는 repo 루트와 동일해 제외 |

## 비교 baseline + 통계검증
- `Compare/` — bootstrap 통계비교에 쓰인 7 baseline: `dfs`, `dmd_original`, `driveact`, `pose_guided`,
  `skateformer`, `spatiotemporal`, `tsm_resnet18` (gaze045_light 튜닝).
- `bootstrap_4545/` + `bootstrap_toolkit/` — 학습된 전 모델 예측 → n=5000 부트스트랩 → proposed vs baseline
  쌍별 95% CI·win-rate. 핵심표: `bootstrap_pairwise_<head>_<condition>.csv`.
- `Eval_Ablation/` — 전 ablation 통합 post-hoc 분석(요약 CSV + headline figure: PDI heatmap/scatter,
  clean-vs-masked, combined PR 등).

## 대표 결과 (gaze, clip macro-F1)
| 모델 | clean | masked | PDI |
|---|---|---|---|
| baseline (hgnet478 simple) | ~0.475 | — | — |
| Model4 (occgateRAW, task_gated_late) | 0.600 | 0.546 | +9.0% |
| V5 task_gated_late (mediapipe) | 0.613 | 0.582 | +4.9% |

PDI = (clean − masked)/clean × 100, 낮을수록 가림에 강건.
</content>
