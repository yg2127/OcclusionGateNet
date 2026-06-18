# Comparison Group (비교군) — external/SOTA baselines

우리 모델(Model4/Model5, occlusion-robust multitask DMS)과 **동일한 데이터·fixed split·48프레임 window·평가 프로토콜**로
재구현한 외부 비교 모델 3종. 모두 본 저장소 `classifier/` 의 "V5 구조"(같은 `src/training`·`src/data`·
`src/models/multitask_classifier` + fusion factory 교체식)를 fork 해, 무거운 shared branch 는 끄고
**fusion 모듈만 갈아끼운** 통제 비교(controlled comparison)다.

> 각 폴더는 실험 스냅샷이라 `classifier/` 와 겹치는 base 코드를 포함한다. 고유 변경분은 `CHANGED_FILES.txt` 참조.
> 체크포인트(`artifacts/`)와 per-clip 예측 덤프는 git 미추적이며, `results/<run>/summary.json` 과
> `analysis/.../*_mean_std.csv` 의 요약 지표만 포함했다. 출처: hyi (2026-05-28~29).

| 폴더 | 원논문 아이디어 | 핵심 fusion 파일 | params | 속도(ms/win) | clean→masked drop(weighted) | seeds |
|---|---|---|---|---|---|---|
| `Compare_SkateFormer/` | SkateFormer — partition-wise skeletal-temporal attention | `src/models/fusion/skateformer.py` | 1.29M | 66.0 | 0.032 | 42 |
| `Compare_Spatiotemporal/` | SDA-TR — graph-distance decoupled attention | `src/models/fusion/spatiotemporal_decoupling_face.py` | 9.82M | 13.6 | 0.033 | 42 |
| `Compare_Pose-guided_Multi-task/` | PO-GUISE — pose/class-guided token selection | `src/models/fusion/pose_guided_token_selection.py` | 3.12M | 6.7 | **0.016**±0.008 | 42,43,44 |

공통 사항:
- 입력: YOLO-17 pose + (경량) MediaPipe FaceMesh. **occlusion(가림) 신호는 미사용** — 우리 occgate 모델과의 핵심 대비점.
- 4개 head(action/gaze/hands/talk) 동일. 평가 지표는 clip-level macro-F1 + clean/masked drop.
- 실행: `bash <model>/train/run_*_dms_seed_sweep.sh` → `tools/summarize_*_dms_results.py`.
- 데이터/split: `/data/shared/DMD*`, `fixed_splits/dms_clean_masked_fixed_items_v1.json`.

(이 폴더의 3종은 standalone seed-sweep 패키지다. bootstrap 통계비교에 쓰인 더 넓은 baseline 집합 — DFS·DriveAct·TSM·
dmd_original 등 — 은 `ablation/Compare/` 에 있다.)
</content>
