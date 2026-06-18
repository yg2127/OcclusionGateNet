# Occlusion-Robust Driver Monitoring (model4 / model5)

DMD 운전자 모니터링용 **멀티태스크 분류 모델**(action·gaze·hands·talk) 와 그 입력을 가림(occlusion)에 강건하게 만드는
**랜드마크 복원 파이프라인**의 전체 구현. 핵심 기여는 *가려진 얼굴 영역의 좌표를 HGNet 으로 복원해 끼워넣는 occlusion-gated
face 입력(occgateRAW)* 으로, 가림 상황에서 gaze 성능 저하(PDI)를 크게 줄인다.

## 핵심 아이디어

얼굴 랜드마크 좌표만으로 gaze/action 을 분류하면 **가림(선글라스·마스크·패치)** 시 mediapipe facemesh 가 무너져 성능이 급락한다.
이를 막기 위해:

1. **정상 영역** → mediapipe facemesh 원본 좌표 (iris 정밀도 보존)
2. **가린 영역** → ORFormer+HGNet 으로 복원한 좌표를 facemesh 좌표계로 정합(Umeyama)해 치환
3. region 별 가림 여부는 **occlusion CNN** 또는 manifest GT 로 판정
4. 이렇게 만든 **occgateRAW** 좌표를 멀티태스크 분류기의 face branch 입력으로 사용

`model4` = `task_region_scalar_gated_late` fusion + occgateRAW + loss/best-score 0.45/0.45/0.05/0.05. `model5` = `task_region_scalar_gated_late` fusion + fixedmask-finetuned HGNet 좌표.

## 저장소 구조

```
model4_dms/
├── classifier/        # 멀티태스크 DMS 분류기 (action/gaze/hands/talk)
│   ├── src/
│   │   ├── models/multitask_classifier.py   # 메인 모델 (pose GCN + face branch + fusion + 4 head)
│   │   ├── models/fusion/                    # occlusion-aware fusion 모듈들
│   │   │   ├── factory.py                    #   fusion kind 선택
│   │   │   ├── task_feature_fusion.py        #   task_gated_late      (model4)
│   │   │   ├── task_region_scalar_fusion.py  #   task_region_scalar_gated_late (model5)
│   │   │   ├── explicit_region_mask_gate.py / occ_attention_bias.py / occ_token_region_transformer.py
│   │   │   └── region_occ_utils.py
│   │   ├── data/        # dataset.py(window/npz_swap 로딩), clip_builder, window, fixed_manifest_split, preprocess_*
│   │   ├── training/    # train.py(엔트리), runner, loops, builders, aggregation
│   │   └── evaluation/  # v1_eval
│   ├── constants/       # gaze_zones(9-class), face_regions
│   ├── configs/         # base + fusion variants + generated/(model4·model5 yaml)
│   └── scripts/         # build_hgnet_cache_from_hyi_split.py 등
│
├── landmark/           # 랜드마크 복원 (ORFormer + StackedHGNet) — pretrain_v4
│   ├── src/models/      # StackedHGNet.py, VQVAE.py, simple_vit.py(ORFormer), quantizer.py
│   ├── src/data/        # dataset_dmd.py, heatmap_gen.py, augmentation.py, face_edge_info.py
│   ├── configs/         # default.py (DMD/DMD_68 cfg)
│   ├── scripts/         # train_phase{1,2,3}_*.py (codebook→ORFormer→HGNet 학습)
│   └── docs/            # METHOD_DIFF, TRAINING_LOG
│
└── pipeline/           # occlusion gating 파이프라인 + 실험 스크립트 (occ_cnn_v1)
    ├── build_occgate_RAW_cache.py        # occgateRAW 좌표 캐시 (정상=facemesh / 가림=hgnet→facemesh)
    ├── build_occgate_RAW_ft_cache.py     #   finetuned HGNet 버전 (model5)
    ├── build_occgate_GT_cache.py         #   occgateGT (hgnet 좌표계 버전, 비교용)
    ├── regen_hgnet478_ft_masked.py       # finetuned HGNet 으로 masked 좌표 재생성
    ├── finetune_hgnet_fixedmask.py       # HGNet 을 8 appearance 가림으로 finetune
    ├── train_occ_cnn.py                  # region occlusion CNN 학습 (TinyRegionCNN)
    ├── verify_ft_nme.py / compare_orformer_vs_hgnet.py   # 랜드마크 NME 검증
    ├── make_*_notebook.py                # 시각화 노트북 생성기
    └── notebooks/                        # 결과 시각화 (gaze 사례, occlusion NME 진단)
```

## 파이프라인 (end-to-end)

### 추론 파이프라인 (occlusion gating)

```mermaid
flowchart TD
    IMG["face frame"] --> FM["mediapipe FaceMesh<br/>478 landmarks"]
    IMG --> OCC["occlusion CNN<br/>region별 가림 판정"]
    IMG --> HG["ORFormer + HGNet<br/>landmark 복원"]

    HG --> ALIGN["Umeyama 정합<br/>hgnet → facemesh 좌표계"]

    FM --> GATE{"region<br/>가림?"}
    OCC --> GATE
    ALIGN --> GATE

    GATE -- "정상" --> KEEP["facemesh 좌표 유지<br/>(iris 정밀도 보존)"]
    GATE -- "가림" --> SWAP["복원 좌표로 치환"]
    KEEP --> OCCGATE["occgateRAW landmarks"]
    SWAP --> OCCGATE

    OCCGATE --> FACE["face branch"]
    POSE["body pose"] --> GCN["pose GCN"]
    FACE --> FUSE["task_region_scalar_gated_late fusion<br/>(model4)"]
    GCN --> FUSE
    FUSE --> H1["action"]
    FUSE --> H2["gaze (9-zone)"]
    FUSE --> H3["hands"]
    FUSE --> H4["talk"]
```

### 학습 / 캐시 생성 순서

```
[학습된 자산]  ORFormer(landmark/) + HGNet(landmark/) + occ CNN(pipeline/)
       │
       ▼
1) HGNet 좌표 캐시      classifier/scripts/build_hgnet_cache_from_hyi_split.py   → *_hgnet478.npz
2) occgateRAW 캐시      pipeline/build_occgate_RAW_cache.py                      → *_hgnet478_occgateRAW.npz
3) 분류기 학습          classifier/src/training/train.py --config <model4.yaml>
   (face npz_swap 으로 occgateRAW 좌표를 face branch 입력으로 로드)

[model5 추가 단계]
4) HGNet fixedmask finetune   pipeline/finetune_hgnet_fixedmask.py
5) 좌표 재생성 + occgateRAW_ft pipeline/regen_hgnet478_ft_masked.py → build_occgate_RAW_ft_cache.py
6) model5 학습                train.py --config <model5.yaml>
```

분류기 학습 실행 예:
```bash
cd classifier
PYTHONPATH=$(pwd):$(pwd)/configs python src/training/train.py \
  --config configs/generated/model4_occgateRAW_taskRegionScalarGatedLate_seed42_loss045.yaml
```


## Modified config note

이 배포본은 `model4_dms`를 `task_region_scalar_gated_late` 버전으로 맞춘 패키지다. 주요 변경은 다음과 같다.

- fusion kind: `task_region_scalar_gated_late`
- loss weights: action/gaze/hands/talk = `0.45 / 0.45 / 0.05 / 0.05`
- best score weights: action/gaze/hands/talk = `0.45 / 0.45 / 0.05 / 0.05`
- early stopping patience: `4`
- primary config: `classifier/configs/model4_occgateRAW_taskRegionScalarGatedLate_seed42_loss045.yaml`
- backward-compatible copied config: `classifier/configs/model4_occgateRAW_taskGated_occCNN_seed42.yaml`도 같은 설정으로 덮어씀

## 주요 결과 (gaze head, clip-level macro-F1)

| 모델 | fusion | face 입력 | clean | masked | PDI |
|---|---|---|---|---|---|
| baseline (hgnet478 단순) | concat | hgnet 좌표 | ~0.475 | — | — |
| **model4** | task_gated_late | occgateRAW | **0.600** | **0.546** | +9.0% |
| hyi v5_task_gated_late | task_gated_late | mediapipe | 0.613 | 0.582 | +4.9% |
| hyi task_region_scalar (gaze045) | task_region_scalar_gated_late | mediapipe | 0.577 | 0.555 | +3.8% |
| **model5** (예정) | task_region_scalar_gated_late | occgateRAW_ft | TBD | TBD | TBD |

- PDI = (clean − masked) / clean × 100. 낮을수록 가림에 강건.
- occgateRAW 가 가림 부위 좌표를 HGNet 으로 복원해 masked gaze 와 robust 를 개선.

### 오분류 사례 시각화

가림(선글라스·마스크·노이즈/체커 패치)으로 facemesh 가 무너져 gaze zone 을 혼동하는 대표 사례.
`[X]` = 오분류, `occ le/re/mo` = 좌안/우안/입 region 가림도, `no landmark` = facemesh 검출 실패.

![occlusion confusion cases](docs/assets/confusion_cases.png)

## 모델 가중치 (`models/` — 로컬 전용, git 미추적)

모든 모델은 `models/` 에 수집됨(occ CNN·ORFormer·codebook·HGNet phase3a/v2/v3·분류기·facemesh tflite, ~268M). 용량 때문에 .gitignore. 상세는 `models/README.md`. inference: `python pipeline/inference_demo.py --image face.png --occ`.

## 기타 외부 자산 (저장소·models 미포함)

대용량 자산은 git 에 넣지 않음(.gitignore). 재현 시 아래 경로/체크포인트 필요:

| 항목 | 경로 |
|---|---|
| ORFormer 라이브러리(vendor) | `/data/shared/orformer/vendor` (PYTHONPATH 추가) |
| ORFormer 가중치 | `pretrain_v4/artifacts/phase2_orformer_fixed/best.pt` |
| HGNet 가중치 (model4=**v3**) | `pretrain_v4/artifacts/phase3a_hgnet_478_v3/best.pt` |
| HGNet fixedmask finetune | `scuppy/yg/hgnet_fixedmask_ft/best.pt` |
| occ CNN (model4 occ_pred 생성) | **hyi** `/home/hyi/Code/Step9_extract_crop_npz/best.pt` (권한거부·미포함) |
| DMD 원본 영상/랜드마크 | `/data/shared/DMD`, `/data/shared/DMD_landmarks` |
| 가림 데이터셋 | `/data/shared/Occlusion_subset_dataset/region_occlusion_video_dataset_v3_original_fixedmask` |
| fixed split manifest | `/data/shared/scuppy/hyi/fixed_splits/dms_clean_masked_fixed_items_v1.json` |

> 코드 내 일부 절대경로는 위 자산을 가리킴. 다른 환경에서 재현 시 경로를 맞춰야 함.

## 데이터셋

DMD (Driver Monitoring Dataset) — distraction(action/gaze_weak/hands/talk) + gaze(s6, 9-zone fine) 세션.
subject-disjoint fixed split (clean/masked 1:1 paired). 가림은 6 zero-paint variant + fixedmask 8 appearance(solid/noise/checker/stripe/blur 등).

## 요구사항

`requirements.txt` 참조. 핵심: PyTorch, torchvision, OpenCV, mediapipe(facemesh GT), scikit-learn, matplotlib, PyYAML.
