# Occlusion Masking Subset (차폐 마스킹 서브셋)

OcclusionGateNet의 **차폐 강건성 학습·평가**를 위해 DMD 얼굴 영상에 **부위별 synthetic occlusion**을
합성해 구축한 데이터셋과 그 생성 코드. 원본 DMD만으로는 부위별 차폐를 통제해 평가하기 어렵기 때문에,
눈·입 등 주요 얼굴 영역을 다양한 appearance로 인위 차폐한 서브셋을 자체 구축했다.

이 서브셋은 두 곳에 쓰인다.
- **차폐 판단 CNN(Occlusion CNN) 학습** — 부위별 가시성 추정 (본 프로젝트 Macro-F1 0.9714)
- **분류기 차폐 평가** — clean vs masked 비교로 차폐 강건성 측정

![region × appearance grid](samples/occlusion_region_appearance_grid.png)

*행 = 차폐 region(clean / both_eyes / left_eye / right_eye / mouth / full_occlusion),
열 = appearance(blur_patch / checker / noise / smooth_noise / soft_noise / soft_solid / solid / stripe).*

## 구성 요소

| 파일 | 역할 |
|---|---|
| `make_region_occlusion_dataset.py` | **부위별 차폐 이미지셋 생성** — DMD FaceMesh로 얼굴 crop(256px) 후 region polygon에 synthetic mask 합성 → `images/{region}/{appearance}/*.jpg` + `labels.jsonl`. occ CNN 학습 데이터. |
| `make_masked_videos.py` | **마스킹 비디오 생성** — sunglasses(both/left/right)·lower/left/right-face-half 등 fixedmask 변형 영상 합성 → 분류기 차폐 평가용. |
| `make_occlusion_sample_grid.py` | region×appearance 샘플 그리드 시각화 생성기(위 그림). |
| `face_regions.py` | mediapipe 478 landmark의 **10-region anatomical** 인덱스 정의(LEFT_EYE/RIGHT_EYE/BROW/NOSE/MOUTH/FACE_OVAL/CHEEK_JAW…). |
| `face_regions7.py` | 위 10-region을 gaze 친화적 **7-region**(left_eye/right_eye/nose/mouth/contour/upper_face/lower_face)으로 재조합 + patch/region membership 유틸. |

## 라벨 형식 (`labels.jsonl`)

각 이미지 1줄 JSON. 핵심 필드:

```json
{"image_path": "...clean_facecrop256.jpg", "region": "left_eye", "appearance": "solid",
 "labels": {"left_eye": 1, "right_eye": 0, "mouth": 0}, "label_vector": [1,0,0],
 "frame_idx": 273, "crop_info": {"face_crop_xyxy": [...], "num_valid_landmarks": 478}}
```

- `region`/`appearance` = 어떤 부위를 어떤 패턴으로 가렸는지
- `labels` = 부위별 **가시성(occluded=1)** 정답 → occ CNN 학습 타깃
- `samples/labels_sample.jsonl` 에 앞부분 예시 40줄 포함.

## 샘플 (`samples/`)

레포에는 **소량 예시**만 포함한다(전체 데이터는 용량 문제로 미포함).
- `occlusion_region_appearance_grid.png` (+ `.csv`) — 위 그리드
- `images/` — region×appearance 대표 crop 12장
- `labels_sample.jsonl` — 라벨 40줄

전체 데이터셋(원본 경로, on-box):
`/data/shared/Occlusion_subset_dataset/region_occlusion_cnn_dataset_v2_facecrop_256/`
(그리고 마스킹 비디오: `/data/shared/DMD_landmarks/facemesh_masked_videos_v3/`).

## 실행 (on-box)

스크립트 상단의 경로 상수(`SRC_ROOT`, `VIDEO_ROOT`, `OUT_ROOT`)를 환경에 맞게 지정 후 실행:

```bash
python make_region_occlusion_dataset.py   # occ CNN 이미지셋
python make_masked_videos.py              # fixedmask 평가 영상
python make_occlusion_sample_grid.py      # 샘플 그리드
```

> 원본 코드 출처: `external_scripts/hyi_masking/` (마스킹 생성, hyi) · `Gaze_image_model/src/data/face_regions7.py`(yg).
> 학습된 occ CNN 및 model4/OcclusionGateNet 연계는 저장소 루트 `README.md`·`models/MODELS.md` 참조.
</content>
