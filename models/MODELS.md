# Runtime Checkpoints — `full_system/` 실시간 통합 시스템용

`full_system/` 실시간 파이프라인이 **온라인으로 로드**하는 체크포인트 목록.
모두 git 미추적(.gitignore). 실행하려면 아래 파일을 `full_system/Model/` 에 두거나
`configs/full_dms_config_template.yaml` 의 경로를 수정.

> 참고: 학습/오프라인 캐시 생성용 체크포인트의 상세 학습 설정은 [`PROVENANCE.md`](PROVENANCE.md) 와
> [`README.md`](README.md) 에 정리되어 있음. 여기서는 **런타임이 직접 로드하는 6개 파일**만 다룬다.
> 오프라인(model4 학습)에서는 ORFormer/HGNet/occ CNN 이 *캐시 생성*에만 쓰였지만,
> `full_system/` 런타임에서는 매 프레임 **직접 추론**에 쓰인다.

| 런타임 파일 (`full_system/Model/`) | 모델 | 역할 | 크기 | 출처 / provenance |
|---|---|---|---|---|
| `yolo_pose.pt`  | YOLO-Pose | body COCO17 skeleton 추출 | ~6 MB | hyi 학습 (Skeleton_extractor). 원본: `Full_System/Model/yolo_pose.pt` |
| `yolo_face.pt`  | YOLO-face | face bbox(단일 얼굴 좌표 기준) | ~6 MB | 사전학습 yolov8n-face 계열(외부) |
| `occ_cnn.pt`    | VisibilityResNet18 (4-label) | region별 가시성 → x_occ(5dim) | ~128 MB | **hyi Step9** occ CNN. → [`PROVENANCE.md` §7](PROVENANCE.md), `occ_cnn_step9_hyi/` |
| `orformer.pt`   | ORFormer (VQ-VAE + ViT) | HGNet용 reference edge heatmap | ~19 MB | yg, `pretrain_v4/.../phase2_orformer_fixed/best.pt`. → [`PROVENANCE.md` §1](PROVENANCE.md) |
| `hgnet.pt`      | StackedHGNet (4-stack, **v3**) | 가린 region 478 landmark 복원 | ~72 MB | yg, `pretrain_v4/.../phase3a_hgnet_478_v3/best.pt`. → [`PROVENANCE.md` §5](PROVENANCE.md) |
| `dms_checkpoint.pt` | Model4 분류기 (explicitRegionScalarMaskGate, gaze045) | action/gaze/hands/talk 4-head | ~28 MB | hyi, `results_gaze045_light/model4_occgateRAW_explicitRegionScalarMaskGate_seed42_loss045/best.pt`. config: `full_system/configs/model4_occgateRAW_explicitRegionScalarMaskGate_seed42_loss045.yaml` |

## 복원 모델은 ORFormer + HGNet (HGNet 단독 아님)

`full_system/full_dms_system/hgnet_restorer.py` 는 두 모델을 모두 로드해 **순차 추론**한다:

```
face crop(64×64) → ORFormer → reference edge heatmap
face crop(256×256) + reference heatmap → StackedHGNet → 478 landmarks
```

`orformer.pt` 가 없으면 복원기는 비활성화되고, 가린 프레임은 MediaPipe 좌표로 폴백한다
(추론은 멈추지 않음). 즉 `orformer.pt` + `hgnet.pt` 가 함께 있어야 "완전한" 복원 경로가 동작한다.

## 비고

- `occ_cnn.pt` 만 100 MB(GitHub 단일 파일 제한)를 넘는다. 전부 git 미추적이라 별도 전달 필요.
- `dms_checkpoint.pt` 는 `models/classifier_model4/` 의 `taskGated_occCNN` variant 와 다른
  **explicitRegionScalarMaskGate / gaze045** variant 다(런타임 데모에서 사용한 버전).
</content>
