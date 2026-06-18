# Models (로컬 전용 — git 미추적)

model4 inference·재현에 필요한 모델 가중치 모음(~268M, .gitignore).

## model4 실제 의존 그래프 (provenance 확인됨)

```
model4 분류기 (추론)
  ├─ x_face  ← occgateRAW 캐시(*_hgnet478_occgateRAW.npz)   [데이터, gitignore]
  │            └ 생성: ORFormer + HGNet **v3** + facemesh(clean GT) → Umeyama gating
  ├─ x_occ   ← occ_pred 캐시(*_occ.npz)                      [데이터, gitignore]
  │            └ 생성: **hyi Step9 occ CNN** (아래 주의)
  └─ x_body  ← yolo-pose 좌표                                [데이터]
* 추론 시 분류기 best.pt 만 로드. ORFormer/HGNet/occCNN 은 캐시 생성에만 사용(오프라인).
```

| 디렉토리 | 모델 | model4 사용처 | 비고 |
|---|---|---|---|
| `classifier_model4/` | 멀티태스크 분류기(ST-GCN) best.pt+config | **추론 본체** | ✅ model4 그 자체 |
| `hgnet_phase3a_v3/` | StackedHGNet v3 best.pt | **occgateRAW 가림좌표 생성** | ✅ model4 가 실제 사용한 variant |
| `orformer/` | ORFormer best.pt | HGNet reference (occgateRAW 생성) | ✅ |
| `facemesh/` | mediapipe tflite | clean 좌표 GT (occgateRAW 정상부위) | ✅ |
| `hgnet_phase3a/`, `hgnet_phase3a_v2/` | HGNet best/v2 | (model4 미사용 — 여분/실험용) | ⚠️ v3 아님 |
| `codebook/` | VQ-VAE codebook(phase1) | (orformer best.pt 에 포함 — 여분) | ⚠️ |
| `occ_cnn_step9_hyi/` | **VisibilityResNet18** (hyi, 4-label visibility) best.pt+생성코드 | **model4 의 x_occ 실제 생성** | ✅✅ |
| `occ_cnn_retrain_mine/` | 내가 재학습한 다른 occ CNN | (model4 무관) | ❌ |

## ⚠️ occ CNN 주의 (중요)

model4 의 `_occ_pred` 가림 게이팅 캐시는 **hyi 의 Step9 occ CNN** 으로 생성됨:
`/home/hyi/Code/Step9_extract_crop_npz/best.pt` (권한 거부 — 이 저장소에 미포함, 복제본 없음).

`occ_cnn_retrain_mine/best.pt` 는 내가 별도로 재학습한 occ CNN(val macroF1 0.958, face_crops_112 도메인)으로,
**model4 의 occ_pred 를 만든 모델이 아님**. inference_demo 의 `--occ` 데모용으로만 사용. model4 재현 시
occ_pred 는 이미 생성된 캐시(데이터)를 그대로 쓰므로 occ CNN 가중치는 불필요.

## inference
```bash
# landmark 복원 데모 (model4 와 동일한 HGNet v3 사용 권장)
python pipeline/inference_demo.py --image face.png --hgnet-ckpt models/hgnet_phase3a_v3/best.pt --occ
```
