# Full Integrated DMS System (runtime / integration layer)

This is the **real-time integration layer** that wires the trained pieces of this repository into a
single end-to-end Driver Monitoring System running on a face + body video pair:

- **Body**: YOLO-Pose → COCO17 skeleton
- **Face bbox**: YOLO-face bbox as the single face-side coordinate basis
- **FaceMesh**: MediaPipe FaceMesh on the YOLO bbox ROI → 478 landmarks
- **Occ CNN**: YOLO-bbox crop → 4-region visibility vector `[left_eye, right_eye, nose, mouth, crop_valid]`
- **ORFormer + HGNet**: occlusion-conditional landmark restoration (the **complete** restorer —
  ORFormer produces a reference edge heatmap that guides StackedHGNet; it is not HGNet-only)
- **Occ-gate**: no occlusion → keep MediaPipe raw; occlusion → replace only the occluded regions
  with the restored coordinates
- **DMS classifier**: the trained Model4 classifier over a 48-frame window → action / gaze / hands / talk

The system is designed to **still produce DMS results even when the face branch fails**. If the face
bbox / FaceMesh fails, it inserts zero FaceMesh and neutral visibility `[0.5, 0.5, 0.5, 0.5, 0.0]`,
then continues to the DMS classifier so the body branch can still contribute.

## How this maps to the rest of the repo

This bundle **reuses** the training-side code already in the repository instead of vendoring copies:

| runtime needs | reused from |
|---|---|
| Model4 classifier code (`build_model`, `preprocess_*`) | repo-root [`../classifier`](../classifier) |
| ORFormer + StackedHGNet model code | repo-root [`../landmark`](../landmark) |
| checkpoints (`*.pt`) | not tracked — see [`../models/MODELS.md`](../models/MODELS.md) |

The wrappers resolve `../classifier` and `../landmark` automatically; `configs/full_dms_config_template.yaml`
also sets them explicitly.

## Folder structure

```text
full_system/
├─ full_dms_system/                 # integrated runtime wrappers (8 stage modules + orchestrator)
│  ├─ full_system.py                #   FullDMSSystem orchestrator + FullDMSConfig
│  ├─ yolo_pose_skeleton_extractor.py
│  ├─ yolo_face_bbox_extractor.py
│  ├─ mediapipe_facemesh_yolo.py
│  ├─ occ_cnn_realtime.py
│  ├─ hgnet_restorer.py             #   ORFormer + HGNet restoration
│  ├─ occ_gate_merger.py
│  ├─ temporal_buffer.py
│  └─ dms_classifier_wrapper.py
├─ configs/
│  ├─ full_dms_config_template.yaml
│  └─ model4_occgateRAW_explicitRegionScalarMaskGate_seed42_loss045.yaml   # DMS classifier config (matches the checkpoint)
├─ scripts/                          # run on a video pair / overlay / TTS warnings / smoke test
├─ experiments/retrain_ablation/     # leave-one-module-out retraining ablation (code, configs, figures)
├─ notebooks/                        # facemesh + worst-case robustness views
├─ outputs/                          # small sample artifacts (large videos/predictions are gitignored)
└─ requirements.txt
```

## Checkpoints

Checkpoints are **not** tracked in git. Place them under `full_system/Model/` (matching the template)
or edit the paths. See [`../models/MODELS.md`](../models/MODELS.md) for each file's training provenance.

```yaml
models:
  yolo_pose_path: Model/yolo_pose.pt
  yolo_face_path: Model/yolo_face.pt
  occ_cnn_path:   Model/occ_cnn.pt
  orformer_ckpt:  Model/orformer.pt
  hgnet_ckpt:     Model/hgnet.pt
  dms_config_path: configs/model4_occgateRAW_explicitRegionScalarMaskGate_seed42_loss045.yaml
  dms_checkpoint_path: Model/dms_checkpoint.pt
```

`orformer_ckpt`, `hgnet_ckpt`, and `occ_cnn_path` are optional at runtime. If the restorer checkpoints
are missing, occluded frames fall back to MediaPipe landmarks. If the Occ CNN is missing, visibility is
neutral. Both degradations are logged and never stop inference.

## Run on two videos

```bash
cd full_system
python scripts/run_video_pair.py \
  --config configs/full_dms_config_template.yaml \
  --face-video /path/to/foo_ir_face.mp4 \
  --body-video /path/to/foo_ir_body.mp4 \
  --out-jsonl outputs/foo_predictions.jsonl
```

The first output appears after the 48-frame buffer is filled.

## Per-frame decision flow

```text
face_frame + body_frame
│
├─ Body: YOLO-Pose → skeleton (17,2), conf (17,)
│
└─ Face:
   ├─ YOLO-face bbox
   ├─ YOLO bbox crop → Occ CNN → [left_eye, right_eye, nose, mouth, crop_valid]
   ├─ YOLO bbox ROI → MediaPipe FaceMesh → 478 landmarks
   └─ if occluded:
        ├─ YOLO bbox crop → ORFormer (reference heatmap) → HGNet → restored 478 landmarks
        └─ replace only the occluded regions
      else:
        └─ keep MediaPipe FaceMesh unchanged

Recent 48 frames → Model4 DMS classifier → action / gaze / hands / talk
```

## Important notes

1. **YOLO bbox is the unified face coordinate basis.** MediaPipe FaceDetection is not used here.
2. **Face failure does not stop inference.** Zero landmarks + neutral occ features keep the body branch alive.
3. **DMS input preprocessing is reused** from `../classifier/src/data/preprocess_face.py` and `preprocess_pose.py`.
4. **Restoration is conditional.** ORFormer+HGNet runs only when the Occ CNN marks a region invisible —
   consistent with the occgateRAW logic: visible region = raw MediaPipe; occluded region = restored coords.

## Smoke test shapes

```bash
python scripts/smoke_test_shapes.py
```

This only checks buffer shapes; it does not load any models.
</content>
