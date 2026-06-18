# Compare_Spatiotemporal

SDA-TR-inspired pose+FaceMesh comparator for the existing DMS clean/masked fixed-split protocol.

## Root path

```bash
/data/shared/scuppy/hyi/Compare/Compare_Spatiotemporal
```

## What this comparator tests

This package adapts the core idea of **Spatiotemporal Decoupling Attention Transformer (SDA-TR)** to the current DMS experiment scaffold:

- input protocol: same fixed clean/masked manifest as the previous Ablation-B-style experiments
- prediction evidence: YOLO pose skeleton only
- ignored evidence: FaceMesh, OCC, reliability scores
- model idea: decoupled spatiotemporal attention over skeleton joints + temporal feature aggregation
- output: same four heads, `action/gaze/hands/talk`
- evaluation: same `test_clean`, `test_masked`, and clean→masked drop summary

This is an implementation for comparison under your current feature-level DMS pipeline, not an exact MMAction2 reproduction of the paper.

## Run

```bash
mkdir -p /data/shared/scuppy/hyi/Compare
unzip Compare_Spatiotemporal_FaceMesh.zip -d /data/shared/scuppy/hyi/Compare
cd /data/shared/scuppy/hyi/Compare/Compare_Spatiotemporal
bash train/run_spatiotemporal_dms_seed_sweep.sh
```

## Live logs

```bash
tail -n 100 -f /data/shared/scuppy/hyi/Compare/Compare_Spatiotemporal/logs/spatiotemporal_dms/spatiotemporal_dms_spatiotemporal_decoupling_face_seed42.log
```

Seed 43/44 are the same path with `seed43.log` or `seed44.log`.

## Outputs

```text
artifacts/spatiotemporal_dms_spatiotemporal_decoupling_face_seed42/summary.json
artifacts/spatiotemporal_dms_spatiotemporal_decoupling_face_seed43/summary.json
artifacts/spatiotemporal_dms_spatiotemporal_decoupling_face_seed44/summary.json

analysis/spatiotemporal_dms/spatiotemporal_dms_seed_results_raw.csv
analysis/spatiotemporal_dms/spatiotemporal_dms_seed_results_mean_std.csv
```

## Important note

The dataloader may still read FaceMesh-related fields because the project dataset class is shared with the previous DMS experiments. However, the model path for this comparator sets `uses_shared = False`, skips the face/shared fusion path in `MultitaskClassifier.forward`, and classifies from the pose stream only.


## Lightweight FaceMesh update

This zip uses the FaceMesh-added comparator with the shared branch disabled. The prediction path is:

```text
pose_feat -> SDA/TFA stream -> pose_z
face_feat -> shallow face projection -> face_z
[pose_z, face_z] -> merge MLP -> action/gaze/hands/talk heads
```

The expensive `ConcatJointFusion + TGCBlock` shared branch is not used in this version.
