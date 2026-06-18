# Compare_SkateFormer

SkateFormer-inspired DMS comparison package.

## Purpose

This package keeps the existing DMS fixed clean/masked protocol and adds one new comparator:

- `skateformer_face`: YOLO pose skeleton SkateFormer-inspired stream + lightweight FaceMesh region stream.

It is not a full reproduction of the original SkateFormer code or training recipe. It adapts the core idea of SkateFormer—partition-specific skeletal-temporal attention over neighboring/distant joints and local/global motion—to the existing feature-level DMS setup.

## Server path

```bash
/data/shared/scuppy/hyi/Compare/Compare_SkateFormer
```

## Run

```bash
mkdir -p /data/shared/scuppy/hyi/Compare
unzip Compare_SkateFormer_FaceMesh.zip -d /data/shared/scuppy/hyi/Compare

cd /data/shared/scuppy/hyi/Compare/Compare_SkateFormer
bash train/run_skateformer_dms_seed_sweep.sh
```

## Real-time logs

```bash
tail -n 100 -f /data/shared/scuppy/hyi/Compare/Compare_SkateFormer/logs/skateformer_dms/skateformer_dms_skateformer_face_seed42.log
```

Seed 43/44:

```bash
tail -n 100 -f /data/shared/scuppy/hyi/Compare/Compare_SkateFormer/logs/skateformer_dms/skateformer_dms_skateformer_face_seed43.log
tail -n 100 -f /data/shared/scuppy/hyi/Compare/Compare_SkateFormer/logs/skateformer_dms/skateformer_dms_skateformer_face_seed44.log
```

## Data protocol retained

- fixed manifest: `/data/shared/scuppy/hyi/fixed_splits/dms_clean_masked_fixed_items_v1.json`
- train variants: clean + masked
- val variants: clean + masked
- test variants: clean + masked
- drop calculation: clean F1 - masked F1
- seeds: 42, 43, 44

## Model path

```text
src/models/fusion/skateformer.py
```

The default comparator uses:

```yaml
model:
  fusion:
    kind: skateformer_face
```

## FaceMesh handling

The original SkateFormer is a skeleton-based action recognition model. In this comparator, FaceMesh is added as a separate lightweight stream:

```text
pose_feat -> SkateFormer-inspired skeletal-temporal blocks -> pose_z
face_feat -> shallow 1x1 projection + pooling -> face_z
[pose_z, face_z] -> merge MLP -> action/gaze/hands/talk heads
```

The expensive shared branch is disabled:

```text
ConcatJointFusion / TGC shared branch: OFF
OCC / reliability: OFF
RGB frame input: OFF
```

## Outputs

Seed artifacts:

```text
artifacts/skateformer_dms_skateformer_face_seed42
artifacts/skateformer_dms_skateformer_face_seed43
artifacts/skateformer_dms_skateformer_face_seed44
```

Summary CSV:

```text
analysis/skateformer_dms/skateformer_dms_seed_results_raw.csv
analysis/skateformer_dms/skateformer_dms_seed_results_mean_std.csv
```
