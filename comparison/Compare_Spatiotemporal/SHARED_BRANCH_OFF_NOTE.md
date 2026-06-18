# Spatiotemporal FaceMesh comparator: shared branch OFF

This package keeps the requested server root:

```bash
/data/shared/scuppy/hyi/Compare/Compare_Spatiotemporal
```

The active fusion kind remains:

```yaml
model:
  fusion:
    kind: spatiotemporal_decoupling_face
```

But the FaceMesh version has been lightened:

- uses YOLO pose skeleton through the SDA/TFA-inspired stream
- uses FaceMesh through `FaceRegionPool(dms_10) -> FaceBranch -> shallow 1x1 projection`
- disables the expensive shared branch:
  - no `ConcatJointFusion`
  - no `TGCBlock` shared stack
  - no `shared_z`
- final feature is `[pose_z, face_z] -> merge MLP -> four task heads`

So FaceMesh is still used, but the old full pose-face shared fusion branch is not used.
