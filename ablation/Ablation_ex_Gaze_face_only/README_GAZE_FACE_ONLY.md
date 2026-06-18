# Gaze Face-Only Baseline

Purpose: test whether the gaze head can be solved from FaceMesh/face-region features alone under the same clean/masked fixed protocol.

## Fusion kind

`task_gated_late_gaze_face_only`

- `gaze`: uses only `face_feat -> pooled face_vec -> face_proj -> gaze_head`
- `gaze` does **not** use pose, shared pose+face representation, OCC, or a scalar gate.
- `action/hands/talk`: kept structurally available but their loss weights are set to `0.0` in the base config.

## Training objective

`configs/gaze_face_only_facemesh_base.yaml` sets:

```yaml
occ:
  enabled: false
loss:
  alpha_action: 0.0
  alpha_gaze: 1.0
  alpha_hands: 0.0
  alpha_talk: 0.0
best_score_weights:
  action: 0.0
  gaze: 1.0
  hands: 0.0
  talk: 0.0
```

This makes the experiment a gaze-only face-feature baseline. The non-gaze head metrics in `summary.json` should not be interpreted.

## Suggested root

```bash
/data/shared/scuppy/hyi/Ablation/Ablation_ex_Gaze_face_only
```

## Run

```bash
cd /data/shared/scuppy/hyi/Ablation/Ablation_ex_Gaze_face_only
bash train/run_gaze_face_only_facemesh.sh
```

Run only seed 43/44:

```bash
SEEDS="43 44" bash train/run_gaze_face_only_facemesh.sh
```

## Summarize

```bash
PYTHONPATH=. /data/shared/envs/scuppy/bin/python tools/summarize_ablation_b_results.py \
  --root /data/shared/scuppy/hyi/Ablation/Ablation_ex_Gaze_face_only \
  --tag gaze_face_only_facemesh
```

Main values to compare:

- `gaze_masked_f1_mean`
- `gaze_drop_mean`
- clean/masked gaze F1 in each seed's `summary.json`

Compare against:

- no-OCC concat gaze
- `task_gated_late`
- `task_gated_late_no_gaze_occ`

Interpretation:

- If face-only gaze is close to or better than fusion gaze, pose/OCC routing is not the main bottleneck.
- If face-only gaze is still low, the bottleneck is likely masked FaceMesh/crop/face geometry quality.
