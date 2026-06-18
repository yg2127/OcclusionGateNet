# No-Gaze-OCC fusion variants

This patch adds fusion variants that **disable OCC conditioning only for the `gaze` head** while keeping OCC conditioning for `action`, `hands`, and `talk`.

## Added fusion kinds

- `task_gated_late_no_gaze_occ`
  - action/hands/talk scalar gates receive `x_occ`
  - gaze scalar gate receives only `pose_vec` and `face_vec`

- `task_region_gated_late_no_gaze_occ`
  - action/hands/talk region gates receive `x_occ`
  - gaze region gate receives only face region tokens

- `task_region_scalar_gated_late_no_gaze_occ`
  - action/hands/talk region/scalar gates receive `x_occ`
  - gaze region/scalar gates do not receive `x_occ`

## Run

```bash
cd /data/shared/scuppy/hyi/Ablation/AblationB
bash train/run_no_gaze_occ_sweep.sh
```

Only seeds 43 and 44:

```bash
SEEDS="43 44" bash train/run_no_gaze_occ_sweep.sh
```

Manual run:

```bash
PYTHONPATH=. /data/shared/envs/scuppy/bin/python tools/run_ablation_b_seed_sweep.py \
  --root /data/shared/scuppy/hyi/Ablation/AblationB \
  --base-config configs/ablation_b_base.yaml \
  --seeds 42 43 44 \
  --tag ablation_b_no_gaze_occ \
  --only task_gated_late_no_gaze_occ task_region_gated_late_no_gaze_occ task_region_scalar_gated_late_no_gaze_occ \
  --skip-existing \
  --continue-on-error
```

## Interpretation

These variants test whether gaze performance improves when OCC is removed from the gaze decision path, while preserving OCC-aware reliability gating for action/hands/talk.
