# Task-gated late, no-gaze-OCC training package

Purpose: train/evaluate a task-gated late fusion variant where OCC is used for action/hands/talk gates, but NOT used by the gaze gate.

Default experiment:

- fusion kind: `task_gated_late_no_gaze_occ`
- OCC map: MediaPipe OCC npz map
- root expected on server: `/data/shared/scuppy/hyi/Ablation/AblationB`
- default seeds: `42 43 44`

## Install

Unzip so that the contents are placed directly under:

```bash
/data/shared/scuppy/hyi/Ablation/AblationB
```

The important added files are:

```text
configs/ablation_b_task_gated_no_gaze_occ_mediapipe_base.yaml
train/run_task_gated_no_gaze_occ_mediapipe.sh
README_TASK_GATED_NO_GAZE_OCC.md
```

This package also includes the required source changes for the new fusion kind.

## Run all 3 seeds

```bash
cd /data/shared/scuppy/hyi/Ablation/AblationB
bash train/run_task_gated_no_gaze_occ_mediapipe.sh
```

## Run only seed 43/44

```bash
cd /data/shared/scuppy/hyi/Ablation/AblationB
SEEDS="43 44" bash train/run_task_gated_no_gaze_occ_mediapipe.sh
```

## Logs

```bash
tail -f logs/ablation_b/ablation_b_mediapipe_task_gated_no_gaze_occ_task_gated_late_no_gaze_occ_seed42.log
```

## Artifacts

```text
artifacts/ablation_b_mediapipe_task_gated_no_gaze_occ_task_gated_late_no_gaze_occ_seed42/
artifacts/ablation_b_mediapipe_task_gated_no_gaze_occ_task_gated_late_no_gaze_occ_seed43/
artifacts/ablation_b_mediapipe_task_gated_no_gaze_occ_task_gated_late_no_gaze_occ_seed44/
```

## Summarize

```bash
PYTHONPATH=. /data/shared/envs/scuppy/bin/python tools/summarize_ablation_b_results.py \
  --root /data/shared/scuppy/hyi/Ablation/AblationB \
  --tag ablation_b_mediapipe_task_gated_no_gaze_occ
```

Expected output:

```text
analysis/ablation_b_mediapipe_task_gated_no_gaze_occ/ablation_b_seed_results_raw.csv
analysis/ablation_b_mediapipe_task_gated_no_gaze_occ/ablation_b_seed_results_mean_std.csv
```

## Interpretation

Compare against `task_gated_late`:

- If gaze masked F1 improves: direct OCC routing into gaze was likely destabilizing.
- If action/hands/talk remain stable: task-specific OCC routing is useful.
- If gaze does not improve: the gaze issue is likely crop/FaceMesh/gaze feature quality rather than OCC routing.
