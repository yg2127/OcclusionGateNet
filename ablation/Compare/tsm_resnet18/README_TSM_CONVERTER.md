# DMD TSM-ResNet18 Prediction Converter

This baseline already contains prediction probability CSVs:

- `test_clean_predictions.csv`
- `test_masked_predictions.csv`

However, their columns are model-specific:

- `action_target`, `action_pred`, `action_p0...`
- `gaze_target`, `gaze_pred`, `gaze_p0...`
- `hands_target`, `hands_pred`, `hands_p0...`
- `talk_target`, `talk_pred`, `talk_p0...`

The common ROC/PR notebook expects:

- `y_true`, `y_pred`, `prob_0...`
- optionally one file per head:
  - `test_clean_action_clip_predictions.csv`
  - `test_masked_action_clip_predictions.csv`

Run:

```bash
cd /data/shared/scuppy/baselines/dmd_tsm_resnet18

PYTHON=/home/hyi/Code/.venv/bin/python
RUN_DIR=/data/shared/scuppy/baselines/dmd_tsm_resnet18/experiments/dmd_tsm_resnet18_fixed_clean_masked_augmentation_baseline_seed42_20260526_122623

$PYTHON convert_tsm_predictions_to_common.py --run-dir "$RUN_DIR"
```

Then add this root to the ROC/PR notebook:

```python
PRED_ARTIFACT_ROOTS = [
    Path("/data/shared/scuppy/baselines/dmd_tsm_resnet18/experiments"),
    Path("/data/shared/scuppy/baselines/dmd_original/experiments"),
    Path("/data/shared/scuppy/baselines/dfs/runs"),
    Path("/data/shared/scuppy/baselines/driveact/runs"),
    Path("/data/shared/scuppy/hyi/Classification_model_V5_transformer/artifacts"),
]
```
