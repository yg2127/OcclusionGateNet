# DriveAct baseline prediction export

This patch exports per-clip probability CSVs for ROC/PR/AUROC/AUPRC from the already trained DriveAct baseline checkpoint.

## Expected extracted baseline structure

```text
/data/shared/scuppy/baselines/driveact/
├── driveact_multitask.py
├── train_driveact_multitask.py
├── experiments/
│   └── driveact_fixed_clean_masked_seed42.yaml
└── runs/
    └── driveact_fixed_clean_masked_seed42/
        ├── best.pt
        ├── config.json
        ├── summary.json
        └── test_*_clip_confusion.csv
```

## Install patch files

```bash
cd /data/shared/scuppy/baselines/driveact
unzip /path/to/driveact_prediction_export_patch.zip -d .
chmod +x run_driveact_export_predictions.sh
```

## Run export

```bash
DRIVEACT_ROOT=/data/shared/scuppy/baselines/driveact \
RUN_DIR=/data/shared/scuppy/baselines/driveact/runs/driveact_fixed_clean_masked_seed42 \
PYTHON=/data/shared/envs/scuppy/bin/python \
bash /data/shared/scuppy/baselines/driveact/run_driveact_export_predictions.sh
```

Alternative Python:

```bash
PYTHON=/home/hyi/Code/.venv/bin/python bash run_driveact_export_predictions.sh
```

## Output files

```text
test_clean_predictions.csv
test_masked_predictions.csv
test_clean_action_clip_predictions.csv
test_masked_action_clip_predictions.csv
...
```

Required columns are exported as:

```text
sample_id, clip_id, split, head, level, n_windows, y_true, y_pred, prob_0, prob_1, ...
```

Then add the run root to the ROC/PR collection notebook:

```python
PRED_ARTIFACT_ROOTS = [
    Path('/data/shared/scuppy/baselines/driveact/runs'),
]
```
