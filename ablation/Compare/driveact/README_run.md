# DriveAct V5-style baseline (seed42, gaze045 light)

Target server path:

```bash
/data/shared/scuppy/hyi/Ablation/Compare/driveact
```

This package aligns the DriveAct comparison baseline with the V5 experiment policy:

- seed: `42` only
- patience: `4`
- epochs: `20`
- lr: `0.00075`
- batch_size: `64`
- num_workers: `6`
- fixed split JSON: `/data/shared/scuppy/hyi/fixed_splits/dms_clean_masked_fixed_items_v1.json`
- loss weights: action `0.45`, gaze `0.45`, hands `0.05`, talk `0.05`
- best score weights: action `0.45`, gaze `0.45`, hands `0.05`, talk `0.05`

## Run

```bash
cd /data/shared/scuppy/hyi/Ablation/Compare/driveact
unzip driveact_v5style_seed42_gaze045.zip
cd driveact_v5style_seed42_gaze045
bash run_seed42.sh
```

Direct command:

```bash
python train_driveact_multitask.py \
  --config configs/driveact_seed42_gaze045_light.yaml \
  2>&1 | tee train_driveact_seed42_gaze045_light.log
```

## Output

```text
/data/shared/scuppy/hyi/Ablation/Compare/driveact/artifacts_gaze045_light/driveact_seed42_gaze045_light
```

Expected key files:

- `config.json`
- `history.json`
- `metrics.csv`
- `summary.json`
- `best.pt`
- `split_info.json`
- `fixed_manifest_split_info.json`
- `test_clean_vs_masked_drop.json`
- `test_clean_vs_masked_drop.csv`
- per-head clean/masked confusion CSVs

## Export prediction CSVs

After training:

```bash
bash run_export_predictions.sh
```

If CPU RAM is tight, lower `train.batch_size` or `train.num_workers` in `configs/driveact_seed42_gaze045_light.yaml`.
