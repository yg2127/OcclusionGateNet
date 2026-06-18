# DMD TSM-ResNet18 baseline — V5-style seed42 gaze/action weighted

This package is prepared for:

```bash
/data/shared/scuppy/hyi/Ablation/Compare/tsm_resnet18
```

## V5-style settings

- seed: `42` only
- fixed split JSON: `/data/shared/scuppy/hyi/fixed_splits/dms_clean_masked_fixed_items_v1.json`
- protocol: `clean_masked_augmentation_baseline`
- loss weights: action `0.45`, gaze `0.45`, hands `0.05`, talk `0.05`
- best checkpoint score weights: action `0.45`, gaze `0.45`, hands `0.05`, talk `0.05`
- patience: `4`
- epochs: `20`

Note: batch size is kept at `8` because this baseline decodes multi-view video frames directly and TSM-ResNet18 is much heavier than the V5 landmark model.

## Run

```bash
cd /data/shared/scuppy/hyi/Ablation/Compare/tsm_resnet18
bash run_seed42.sh
```

Equivalent direct command:

```bash
python train_dmd_tsm_resnet18_multitask.py \
  --config configs/tsm_resnet18_seed42_gaze045_light.yaml \
  2>&1 | tee train_tsm_resnet18_seed42_gaze045_light.log
```

## Output

Default output directory:

```bash
/data/shared/scuppy/hyi/Ablation/Compare/tsm_resnet18/artifacts_gaze045_light/tsm_resnet18_seed42_gaze045_light
```

Main files:

- `config.json`
- `run_meta.json`
- `split_info.json`
- `fixed_manifest_split_info.json`
- `metrics.csv`
- `history.json`
- `best.pt`
- `last.pt`
- `summary.json`
- `test_clean_vs_masked_drop.json`
- `test_clean_vs_masked_drop.csv`
- `test_clean_predictions.csv`
- `test_masked_predictions.csv`
