# DMD Original baseline — V5-aligned seed42 gaze045-light

This package is refactored to follow the V5 comparison settings as closely as possible.

## Key settings

- seed: `42` only
- fixed split: `/data/shared/scuppy/hyi/fixed_splits/dms_clean_masked_fixed_items_v1.json`
- protocol: `clean_masked_augmentation_baseline`
- train/val/test variants: clean + masked fixed manifest
- loss weights: action `0.45`, gaze `0.45`, hands `0.05`, talk `0.05`
- best checkpoint score weights: action `0.45`, gaze `0.45`, hands `0.05`, talk `0.05`
- epochs: `20`
- patience: `4`
- batch size: `64`
- num_workers: `6`
- lr: `0.00075`
- save_every_epoch: `false`

## Run

Recommended location:

```bash
cd /data/shared/scuppy/hyi/Ablation/Compare/dmd_original
bash run_seed42.sh
```

Direct command:

```bash
python train_dmd_original_multitask.py   --config configs/dmd_original_seed42_gaze045_light.yaml   2>&1 | tee train_dmd_original_seed42_gaze045_light.log
```

## Output

Default output directory:

```text
/data/shared/scuppy/hyi/Ablation/Compare/dmd_original/artifacts_gaze045_light/dmd_original_seed42_gaze045_light
```

Expected output files include:

- `config.json`
- `history.json`
- `metrics.csv`
- `summary.json`
- `best.pt`
- `fixed_manifest_split_info.json`
- `test_clean_vs_masked_drop.json`
- `test_clean_vs_masked_drop.csv`
- per-head clean/masked confusion matrices
