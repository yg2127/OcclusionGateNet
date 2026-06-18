# Spatiotemporal DMS V5-style seed42 gaze045

Target run directory:

```bash
/data/shared/scuppy/hyi/Ablation/Compare/spatiotemporal
```

## Run

```bash
cd /data/shared/scuppy/hyi/Ablation/Compare/spatiotemporal
bash run_seed42.sh
```

Direct command:

```bash
python -m src.training.train   --config configs/spatiotemporal_seed42_gaze045_light.yaml   2>&1 | tee train_spatiotemporal_seed42_gaze045_light.log
```

## Export predictions

```bash
cd /data/shared/scuppy/hyi/Ablation/Compare/spatiotemporal
bash run_export_predictions.sh
```

## Key settings

- seed: 42 only
- patience: 4
- fixed split: `/data/shared/scuppy/hyi/fixed_splits/dms_clean_masked_fixed_items_v1.json`
- loss weights: action 0.45, gaze 0.45, hands 0.05, talk 0.05
- best score weights: action 0.45, gaze 0.45, hands 0.05, talk 0.05
- train/val/test variants: clean + masked

## Result path

```bash
/data/shared/scuppy/hyi/Ablation/Compare/spatiotemporal/artifacts_gaze045_light/spatiotemporal_seed42_gaze045_light
```
