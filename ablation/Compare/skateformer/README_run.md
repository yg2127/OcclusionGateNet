# SkateFormer DMS comparator - V5-style seed42 gaze045

Target server path:

```bash
/data/shared/scuppy/hyi/Ablation/Compare/skateformer
```

Main config:

```bash
configs/skateformer_seed42_gaze045_light.yaml
```

V5-aligned settings:

```yaml
seed: 42
train:
  epochs: 20
  patience: 4
  lr: 0.00075
  batch_size: 64
  num_workers: 6
  save_every_epoch: false
loss:
  alpha_action: 0.45
  alpha_gaze: 0.45
  alpha_hands: 0.05
  alpha_talk: 0.05
  gaze_weak_weight: 0.0
best_score_weights:
  action: 0.45
  gaze: 0.45
  hands: 0.05
  talk: 0.05
paths:
  fixed_items_json: /data/shared/scuppy/hyi/fixed_splits/dms_clean_masked_fixed_items_v1.json
```

Run:

```bash
cd /data/shared/scuppy/hyi/Ablation/Compare/skateformer
bash run_seed42.sh
```

Direct run:

```bash
python -m src.training.train   --config configs/skateformer_seed42_gaze045_light.yaml   2>&1 | tee train_skateformer_seed42_gaze045_light.log
```

Result path:

```bash
/data/shared/scuppy/hyi/Ablation/Compare/skateformer/artifacts_gaze045_light/skateformer_seed42_gaze045_light
```

Export predictions after training:

```bash
bash run_export_predictions.sh
```

Note: this package excludes previous artifacts/logs/results and keeps code/config/scripts only.
