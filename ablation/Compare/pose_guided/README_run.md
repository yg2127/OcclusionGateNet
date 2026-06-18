# Pose-guided V5-style baseline (seed 42, gaze/action weighted)

Target server path:

```bash
/data/shared/scuppy/hyi/Ablation/Compare/pose_guided
```

Run:

```bash
cd /data/shared/scuppy/hyi/Ablation/Compare/pose_guided
unzip pose_guided_v5style_seed42_gaze045.zip
cd pose_guided_v5style_seed42_gaze045
bash run_seed42.sh
```

Direct command:

```bash
python train_pose_guided_multitask.py   --config configs/pose_guided_seed42_gaze045_light.yaml   2>&1 | tee train_pose_guided_seed42_gaze045_light.log
```

Key settings:

```yaml
seed: 42
train:
  epochs: 20
  patience: 4
  batch_size: 64
  num_workers: 6
  lr: 0.00075
  save_every_epoch: false
loss:
  alpha_action: 0.45
  alpha_gaze: 0.45
  alpha_hands: 0.05
  alpha_talk: 0.05
best_score_weights:
  action: 0.45
  gaze: 0.45
  hands: 0.05
  talk: 0.05
paths:
  fixed_items_json: /data/shared/scuppy/hyi/fixed_splits/dms_clean_masked_fixed_items_v1.json
```

Output directory:

```bash
/data/shared/scuppy/hyi/Ablation/Compare/pose_guided/artifacts_gaze045_light/pose_guided_seed42_gaze045_light
```

Prediction export after training:

```bash
bash run_export_predictions.sh
```
