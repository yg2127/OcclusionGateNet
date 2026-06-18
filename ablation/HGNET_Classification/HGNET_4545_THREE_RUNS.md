# HGNET Classification - 4545 Three Fusion Runs

Server target path:

```bash
/data/shared/scuppy/hyi/Ablation/HGNET_Classification
```

This package is configured to run only these three variants:

1. `task_gated_late`
2. `task_region_scalar_gated_late`
3. `explicit_region_scalar_mask_gate`

Common settings:

```yaml
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
train:
  patience: 4
seed: 42
```

Configs are in:

```bash
classifier/configs/hgnet_4545/
```

Run all three:

```bash
cd /data/shared/scuppy/hyi/Ablation/HGNET_Classification/classifier
bash scripts/run_hgnet_4545_three.sh
```

Or run one:

```bash
cd /data/shared/scuppy/hyi/Ablation/HGNET_Classification/classifier
PYTHONPATH=$(pwd):$(pwd)/configs python src/training/train.py   --config configs/hgnet_4545/model4_occgateRAW_taskRegionScalarGatedLate_seed42_loss045.yaml
```

Export predictions after training:

```bash
cd /data/shared/scuppy/hyi/Ablation/HGNET_Classification/classifier
bash scripts/export_hgnet_4545_three_predictions.sh
```
