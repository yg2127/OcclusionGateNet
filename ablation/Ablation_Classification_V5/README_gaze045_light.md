# V5 gaze/action-balanced lightweight retraining bundle

Target root:

```bash
/data/shared/scuppy/hyi/Ablation/Ablation_Classification_V5
```

This bundle adds scripts for retraining all V5 ablation models with:

```text
action = 0.45
gaze   = 0.45
hands  = 0.05
talk   = 0.05
```

The same weights are applied to:

1. `loss.alpha_*` — actual training loss weights
2. `best_score_weights` — validation checkpoint selection score

The original configs are not overwritten. New configs are written to:

```bash
configs_gaze045_light/
```

Training outputs are written to:

```bash
artifacts_gaze045_light/
results_gaze045_light/
logs/
```

## Install/copy

From any location:

```bash
rsync -av v5_gaze045_light_code_bundle/ /data/shared/scuppy/hyi/Ablation/Ablation_Classification_V5/
```

Or manually copy the `tools/` and `train/` files into the project root.

## 1. Generate configs only

```bash
cd /data/shared/scuppy/hyi/Ablation/Ablation_Classification_V5
PYTHON=/data/shared/envs/scuppy/bin/python \
  python tools/make_gaze045_light_configs.py \
  --root /data/shared/scuppy/hyi/Ablation/Ablation_Classification_V5
```

Default lightweight settings:

```text
epochs = 20
patience = 4
save_every_epoch = false
num_workers = 6
batch_size = keep original config value
lr = keep original config value
```

If you want even lighter training:

```bash
python tools/make_gaze045_light_configs.py \
  --root /data/shared/scuppy/hyi/Ablation/Ablation_Classification_V5 \
  --epochs 15 \
  --patience 3 \
  --num_workers 4
```

## 2. Train all gaze045-light models

```bash
cd /data/shared/scuppy/hyi/Ablation/Ablation_Classification_V5
bash train/run_gaze045_light_all.sh
```

Included configs:

```text
v5_no_occ_original_mediapipe_seed42
v5_task_gated_late
v5_task_region_gated_late
v5_task_region_scalar_gated_late
v5_explicit_region_mask_gate
v5_explicit_region_scalar_mask_gate
v5_occ_token_region_transformer
v5_occ_attention_bias
```

## 3. Export prediction CSVs after training

```bash
bash train/run_gaze045_light_eval_predictions.sh
```

This writes files like:

```text
artifacts_gaze045_light/<model>_gaze045_light/test_clean_action_clip_predictions.csv
artifacts_gaze045_light/<model>_gaze045_light/test_masked_action_clip_predictions.csv
...
```

## 4. Train then evaluate in one command

```bash
bash train/run_gaze045_light_train_then_eval_all.sh
```

## 5. Bootstrap the new prediction CSVs

```bash
N_BOOT=5000 bash train/run_gaze045_light_bootstrap.sh
```

Outputs:

```bash
bootstrap_results_gaze045_light/bootstrap_raw_by_seed.csv
bootstrap_results_gaze045_light/bootstrap_summary_by_model_task.csv
bootstrap_results_gaze045_light/quick_compare.txt
```

## Notes

- This is a new training experiment, not merely a re-scoring experiment.
- If results are based on one seed only, describe it as an exploratory gaze-aware weighting experiment.
- Main comparison should separate:
  - original action-heavy score: 0.70 / 0.15 / 0.10 / 0.05
  - gaze/action-balanced score: 0.45 / 0.45 / 0.05 / 0.05
