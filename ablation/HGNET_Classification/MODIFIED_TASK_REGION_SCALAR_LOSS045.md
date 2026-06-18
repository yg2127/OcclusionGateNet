# model4_dms task_region_scalar_gated_late loss045 package

Applied changes:

- `model.fusion.kind`: `task_region_scalar_gated_late`
- `loss.alpha_action`: `0.45`
- `loss.alpha_gaze`: `0.45`
- `loss.alpha_hands`: `0.05`
- `loss.alpha_talk`: `0.05`
- `best_score_weights.action`: `0.45`
- `best_score_weights.gaze`: `0.45`
- `best_score_weights.hands`: `0.05`
- `best_score_weights.talk`: `0.05`
- `train.patience`: `4`

Primary run config:

```bash
cd classifier
PYTHONPATH=$(pwd):$(pwd)/configs python src/training/train.py \
  --config configs/model4_occgateRAW_taskRegionScalarGatedLate_seed42_loss045.yaml
```

The original task-gated YAML was backed up as:

```text
classifier/configs/model4_occgateRAW_taskGated_occCNN_seed42.original_task_gated.yaml
```

Note: absolute data paths still assume the shared server layout. If extracting outside `/data/shared/scuppy/hyi/Ablation/Compare/model4_dms`, adjust `paths.frame_shifts`, `paths.results_root`, and `paths.save_root` in the YAML.
