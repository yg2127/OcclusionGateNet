# Ablation_ex_1.2 requested full-FaceMesh run

This package keeps the requested V1-style full FaceMesh condition:

- `face.encoder: full`
- `face.num_regions: 478`
- `face.use_det_score_channel: true`
- `fusion.kind: concat`

The runner order is patched:

1. build clip split
2. build model and move model to CUDA first
3. preload train/val only
4. train
5. release train/val loaders/items
6. preload one test split at a time and evaluate

Experiments:

```bash
bash train/run_requested_two.sh
```

Single config:

```bash
bash train/run_single.sh configs/clean_only_train_clean_test.yaml
bash train/run_single.sh configs/clean_mask_train_clean_mask_test.yaml
```
