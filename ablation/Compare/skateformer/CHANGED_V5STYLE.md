# Changes from uploaded Compare_SkateFormer.zip

- Set target root to `/data/shared/scuppy/hyi/Ablation/Compare/skateformer`.
- Added `configs/skateformer_seed42_gaze045_light.yaml`.
- Fixed seed to 42 only.
- Set `train.patience=4`, `epochs=20`, `lr=0.00075`, `num_workers=6`, `save_every_epoch=false`.
- Set loss weights to action/gaze/hands/talk = 0.45/0.45/0.05/0.05.
- Set best score weights to the same 0.45/0.45/0.05/0.05.
- Fixed manifest path to `/data/shared/scuppy/hyi/fixed_splits/dms_clean_masked_fixed_items_v1.json`.
- Kept clean+masked train/val/test variants.
- Added root-level `run_seed42.sh` and `run_export_predictions.sh`.
- Removed previous `artifacts`, `results`, `logs`, `analysis`, `__pycache__`, and `*.pyc` from the packaged zip.
