#!/usr/bin/env bash
set -euo pipefail

cd /data/shared/scuppy/hyi/Ablation/Compare/tsm_resnet18

python train_dmd_tsm_resnet18_multitask.py \
  --config configs/tsm_resnet18_seed42_gaze045_light.yaml \
  2>&1 | tee train_tsm_resnet18_seed42_gaze045_light.log
