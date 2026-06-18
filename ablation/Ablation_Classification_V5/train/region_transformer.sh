#!/usr/bin/env bash
# V1.1 multi-task 학습 실행. tmux 에서 실행 권장.
#
#   tmux new -s region_landmark_concat
#   bash train/region_transformer.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${HERE}"

PY="${PY:-/data/shared/envs/scuppy/bin/python}"
CONFIG="${CONFIG:-configs/region_transformer.yaml}"

if [ ! -f constants/frame_shifts.json ]; then
    echo "[prebuild] constants/frame_shifts.json not found, generating..."
    "${PY}" scripts/build_frame_shifts.py
fi

echo "[train] config=${CONFIG} python=${PY}"
exec "${PY}" -m src.training.train --config "${CONFIG}"
