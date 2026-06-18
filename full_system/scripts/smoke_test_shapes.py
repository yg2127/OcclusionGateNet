from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from full_dms_system.temporal_buffer import TemporalDMSBuffer


def main():
    buf = TemporalDMSBuffer(window_size=48)
    for _ in range(48):
        buf.append(
            body_keypoints=np.zeros((17, 2), dtype=np.float32),
            body_conf=np.zeros((17,), dtype=np.float32),
            face_lm=np.zeros((478, 3), dtype=np.float32),
            face_detected=False,
            face_bbox=np.zeros((4,), dtype=np.float32),
            face_det_score=0.0,
            face_bbox_detected=False,
            occ_feature=np.array([0.5, 0.5, 0.5, 0.5, 0.0], dtype=np.float32),
        )
    a = buf.as_arrays()
    for k, v in a.items():
        print(k, v.shape, v.dtype)


if __name__ == "__main__":
    main()
