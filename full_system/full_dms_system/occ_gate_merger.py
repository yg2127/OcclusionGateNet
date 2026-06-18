from __future__ import annotations

from typing import Dict, Iterable, Optional
import numpy as np

# MediaPipe 478 region indices. This mirrors the classifier/constants/face_regions.py groups.
LEFT_EYE = sorted(set([246,161,160,159,158,157,173,33,7,163,144,145,153,154,155,133,468,469,470,471,472]))
RIGHT_EYE = sorted(set([263,249,390,373,374,380,381,382,362,466,388,387,386,385,384,398,473,474,475,476,477]))
NOSE = sorted(set([168,6,197,195,5,4,1,19,94,2,141,370,98,97,326,327,45,51,115,220,219,218,237,275,281,344,440,439,438,457,129,358,102,331]))
MOUTH = sorted(set([61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185,78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191]))

REGION_PTS = {
    "left_eye": LEFT_EYE,
    "right_eye": RIGHT_EYE,
    "nose": NOSE,
    "mouth": MOUTH,
}
GATE = sorted(set().union(*[set(v) for v in REGION_PTS.values()]))
FIT = np.array([i for i in range(478) if i not in set(GATE)], dtype=np.int64)


def umeyama2d(src: np.ndarray, dst: np.ndarray):
    ok = np.isfinite(src).all(1) & np.isfinite(dst).all(1)
    src, dst = src[ok], dst[ok]
    if len(src) < 8:
        return None
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    cov = (d0.T @ s0) / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2, dtype=np.float32)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    var_s = (s0 ** 2).sum() / len(src)
    scale = np.trace(np.diag(D) @ S) / max(var_s, 1e-9)
    t = mu_d - scale * (R @ mu_s)
    return float(scale), R.astype(np.float32), t.astype(np.float32)


class OccGatedFaceMeshMerger:
    """Merge MediaPipe raw landmarks and HGNet restored landmarks.

    - No occlusion: return MediaPipe landmarks unchanged.
    - Occlusion: transform HGNet landmarks into MediaPipe coordinate space using stable points,
      then replace only occluded regions.
    """

    def __init__(self, visible_threshold: float = 0.5, use_nose_gate: bool = True):
        self.visible_threshold = float(visible_threshold)
        self.use_nose_gate = bool(use_nose_gate)

    def visible_probs_to_labels(self, occ_probs: np.ndarray, crop_valid: bool) -> Dict[str, int]:
        if occ_probs is None or len(occ_probs) < 4 or not crop_valid:
            return {"left_eye": 0, "right_eye": 0, "nose": 0, "mouth": 0}
        p = np.asarray(occ_probs, dtype=np.float32)
        return {
            "left_eye": int(p[0] < self.visible_threshold),
            "right_eye": int(p[1] < self.visible_threshold),
            "nose": int(p[2] < self.visible_threshold) if self.use_nose_gate else 0,
            "mouth": int(p[3] < self.visible_threshold),
        }

    def merge(
        self,
        mediapipe_lm: np.ndarray,
        hgnet_lm_frame: Optional[np.ndarray],
        occ_probs: np.ndarray,
        crop_valid: bool,
        mp_detected: bool,
        hg_detected: bool,
    ) -> tuple[np.ndarray, Dict[str, int]]:
        labels = self.visible_probs_to_labels(occ_probs, crop_valid)
        if not mp_detected:
            # If face branch failed, return zero mp landmarks. DMS fallback will still run.
            return np.zeros((478, 3), dtype=np.float32), labels

        out = np.asarray(mediapipe_lm, dtype=np.float32).copy()
        if not any(v == 1 for v in labels.values()):
            return out, labels
        if hgnet_lm_frame is None or not hg_detected:
            return out, labels

        hg = np.asarray(hgnet_lm_frame, dtype=np.float32)
        if hg.shape[0] < 478 or out.shape[0] < 478:
            return out, labels

        # Fit HGNet full-frame estimate to MediaPipe full-frame coordinates.
        res = umeyama2d(hg[FIT, :2], out[FIT, :2])
        if res is not None:
            scale, R, t = res
            hg_xy = (scale * (R @ hg[:, :2].T)).T + t
        else:
            hg_xy = hg[:, :2]

        for rname, pts in REGION_PTS.items():
            if labels.get(rname, 0) == 1:
                out[pts, :2] = hg_xy[pts]
                if hg.shape[1] >= 3:
                    out[pts, 2] = hg[pts, 2]
        return out.astype(np.float32), labels
