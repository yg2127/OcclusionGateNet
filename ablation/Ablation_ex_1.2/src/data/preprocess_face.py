"""FaceMesh (478, 3) 전처리: bbox 기반 face-local 정규화 + K=10 region pooling + detected channel."""
from __future__ import annotations

import numpy as np

from constants.face_regions import region_mean_matrix


_MEAN_MATRIX_CACHE: np.ndarray | None = None


def get_region_mean_matrix() -> np.ndarray:
    global _MEAN_MATRIX_CACHE
    if _MEAN_MATRIX_CACHE is None:
        _MEAN_MATRIX_CACHE = region_mean_matrix()    # (K=10, 478)
    return _MEAN_MATRIX_CACHE


def _self_bbox(landmarks_xyz: np.ndarray) -> np.ndarray:
    """landmarks: (478, 3). self-bbox from xy min/max → [x1, y1, x2, y2]."""
    xy = landmarks_xyz[:, :2]
    mn = xy.min(axis=0)
    mx = xy.max(axis=0)
    return np.array([mn[0], mn[1], mx[0], mx[1]], dtype=np.float32)


def normalize_face_frame(
    landmarks: np.ndarray,
    bbox: np.ndarray | None,
    bbox_valid: bool,
) -> np.ndarray:
    """landmarks: (478, 3), bbox: (4,) or None.
    bbox_valid=False 이면 self-bbox fallback.
    반환: (478, 3) float32. bbox 중심·크기로 정규화.
    """
    if not bbox_valid or bbox is None:
        bb = _self_bbox(landmarks)
    else:
        bb = np.asarray(bbox, dtype=np.float32)

    x1, y1, x2, y2 = bb
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    s = float(max(x2 - x1, y2 - y1))
    if s < 1e-6:
        s = 1.0

    out = np.zeros_like(landmarks, dtype=np.float32)
    out[:, 0] = (landmarks[:, 0] - cx) / s
    out[:, 1] = (landmarks[:, 1] - cy) / s
    out[:, 2] = landmarks[:, 2] / s
    return out


def preprocess_face_clip(
    landmarks: np.ndarray,
    detected: np.ndarray,
    bbox: np.ndarray,
    bbox_det_score: np.ndarray,
    bbox_detected: np.ndarray,
    use_z: bool = True,
    use_detected_channel: bool = True,
    bbox_det_thres: float = 0.25,
    use_region_pool: bool = True,
) -> np.ndarray:
    """clip 범위로 이미 슬라이스된 face 텐서들에서 (C, T, V) 생성.

    - use_region_pool=True  -> V=10 (해부학 mean pool)
    - use_region_pool=False -> V=478 (raw landmark 전체)

    - landmarks:      (T, 478, 3)  float32  (0~W, 0~H 픽셀 좌표)
    - detected:       (T,)         bool     (FaceMesh detection per frame)
    - bbox:           (T, 4)       float32  (from yolo_face)
    - bbox_det_score: (T,)         float32
    - bbox_detected:  (T,)         bool

    C = xyz(3 or 2) + (detected? 1 : 0)
    """
    landmarks = np.nan_to_num(landmarks.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    detected = detected.astype(bool)
    bbox = np.nan_to_num(bbox.astype(np.float32), nan=0.0)
    bbox_det_score = np.nan_to_num(bbox_det_score.astype(np.float32), nan=0.0)
    bbox_detected = bbox_detected.astype(bool)

    T = landmarks.shape[0]
    norm = np.zeros_like(landmarks, dtype=np.float32)
    for t in range(T):
        bv = bool(bbox_detected[t]) and float(bbox_det_score[t]) >= bbox_det_thres
        norm[t] = normalize_face_frame(landmarks[t], bbox[t] if bv else None, bbox_valid=bv)

    # zero-mask 프레임: facemesh.detected=False 인 프레임의 좌표 전부 0
    invalid_frames = ~detected
    if invalid_frames.any():
        norm[invalid_frames] = 0.0

    if use_region_pool:
        # (T, 478, 3) @ (K=10, 478) → (T, 10, 3)
        M = get_region_mean_matrix()
        pooled = np.einsum("kv,tvc->tkc", M, norm)
    else:
        # 풀링 없이 raw 478 유지 → (T, 478, 3)
        pooled = norm

    if not use_z:
        pooled = pooled[..., :2]

    if use_detected_channel:
        det_ch = detected.astype(np.float32)[:, None, None]       # (T, 1, 1)
        det_ch = np.broadcast_to(det_ch, (T, pooled.shape[1], 1)).copy()   # (T, K, 1)
        pooled = np.concatenate([pooled, det_ch], axis=-1)         # (T, K, C)

    x = np.transpose(pooled, (2, 0, 1)).astype(np.float32)         # (C, T, K)
    return x


def window_face_detected_ratio(detected_win: np.ndarray) -> float:
    if detected_win is None or detected_win.size == 0:
        return 0.0
    return float(np.mean(detected_win.astype(np.float32)))
