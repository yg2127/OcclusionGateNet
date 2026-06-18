"""yolo_face 5-point 전처리: bbox 기반 정규화 + detected/det_score 채널."""
from __future__ import annotations

import numpy as np


NUM_KPS5 = 5


def _self_bbox_kps5(kps5: np.ndarray) -> np.ndarray:
    """kps5: (5, 2). self-bbox."""
    mn = kps5.min(axis=0)
    mx = kps5.max(axis=0)
    return np.array([mn[0], mn[1], mx[0], mx[1]], dtype=np.float32)


def normalize_kps5_frame(kps5: np.ndarray, bbox: np.ndarray | None, bbox_valid: bool) -> np.ndarray:
    """kps5: (5, 2), bbox: (4,) or None. 반환: (5, 2)."""
    if not bbox_valid or bbox is None:
        bb = _self_bbox_kps5(kps5)
    else:
        bb = np.asarray(bbox, dtype=np.float32)
    x1, y1, x2, y2 = bb
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    s = float(max(x2 - x1, y2 - y1))
    if s < 1e-6:
        s = 1.0
    out = np.zeros_like(kps5, dtype=np.float32)
    out[:, 0] = (kps5[:, 0] - cx) / s
    out[:, 1] = (kps5[:, 1] - cy) / s
    return out


def preprocess_face5pt_clip(
    kps5: np.ndarray,
    bbox: np.ndarray,
    det_score: np.ndarray,
    detected: np.ndarray,
    use_detected_channel: bool = True,
    use_det_score_channel: bool = True,
    bbox_det_thres: float = 0.25,
) -> np.ndarray:
    """clip 범위 슬라이스된 face5pt 텐서 → (C, T, 5).

    - kps5:       (T, 5, 2) float32
    - bbox:       (T, 4)    float32
    - det_score:  (T,)      float32
    - detected:   (T,)      bool

    C = 2 (xy) + (detected ? 1 : 0) + (det_score ? 1 : 0)
    """
    kps5 = np.nan_to_num(kps5.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    bbox = np.nan_to_num(bbox.astype(np.float32), nan=0.0)
    det_score = np.nan_to_num(det_score.astype(np.float32), nan=0.0)
    detected = detected.astype(bool)

    T = kps5.shape[0]
    norm = np.zeros_like(kps5, dtype=np.float32)
    for t in range(T):
        bv = bool(detected[t]) and float(det_score[t]) >= bbox_det_thres
        norm[t] = normalize_kps5_frame(kps5[t], bbox[t] if bv else None, bbox_valid=bv)

    # zero-mask: detected=False 프레임의 xy 0
    invalid = ~detected
    if invalid.any():
        norm[invalid] = 0.0

    # (T, 5, 2) → 채널 확장
    feats = [norm]     # (T, 5, 2)

    if use_detected_channel:
        det_ch = detected.astype(np.float32)[:, None, None]
        det_ch = np.broadcast_to(det_ch, (T, NUM_KPS5, 1)).copy()
        feats.append(det_ch)

    if use_det_score_channel:
        sc_ch = det_score[:, None, None]
        sc_ch = np.broadcast_to(sc_ch, (T, NUM_KPS5, 1)).copy()
        feats.append(sc_ch)

    pooled = np.concatenate(feats, axis=-1)       # (T, 5, C)
    x = np.transpose(pooled, (2, 0, 1)).astype(np.float32)   # (C, T, 5)
    return x


def face5pt_detected_ratio(detected_win: np.ndarray) -> float:
    if detected_win is None or detected_win.size == 0:
        return 0.0
    return float(np.mean(detected_win.astype(np.float32)))
