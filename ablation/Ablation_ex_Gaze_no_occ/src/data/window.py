"""Mosaic 기준 sliding window → body/face 양쪽 인덱스 생성 + 품질 필터."""
from __future__ import annotations

import numpy as np


def center_crop_indices(length: int, target_len: int) -> np.ndarray:
    if length <= 0:
        return np.zeros((target_len,), dtype=np.int64)
    if length >= target_len:
        start = (length - target_len) // 2
        return np.arange(start, start + target_len, dtype=np.int64)
    pad_total = target_len - length
    left_pad = pad_total // 2
    right_pad = target_len - length - left_pad
    center = np.arange(length, dtype=np.int64)
    left = np.full((left_pad,), 0, dtype=np.int64)
    right = np.full((right_pad,), length - 1, dtype=np.int64)
    return np.concatenate([left, center, right], axis=0)


def sliding_window_mosaic_indices(clip_len: int, window_size: int, stride: int) -> list[np.ndarray]:
    """clip 내부(길이 clip_len)의 local frame 기준 window indices 리스트.
    이 값에 mosaic_start 를 더하면 mosaic 절대 인덱스가 됨.
    """
    if clip_len <= 0:
        return [np.zeros((window_size,), dtype=np.int64)]
    if clip_len <= window_size:
        return [center_crop_indices(clip_len, window_size)]

    out: list[np.ndarray] = []
    s = 0
    while s + window_size <= clip_len:
        out.append(np.arange(s, s + window_size, dtype=np.int64))
        s += stride
    last = np.arange(clip_len - window_size, clip_len, dtype=np.int64)
    if not out or not np.array_equal(out[-1], last):
        out.append(last)
    return out
