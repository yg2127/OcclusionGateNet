"""Pose (COCO17) 전처리: 정규화, bone/velocity, conf gating."""
from __future__ import annotations

import numpy as np


COCO_EDGES = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

COCO_PARENTS = {
    0: 0,
    1: 0, 2: 0,
    3: 1, 4: 2,
    5: 0, 6: 0,
    7: 5, 8: 6,
    9: 7, 10: 8,
    11: 5, 12: 6,
    13: 11, 14: 12,
    15: 13, 16: 14,
}


def build_coco_adjacency(num_joints: int = 17, self_link: bool = True) -> np.ndarray:
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    if self_link:
        for i in range(num_joints):
            A[i, i] = 1.0
    for i, j in COCO_EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-6))
    return (D_inv_sqrt @ A @ D_inv_sqrt).astype(np.float32)


def center_scale_normalize_xy(xy: np.ndarray, conf: np.ndarray | None = None) -> np.ndarray:
    """xy: (T, V, 2). conf: (T, V) or None.
    각 프레임마다 유효 joint 평균으로 중심 이동 + bbox 대각선 크기로 스케일 정규화.
    """
    out = xy.copy().astype(np.float32)
    T = out.shape[0]

    for t in range(T):
        pts = out[t]
        if conf is not None:
            valid = conf[t] > 0
        else:
            valid = ~np.isnan(pts).any(axis=1)
        valid_pts = pts[valid]
        if len(valid_pts) < 2:
            out[t] = 0.0
            continue
        center = valid_pts.mean(axis=0)
        pts = pts - center
        mn = valid_pts.min(axis=0)
        mx = valid_pts.max(axis=0)
        scale = float(np.linalg.norm(mx - mn))
        if scale < 1e-6:
            scale = 1.0
        pts = pts / scale
        pts[~valid] = 0.0
        out[t] = pts
    return out


def compute_bone_feature(xy: np.ndarray, parents: dict = COCO_PARENTS) -> np.ndarray:
    T, V, C = xy.shape
    bone = np.zeros_like(xy, dtype=np.float32)
    for v in range(V):
        p = parents[v]
        if p == v:
            continue
        bone[:, v, :] = xy[:, v, :] - xy[:, p, :]
    return bone


def compute_velocity_feature(xy: np.ndarray) -> np.ndarray:
    vel = np.zeros_like(xy, dtype=np.float32)
    if xy.shape[0] >= 2:
        vel[1:] = xy[1:] - xy[:-1]
    return vel


def apply_conf_gating(bone: np.ndarray, vel: np.ndarray, conf: np.ndarray, thres: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """conf < thres 인 joint 의 bone/velocity 를 0 으로 마스킹. shapes: bone/vel=(T,V,2), conf=(T,V)."""
    invalid = conf < thres
    bone = bone.copy()
    vel = vel.copy()
    bone[invalid] = 0.0
    vel[invalid] = 0.0
    return bone, vel


def preprocess_pose_clip(
    keypoints: np.ndarray,
    conf: np.ndarray | None,
    use_bone: bool = True,
    use_velocity: bool = True,
    use_conf_channel: bool = True,
    joint_conf_thres: float = 0.2,
) -> np.ndarray:
    """clip 범위로 이미 슬라이스된 (T, 17, 2) / (T, 17) 에서 (C, T, V) 멀티채널 텐서 생성."""
    if conf is not None:
        conf = np.nan_to_num(conf.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    keypoints = np.nan_to_num(keypoints.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    xy = center_scale_normalize_xy(keypoints, conf=conf)
    xy = np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)

    feats = [xy]  # (T, V, 2)
    if use_bone:
        bone = compute_bone_feature(xy)
        if conf is not None:
            bone, _ = apply_conf_gating(bone, bone, conf, thres=joint_conf_thres)
        feats.append(bone)
    if use_velocity:
        vel = compute_velocity_feature(xy)
        if conf is not None:
            _, vel = apply_conf_gating(vel, vel, conf, thres=joint_conf_thres)
        feats.append(vel)
    if use_conf_channel:
        if conf is None:
            conf = np.ones((xy.shape[0], xy.shape[1]), dtype=np.float32)
        feats.append(conf[..., None])

    x = np.concatenate(feats, axis=-1).astype(np.float32)   # (T, V, C)
    x = np.transpose(x, (2, 0, 1)).astype(np.float32)       # (C, T, V)
    return x


def window_valid_stats(conf_win: np.ndarray, min_joint_ratio: float = 0.15) -> tuple[int, float]:
    if conf_win is None or conf_win.size == 0:
        return 0, 0.0
    joint_ratio_per_frame = np.mean(conf_win > 0, axis=1)
    valid_mask = joint_ratio_per_frame >= min_joint_ratio
    n_valid = int(valid_mask.sum())
    ratio = float(n_valid / max(len(conf_win), 1))
    return n_valid, ratio
