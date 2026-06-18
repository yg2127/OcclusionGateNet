"""Fusion-I: pose branch와 face branch의 joint(V) 축 concatenation."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def build_fused_adjacency(pose_A: np.ndarray, num_face_regions: int = 10) -> np.ndarray:
    """pose joints 17 + face regions 10 → (27, 27). pose는 기존 인접, face는 self-link만.
    두 모달리티 간 cross-edge 없음 (v0.2 실험 대상).
    """
    V_pose = pose_A.shape[0]
    V_face = num_face_regions
    V = V_pose + V_face
    A = np.zeros((V, V), dtype=np.float32)
    A[:V_pose, :V_pose] = pose_A
    for i in range(V_face):
        A[V_pose + i, V_pose + i] = 1.0
    # 정규화 (pose 부분은 이미 정규화됐으나 face 블록 포함을 위해 재정규화)
    D = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-6))
    return (D_inv_sqrt @ A @ D_inv_sqrt).astype(np.float32)


class ConcatJointFusion(nn.Module):
    """pose_feat, face_feat 를 joint 축으로 이어 붙임.

    입력:
      pose_feat: (N, C, T, V_pose)
      face_feat: (N, C, T, V_face)
    출력:
      (N, C, T, V_pose + V_face)
    """
    def forward(self, pose_feat: torch.Tensor, face_feat: torch.Tensor) -> torch.Tensor:
        assert pose_feat.dim() == 4 and face_feat.dim() == 4, "rank must be 4"
        assert pose_feat.shape[0] == face_feat.shape[0], f"batch mismatch: {pose_feat.shape} vs {face_feat.shape}"
        assert pose_feat.shape[1] == face_feat.shape[1], f"channel mismatch: {pose_feat.shape} vs {face_feat.shape}"
        assert pose_feat.shape[2] == face_feat.shape[2], f"temporal mismatch: {pose_feat.shape} vs {face_feat.shape}"
        return torch.cat([pose_feat, face_feat], dim=3)
