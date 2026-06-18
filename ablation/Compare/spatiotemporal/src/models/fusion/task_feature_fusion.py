from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

TASK_NAMES = ("action", "gaze", "hands", "talk")


class TaskFeatureFusion(nn.Module, ABC):
    """Common interface for task-level fusion modules."""

    out_dim: int

    @abstractmethod
    def forward(
        self,
        *,
        shared: torch.Tensor,
        pose_feat: torch.Tensor,
        face_feat: torch.Tensor,
        x_occ: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class IdentityTaskFusion(TaskFeatureFusion):
    """Debug fallback: all heads receive the same shared fused representation."""

    def __init__(self, fused_channels: int):
        super().__init__()
        self.out_dim = int(fused_channels)

    def forward(self, *, shared, pose_feat, face_feat, x_occ=None):
        del pose_feat, face_feat, x_occ
        return {name: shared for name in TASK_NAMES}
