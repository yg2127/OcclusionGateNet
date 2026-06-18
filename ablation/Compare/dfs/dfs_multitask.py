"""DFS baseline adapted to DMD multi-task labels.

Paper: "Multi-modality action recognition based on dual feature shift in
vehicle cabin monitoring" (arXiv:2401.14838).

The paper uses RGB/IR/depth clips and a ResNet backbone with dual feature
shift: channel exchange across modalities and temporal neighbour propagation.
For the DMD project, we treat body pose, face/facemesh, and optional head pose
as the available modalities and keep the project labels:

  action: 11 classes
  gaze:   9 classes
  hands:  4 classes
  talk:   2 classes
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


def modality_shift(features: list[torch.Tensor], fold_div: int = 8) -> list[torch.Tensor]:
    """Exchange the last channel fold with the next modality.

    Input tensors are shaped (N, T, C). This mirrors the paper's zero-cost
    modality shift idea while supporting any number of DMD modalities.
    """
    if len(features) <= 1:
        return features
    channels = features[0].shape[-1]
    fold = max(1, channels // fold_div)
    shifted = []
    for idx, feat in enumerate(features):
        donor = features[(idx + 1) % len(features)]
        shifted.append(torch.cat([feat[..., :-fold], donor[..., -fold:]], dim=-1))
    return shifted


def temporal_shift(x: torch.Tensor, fold_div: int = 8) -> torch.Tensor:
    """Propagate features from previous and next frames.

    Input is (N, T, C). First fold receives t-1, second fold receives t+1,
    and remaining channels stay at the current timestamp.
    """
    n, t, c = x.shape
    fold = max(1, c // fold_div)
    out = x.clone()
    if t <= 1:
        return out
    out[:, 1:, :fold] = x[:, :-1, :fold]
    out[:, 0, :fold] = 0
    out[:, :-1, fold : 2 * fold] = x[:, 1:, fold : 2 * fold]
    out[:, -1, fold : 2 * fold] = 0
    return out


class StageBlock(nn.Module):
    """Lightweight temporal feature extractor used as a DFS stage."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.temporal = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = self.temporal(y.transpose(1, 2)).transpose(1, 2)
        x = x + self.dropout(self.act(y))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class ModalityInputProjector(nn.Module):
    """Flatten per-frame modality joints/regions and project to shared dim."""

    def __init__(self, in_channels: int, num_regions: int, embed_dim: int, dropout: float):
        super().__init__()
        self.in_dim = in_channels * num_regions
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, t, v = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(n, t, c * v)
        return self.net(x)


class DFSModalityEncoder(nn.Module):
    """Five-stage DFS encoder with middle shared stages."""

    def __init__(self, num_modalities: int, embed_dim: int, dropout: float, shift_fold_div: int = 8):
        super().__init__()
        self.shift_fold_div = shift_fold_div
        self.stage1 = nn.ModuleList([StageBlock(embed_dim, dropout) for _ in range(num_modalities)])
        self.stage2_shared = StageBlock(embed_dim, dropout)
        self.stage3_shared = StageBlock(embed_dim, dropout)
        self.stage4 = nn.ModuleList([StageBlock(embed_dim, dropout) for _ in range(num_modalities)])
        self.stage5 = nn.ModuleList([StageBlock(embed_dim, dropout) for _ in range(num_modalities)])

    def _dual_shift(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        xs = modality_shift(xs, self.shift_fold_div)
        return [temporal_shift(x, self.shift_fold_div) for x in xs]

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        xs = [block(x) for block, x in zip(self.stage1, xs)]
        xs = self._dual_shift(xs)
        xs = [self.stage2_shared(x) for x in xs]
        xs = self._dual_shift(xs)
        xs = [self.stage3_shared(x) for x in xs]
        xs = self._dual_shift(xs)
        xs = [block(x) for block, x in zip(self.stage4, xs)]
        xs = self._dual_shift(xs)
        return [block(x) for block, x in zip(self.stage5, xs)]


class DFSDMDMultitaskClassifier(nn.Module):
    """DFS-style multi-modality baseline for DMD action/hands/talk/gaze."""

    def __init__(
        self,
        pose_in_channels: int,
        face_in_channels: int,
        num_pose_joints: int = 17,
        num_face_regions: int = 478,
        embed_dim: int | None = None,
        stream_dim: int = 256,
        proj_dim: int | None = None,
        fusion_dim: int = 512,
        dropout: float = 0.3,
        shift_fold_div: int = 8,
        use_head_pose: bool = False,
        head_pose_in_channels: int = 2,
        num_head_pose_axes: int = 3,
        num_action: int = 11,
        num_gaze: int = 9,
        num_hands: int = 4,
        num_talk: int = 2,
        **_: object,
    ):
        super().__init__()
        embed_dim = int(embed_dim or stream_dim or proj_dim or 256)
        self.use_head_pose = use_head_pose
        self.pose_proj = ModalityInputProjector(pose_in_channels, num_pose_joints, embed_dim, dropout)
        self.face_proj = ModalityInputProjector(face_in_channels, num_face_regions, embed_dim, dropout)
        num_modalities = 2
        if use_head_pose:
            self.head_pose_proj = ModalityInputProjector(
                head_pose_in_channels, num_head_pose_axes, embed_dim, dropout
            )
            num_modalities += 1
        self.encoder = DFSModalityEncoder(
            num_modalities=num_modalities,
            embed_dim=embed_dim,
            dropout=dropout,
            shift_fold_div=shift_fold_div,
        )
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.action_head = nn.Linear(fusion_dim, num_action)
        self.gaze_head = nn.Linear(fusion_dim, num_gaze)
        self.hands_head = nn.Linear(fusion_dim, num_hands)
        self.talk_head = nn.Linear(fusion_dim, num_talk)

    def forward(
        self,
        x_body: torch.Tensor,
        x_face: torch.Tensor,
        x_head_pose: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        modalities = [self.pose_proj(x_body), self.face_proj(x_face)]
        if self.use_head_pose:
            if x_head_pose is None:
                n, _, t, _ = x_body.shape
                x_head_pose = x_body.new_zeros((n, 2, t, 3))
            modalities.append(self.head_pose_proj(x_head_pose))
        encoded = self.encoder(modalities)
        pooled = torch.stack([x.mean(dim=1) for x in encoded], dim=0).mean(dim=0)
        fused = self.fusion(pooled)
        return {
            "action": self.action_head(fused),
            "gaze": self.gaze_head(fused),
            "hands": self.hands_head(fused),
            "talk": self.talk_head(fused),
        }
