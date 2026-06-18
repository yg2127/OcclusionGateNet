"""Drive&Act-inspired multi-stream classifier adapted to DMD labels.

Baseline models are kept under ``src.baselines`` so paper-comparison code does
not get mixed with the main project models.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _TemporalPoseStream(nn.Module):
    """Body dynamics stream: (N, C, T, V) -> (N, D)."""

    def __init__(self, pose_channels: int, num_joints: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.input_proj = _MLP(pose_channels * num_joints, hidden_dim, hidden_dim, dropout)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=out_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x_body: torch.Tensor) -> torch.Tensor:
        n, c, t, v = x_body.shape
        x = x_body.permute(0, 2, 1, 3).contiguous().view(n, t, c * v)
        x = self.input_proj(x)
        y, _ = self.lstm(x)
        return self.norm(y.mean(dim=1))


class _SpatialPoseStream(nn.Module):
    """Pose configuration stream over joints: (N, C, T, V) -> (N, D)."""

    def __init__(self, pose_channels: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.joint_proj = _MLP(pose_channels, hidden_dim, hidden_dim, dropout)
        self.joint_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=out_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.temporal_gate = nn.Sequential(nn.Linear(out_dim, out_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x_body: torch.Tensor) -> torch.Tensor:
        n, c, t, v = x_body.shape
        x = x_body.permute(0, 2, 3, 1).contiguous().view(n * t, v, c)
        x = self.joint_proj(x)
        y, _ = self.joint_lstm(x)
        y = y.mean(dim=1).view(n, t, -1)
        pooled = y.mean(dim=1)
        return self.norm(pooled * self.temporal_gate(pooled))


class _ContextStream(nn.Module):
    """DMD replacement for Drive&Act's car-interior stream."""

    def __init__(
        self,
        face_channels: int,
        num_face_regions: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float,
        use_head_pose: bool,
        head_pose_channels: int = 2,
        num_head_pose_axes: int = 3,
        **_: object,
    ):
        super().__init__()
        self.use_head_pose = use_head_pose
        context_dim = face_channels * num_face_regions
        if use_head_pose:
            context_dim += head_pose_channels * num_head_pose_axes
        self.input_proj = _MLP(context_dim, hidden_dim, hidden_dim, dropout)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=out_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x_face: torch.Tensor, x_head_pose: Optional[torch.Tensor] = None) -> torch.Tensor:
        n, c, t, k = x_face.shape
        face = x_face.permute(0, 2, 1, 3).contiguous().view(n, t, c * k)
        if self.use_head_pose:
            if x_head_pose is None:
                hp = face.new_zeros((n, t, 6))
            else:
                _, ch, th, axes = x_head_pose.shape
                hp = x_head_pose.permute(0, 2, 1, 3).contiguous().view(n, th, ch * axes)
                if th != t:
                    hp = torch.nn.functional.interpolate(
                        hp.permute(0, 2, 1), size=t, mode="linear", align_corners=False
                    ).permute(0, 2, 1)
            x = torch.cat([face, hp], dim=-1)
        else:
            x = face
        x = self.input_proj(x)
        y, _ = self.lstm(x)
        return self.norm(y.mean(dim=1))


class DriveActDMDMultitaskClassifier(nn.Module):
    """Drive&Act-style three-stream late-fusion model for DMD multi-task labels."""

    def __init__(
        self,
        pose_in_channels: int,
        face_in_channels: int,
        num_pose_joints: int = 17,
        num_face_regions: int = 478,
        hidden_dim: int = 256,
        stream_dim: int = 256,
        fusion_dim: int = 512,
        dropout: float = 0.3,
        num_action: int = 11,
        num_gaze: int = 9,
        num_hands: int = 4,
        num_talk: int = 2,
        use_head_pose: bool = False,
        head_pose_in_channels: int = 2,
        num_head_pose_axes: int = 3,
        **_: object,
    ):
        super().__init__()
        self.use_head_pose = use_head_pose
        self.temporal_stream = _TemporalPoseStream(
            pose_in_channels, num_pose_joints, hidden_dim, stream_dim, dropout
        )
        self.spatial_stream = _SpatialPoseStream(pose_in_channels, hidden_dim, stream_dim, dropout)
        self.context_stream = _ContextStream(
            face_in_channels,
            num_face_regions,
            hidden_dim,
            stream_dim,
            dropout,
            use_head_pose=use_head_pose,
            head_pose_channels=head_pose_in_channels,
            num_head_pose_axes=num_head_pose_axes,
        )
        self.stream_logits = nn.Parameter(torch.zeros(3))
        self.fusion = nn.Sequential(
            nn.Linear(stream_dim * 3, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )
        self.head_dropout = nn.Dropout(dropout)
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
        temporal = self.temporal_stream(x_body)
        spatial = self.spatial_stream(x_body)
        context = self.context_stream(x_face, x_head_pose if self.use_head_pose else None)
        weights = torch.softmax(self.stream_logits, dim=0)
        fused = torch.cat(
            [temporal * weights[0], spatial * weights[1], context * weights[2]],
            dim=1,
        )
        z = self.head_dropout(self.fusion(fused))
        return {
            "action": self.action_head(z),
            "gaze": self.gaze_head(z),
            "hands": self.hands_head(z),
            "talk": self.talk_head(z),
        }
