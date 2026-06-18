"""Explicit region reliability gating.

3-A variant:
    Transformer(face_region_tokens) -> task region gate -> gate * region_reliability
    -> weighted region pooling -> task-specific residual.

This is the most direct implementation of:
    g_gaze_region = gaze_region_gate(face_region_tokens)
    g_gaze_region = g_gaze_region * eye_visibility_mask
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .region_occ_utils import occ_to_region_reliability
from .task_feature_fusion import (
    TASK_NAMES,
    FaceRegionRelationEncoder,
    TaskFeatureFusion,
    TaskRegionGate,
)


class ExplicitRegionMaskGateFusion(TaskFeatureFusion):
    """Task region gate multiplied by explicit region reliability from x_occ."""

    def __init__(
        self,
        *,
        fused_channels: int,
        face_channels: int,
        occ_dim: int = 0,
        gate_hidden_channels: int = 128,
        gate_dropout: float = 0.2,
        gate_feature_scale: float = 0.25,
        init_bias: dict | None = None,
        region_num_heads: int = 4,
        region_num_layers: int = 1,
        region_dropout: float = 0.1,
        region_ff_mult: int = 2,
        region_occ_indices: Sequence[int] | None = None,
        default_visible: float = 1.0,
        mask_strength: float = 1.0,
        min_reliability: float = 0.05,
        gate_condition_occ: bool = True,
    ):
        super().__init__()
        init_bias = init_bias or {}
        self.out_dim = int(fused_channels)
        self.occ_dim = int(occ_dim)
        self.gate_feature_scale = float(gate_feature_scale)
        self.region_occ_indices = None if region_occ_indices is None else [int(x) for x in region_occ_indices]
        self.default_visible = float(default_visible)
        self.mask_strength = float(mask_strength)
        self.min_reliability = float(min_reliability)

        self.relation = FaceRegionRelationEncoder(
            channels=face_channels,
            num_heads=region_num_heads,
            num_layers=region_num_layers,
            dropout=region_dropout,
            ff_mult=region_ff_mult,
        )
        self.gates = nn.ModuleDict(
            {
                name: TaskRegionGate(
                    channels=face_channels,
                    hidden_dim=gate_hidden_channels,
                    dropout=gate_dropout,
                    init_bias=float(init_bias.get(name, 0.0)),
                    cond_dim=self.occ_dim if gate_condition_occ else 0,
                )
                for name in TASK_NAMES
            }
        )
        self.face_proj = nn.Sequential(
            nn.LayerNorm(face_channels),
            nn.Linear(face_channels, fused_channels),
            nn.GELU(),
            nn.Dropout(gate_dropout),
        )

    @staticmethod
    def _face_region_tokens(face_feat: torch.Tensor) -> torch.Tensor:
        if face_feat.ndim != 4:
            raise ValueError(f"expected face feature (N, C, T, R), got {tuple(face_feat.shape)}")
        return face_feat.mean(dim=2).transpose(1, 2).contiguous()

    @staticmethod
    def _weighted_region_pool(region_tokens: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        weighted = region_tokens * gate
        denom = gate.sum(dim=1).clamp_min(1e-6)
        return weighted.sum(dim=1) / denom

    def forward(self, *, shared, pose_feat, face_feat, x_occ=None):
        del pose_feat
        tokens = self._face_region_tokens(face_feat)
        tokens = self.relation(tokens)
        n, r, _ = tokens.shape

        reliability = occ_to_region_reliability(
            x_occ,
            num_regions=r,
            occ_dim=self.occ_dim,
            region_occ_indices=self.region_occ_indices,
            default_visible=self.default_visible,
            mask_strength=self.mask_strength,
            min_reliability=self.min_reliability,
        )
        if reliability is None:
            reliability = torch.ones(n, r, device=tokens.device, dtype=tokens.dtype)
        reliability = reliability.unsqueeze(-1)

        out = {}
        for name in TASK_NAMES:
            gate = self.gates[name](tokens, x_occ=x_occ)
            gate = gate * reliability
            face_vec = self._weighted_region_pool(tokens, gate)
            face_vec = self.face_proj(face_vec)
            out[name] = shared + self.gate_feature_scale * face_vec
        return out
