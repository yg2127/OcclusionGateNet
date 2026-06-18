"""OCC-token region transformer fusion.

3-B variant:
    occ_token = MLP(x_occ)
    tokens = [occ_token, face_region_tokens]
    tokens = Transformer(tokens)
    region tokens -> task region gate -> weighted pool.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .task_feature_fusion import TASK_NAMES, TaskFeatureFusion, TaskRegionGate


class OccTokenRegionTransformerFusion(TaskFeatureFusion):
    """Inject x_occ as a dedicated token before region self-attention."""

    def __init__(
        self,
        *,
        fused_channels: int,
        face_channels: int,
        occ_dim: int = 0,
        occ_token_hidden_dim: int = 128,
        gate_hidden_channels: int = 128,
        gate_dropout: float = 0.2,
        gate_feature_scale: float = 0.25,
        init_bias: dict | None = None,
        region_num_heads: int = 4,
        region_num_layers: int = 1,
        region_dropout: float = 0.1,
        region_ff_mult: int = 2,
        gate_condition_occ: bool = True,
        use_occ_token_residual: bool = True,
    ):
        super().__init__()
        if occ_dim <= 0:
            raise ValueError("occ_token_region_transformer requires occ.enabled=true and occ.dim > 0")

        init_bias = init_bias or {}
        self.out_dim = int(fused_channels)
        self.occ_dim = int(occ_dim)
        self.gate_feature_scale = float(gate_feature_scale)
        self.use_occ_token_residual = bool(use_occ_token_residual)

        self.occ_to_token = nn.Sequential(
            nn.LayerNorm(self.occ_dim),
            nn.Linear(self.occ_dim, occ_token_hidden_dim),
            nn.GELU(),
            nn.Dropout(region_dropout),
            nn.Linear(occ_token_hidden_dim, face_channels),
        )

        if face_channels % region_num_heads != 0:
            raise ValueError(
                f"face_channels must be divisible by region_num_heads: "
                f"{face_channels=} {region_num_heads=}"
            )
        layer = nn.TransformerEncoderLayer(
            d_model=face_channels,
            nhead=region_num_heads,
            dim_feedforward=face_channels * region_ff_mult,
            dropout=region_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=region_num_layers)

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
        self.occ_residual_proj = nn.Sequential(
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
        if x_occ is None:
            raise ValueError("occ_token_region_transformer requires x_occ but received None")

        region_tokens = self._face_region_tokens(face_feat)
        occ_token = self.occ_to_token(x_occ).unsqueeze(1)
        tokens = torch.cat([occ_token, region_tokens], dim=1)
        tokens = self.encoder(tokens)

        occ_context = tokens[:, 0]
        region_tokens = tokens[:, 1:]
        occ_residual = self.occ_residual_proj(occ_context) if self.use_occ_token_residual else 0.0

        out = {}
        for name in TASK_NAMES:
            gate = self.gates[name](region_tokens, x_occ=x_occ)
            face_vec = self._weighted_region_pool(region_tokens, gate)
            face_vec = self.face_proj(face_vec)
            out[name] = shared + self.gate_feature_scale * (face_vec + occ_residual)
        return out
