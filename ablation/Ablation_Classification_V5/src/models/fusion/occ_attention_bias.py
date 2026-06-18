"""OCC attention-bias region transformer fusion.

3-C variant:
    region reliability from x_occ is converted to an additive attention bias.
    Low-reliability regions receive negative key-side attention bias.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .region_occ_utils import occ_to_region_reliability
from .task_feature_fusion import TASK_NAMES, TaskFeatureFusion, TaskRegionGate


class OccBiasedSelfAttentionBlock(nn.Module):
    """Transformer-style block with per-sample additive attention bias."""

    def __init__(self, *, channels: int, num_heads: int, dropout: float, ff_mult: int):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels must be divisible by num_heads: {channels=} {num_heads=}")
        self.num_heads = int(num_heads)
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * ff_mult, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        attn_mask = None
        if attn_bias is not None:
            # attn_bias: (N, R, R). MultiheadAttention expects
            # (N * num_heads, R, R) for per-sample masks.
            attn_mask = attn_bias.repeat_interleave(self.num_heads, dim=0)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.drop1(a)
        x = x + self.ffn(self.norm2(x))
        return x


class OccAttentionBiasFusion(TaskFeatureFusion):
    """Region transformer where x_occ becomes attention bias, not just a token."""

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
        attention_bias_strength: float = 2.0,
        gate_condition_occ: bool = True,
    ):
        super().__init__()
        if occ_dim <= 0:
            raise ValueError("occ_attention_bias requires occ.enabled=true and occ.dim > 0")

        init_bias = init_bias or {}
        self.out_dim = int(fused_channels)
        self.occ_dim = int(occ_dim)
        self.gate_feature_scale = float(gate_feature_scale)
        self.region_occ_indices = None if region_occ_indices is None else [int(x) for x in region_occ_indices]
        self.default_visible = float(default_visible)
        self.mask_strength = float(mask_strength)
        self.min_reliability = float(min_reliability)
        self.attention_bias_strength = float(attention_bias_strength)

        self.layers = nn.ModuleList(
            [
                OccBiasedSelfAttentionBlock(
                    channels=face_channels,
                    num_heads=region_num_heads,
                    dropout=region_dropout,
                    ff_mult=region_ff_mult,
                )
                for _ in range(region_num_layers)
            ]
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

    def _build_attention_bias(self, x_occ: torch.Tensor | None, *, num_regions: int) -> torch.Tensor | None:
        reliability = occ_to_region_reliability(
            x_occ,
            num_regions=num_regions,
            occ_dim=self.occ_dim,
            region_occ_indices=self.region_occ_indices,
            default_visible=self.default_visible,
            mask_strength=self.mask_strength,
            min_reliability=self.min_reliability,
        )
        if reliability is None:
            return None

        # Key-side bias: unreliable key tokens get negative bias for all queries.
        key_bias = torch.log(reliability.clamp_min(self.min_reliability))
        key_bias = self.attention_bias_strength * key_bias
        return key_bias[:, None, :].expand(-1, num_regions, -1).contiguous()

    def forward(self, *, shared, pose_feat, face_feat, x_occ=None):
        del pose_feat
        tokens = self._face_region_tokens(face_feat)
        _, r, _ = tokens.shape
        attn_bias = self._build_attention_bias(x_occ, num_regions=r)

        for layer in self.layers:
            tokens = layer(tokens, attn_bias=attn_bias)

        out = {}
        for name in TASK_NAMES:
            gate = self.gates[name](tokens, x_occ=x_occ)
            face_vec = self._weighted_region_pool(tokens, gate)
            face_vec = self.face_proj(face_vec)
            out[name] = shared + self.gate_feature_scale * face_vec
        return out
