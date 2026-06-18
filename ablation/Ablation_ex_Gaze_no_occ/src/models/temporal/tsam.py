"""TSAM: Temporal-Spatial Attention Module. swappable 설계 — kind="identity"도 지원."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        return x + self.pe[:, :L, :]


class TSAM(nn.Module):
    """(N, C, T, V) → self-attention across T·V tokens → (N, C, T, V)."""
    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.1, max_len: int = 4096):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.pos_enc = SinusoidalPositionalEncoding(channels, max_len=max_len)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V = x.shape
        tokens = x.permute(0, 2, 3, 1).contiguous().view(N, T * V, C)
        tokens = self.norm1(tokens)
        tokens = self.pos_enc(tokens)
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = tokens + self.dropout(attn_out)
        tokens = tokens + self.ffn(self.norm2(tokens))
        out = tokens.view(N, T, V, C).permute(0, 3, 1, 2).contiguous()
        return out


class IdentityTemporal(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def build_temporal(kind: str, channels: int, num_heads: int = 4, dropout: float = 0.1, max_len: int = 4096) -> nn.Module:
    k = kind.lower()
    if k == "tsam":
        return TSAM(channels=channels, num_heads=num_heads, dropout=dropout, max_len=max_len)
    if k == "identity":
        return IdentityTemporal()
    raise ValueError(f"unknown temporal.kind: {kind}")
