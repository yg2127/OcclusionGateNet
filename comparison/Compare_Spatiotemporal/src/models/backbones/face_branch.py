"""Face branch: (N, 4, T, K=10) → (N, C_mid, T/2, K). 1x1 conv + temporal conv (GCN 없이)."""
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 9, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = kernel_size // 2
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.tconv = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(pad, 0), bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)

        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        out = self.pointwise(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.tconv(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out = out + res
        out = self.act(out)
        return out


class FaceBranch(nn.Module):
    """(N, C_in, T, K) → (N, C_mid, T/2, K)."""
    def __init__(self, in_channels: int, mid_channels: int, num_regions: int = 10, dropout: float = 0.1):
        super().__init__()
        self.num_regions = num_regions
        self.data_bn = nn.BatchNorm1d(in_channels * num_regions)

        hidden = max(32, mid_channels // 2)
        self.block1 = TemporalConvBlock(in_channels, hidden, kernel_size=9, stride=1, dropout=dropout)
        self.block2 = TemporalConvBlock(hidden, mid_channels, kernel_size=9, stride=2, dropout=dropout)  # T/2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, K = x.shape
        x = x.permute(0, 1, 3, 2).contiguous().view(N, C * K, T)
        x = self.data_bn(x)
        x = x.view(N, C, K, T).permute(0, 1, 3, 2).contiguous()
        x = self.block1(x)
        x = self.block2(x)
        return x
