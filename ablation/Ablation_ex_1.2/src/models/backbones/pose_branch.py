"""Pose branch: (N, 7, T, 17) → (N, C_mid, T/2, 17). TGCBlock × 4."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class GraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, A: np.ndarray):
        super().__init__()
        self.register_buffer("A", torch.tensor(A, dtype=torch.float32))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.einsum("nctv,vw->nctw", x, self.A)
        return x


class TGCBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, A: np.ndarray, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        self.gconv = GraphConv(in_channels, out_channels, A)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.tconv = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=(9, 1), stride=(stride, 1), padding=(4, 0), bias=False,
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
        out = self.gconv(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.tconv(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out = out + res
        out = self.act(out)
        return out


class PoseBranch(nn.Module):
    """(N, C_in, T, V=17) → (N, C_mid, T/2, V=17)."""
    def __init__(self, in_channels: int, mid_channels: int, A: np.ndarray, num_joints: int = 17, dropout: float = 0.1):
        super().__init__()
        self.num_joints = num_joints
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)

        self.layer1 = TGCBlock(in_channels, 64, A, stride=1, dropout=dropout)
        self.layer2 = TGCBlock(64, 64, A, stride=1, dropout=dropout)
        self.layer3 = TGCBlock(64, mid_channels, A, stride=2, dropout=dropout)   # T/2
        self.layer4 = TGCBlock(mid_channels, mid_channels, A, stride=1, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V = x.shape
        x = x.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        x = self.data_bn(x)
        x = x.view(N, C, V, T).permute(0, 1, 3, 2).contiguous()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
