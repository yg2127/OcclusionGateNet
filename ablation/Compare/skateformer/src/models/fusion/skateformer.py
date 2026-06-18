"""SkateFormer-inspired DMS comparator.

This module adapts the main SkateFormer idea to the existing DMS fixed
clean/masked protocol:

- skeleton stream: Skate-type partition attention over YOLO-17 pose features,
- optional FaceMesh stream: shallow region-feature projection,
- shared branch is deliberately disabled to keep the comparator lightweight,
- OCC/reliability inputs are ignored.

It is not a full reproduction of the original SkateFormer training recipe.
The goal is a controlled feature-level comparator under the same DMS data,
split, evaluation and drop-calculation pipeline.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .task_feature_fusion import TASK_NAMES, TaskFeatureFusion


# Neighboring joint partitions for YOLO-17. Each row is padded to the same
# length. The ordering roughly follows body-part chains so the same position
# across rows forms a "distant-joint" partition, following SkateFormer.
YOLO17_NEIGHBOR_GROUPS = [
    [0, 1, 3, 5],       # left face -> left shoulder
    [0, 2, 4, 6],       # right face -> right shoulder
    [5, 7, 9, 11],      # left arm -> left hip
    [6, 8, 10, 12],     # right arm -> right hip
    [11, 13, 15, 16],   # lower chain / pad-like endpoint
]


def _make_group_tensors(groups: Iterable[Iterable[int]], *, num_joints: int) -> tuple[torch.Tensor, torch.Tensor]:
    groups = [list(g) for g in groups]
    k = len(groups)
    l = max(len(g) for g in groups)
    idx = torch.zeros(k, l, dtype=torch.long)
    valid = torch.zeros(k, l, dtype=torch.bool)
    for i, g in enumerate(groups):
        for j, v in enumerate(g):
            if 0 <= int(v) < num_joints:
                idx[i, j] = int(v)
                valid[i, j] = True
    return idx, valid


class LearnableGraphConv(nn.Module):
    """Small learnable graph-conv branch used as SkateFormer inductive bias."""

    def __init__(self, channels: int, num_joints: int, dropout: float = 0.1):
        super().__init__()
        self.num_joints = int(num_joints)
        self.adj_logits = nn.Parameter(torch.zeros(num_joints, num_joints))
        nn.init.normal_(self.adj_logits, std=0.02)
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.softmax(self.adj_logits, dim=-1)
        x = torch.einsum('nctv,vw->nctw', x, a)
        return self.proj(x)


class TemporalConvBranch(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        pad = int(kernel_size) // 2
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), padding=(pad, 0), groups=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SkatePartitionMSA(nn.Module):
    """Lightweight Skate-Type partition mixer.

    This keeps the four SkateFormer relation types but replaces expensive
    pairwise MSA with partition-wise context mixing. It is intentionally light
    for the DMS comparator, where the goal is clean/masked drop comparison
    under the same data protocol rather than full SkateFormer reproduction.

    Types:
      1) neighboring joints + local motion
      2) distant joints     + local motion
      3) neighboring joints + global motion
      4) distant joints     + global motion

    Input/output: (N, C, T, V).
    """

    def __init__(
        self,
        *,
        channels: int,
        num_joints: int = 17,
        local_size: int = 4,
        num_heads_per_type: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        del num_heads_per_type
        self.channels = int(channels)
        self.num_joints = int(num_joints)
        self.local_size = int(local_size)
        if self.channels % 4 != 0:
            raise ValueError(f'SkatePartitionMSA channels must be divisible by 4, got {channels}')
        self.branch_channels = self.channels // 4

        idx, valid = _make_group_tensors(YOLO17_NEIGHBOR_GROUPS, num_joints=num_joints)
        self.register_buffer('group_idx', idx, persistent=False)       # (K,L)
        self.register_buffer('group_valid', valid, persistent=False)   # (K,L)
        self.branch_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.branch_channels, self.branch_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.branch_channels),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(4)
        ])
        self.scale = nn.Parameter(torch.ones(4))
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Dropout(dropout),
        )

    def _gather_groups(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,C,T,V) -> (N,C,T,K,L)
        flat_idx = self.group_idx.reshape(-1).to(device=x.device)
        xg = x.index_select(dim=-1, index=flat_idx)
        k, l = self.group_idx.shape
        valid = self.group_valid.to(device=x.device, dtype=x.dtype).view(1, 1, 1, k, l)
        return xg.view(x.shape[0], x.shape[1], x.shape[2], k, l) * valid

    def _scatter_groups(self, xg: torch.Tensor, *, original_t: int) -> torch.Tensor:
        # xg: (N,C,Tpad,K,L) -> (N,C,T,V), averaging duplicated indices.
        n, c, tpad, k, l = xg.shape
        out = xg.new_zeros(n, c, tpad, self.num_joints)
        cnt = xg.new_zeros(1, 1, tpad, self.num_joints)
        valid = self.group_valid.to(device=xg.device)
        idx = self.group_idx.to(device=xg.device)
        for i in range(k):
            for j in range(l):
                if bool(valid[i, j]):
                    joint = int(idx[i, j].item())
                    out[:, :, :, joint] += xg[:, :, :, i, j]
                    cnt[:, :, :, joint] += 1.0
        out = out / cnt.clamp_min(1.0)
        return out[:, :, :original_t, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, t, v = x.shape
        if v != self.num_joints:
            raise ValueError(f'expected V={self.num_joints}, got {v}')

        branches = torch.chunk(x, 4, dim=1)
        k, l = self.group_idx.shape
        local = self.local_size
        m = math.ceil(t / local)
        tpad = m * local
        valid_kl = self.group_valid.to(device=x.device, dtype=x.dtype).view(1, 1, 1, k, l)

        outputs: list[torch.Tensor] = []
        for bidx, xb in enumerate(branches):
            xb_pad = F.pad(xb, (0, 0, 0, tpad - t)) if tpad > t else xb
            xg = self._gather_groups(xb_pad)  # (N,C,Tpad,K,L)
            xgm = xg.view(n, self.branch_channels, m, local, k, l)

            if bidx == 0:
                # neighboring joints + local motion: context per local window and neighboring part
                denom = valid_kl.sum(dim=-1, keepdim=True).clamp_min(1.0) * float(local)
                ctx = xgm.sum(dim=(3, 5), keepdim=True) / denom.view(1, 1, 1, 1, k, 1)
                yg = xgm + self.scale[bidx] * ctx
            elif bidx == 1:
                # distant joints + local motion: context across same chain-depth among body parts
                denom = valid_kl.sum(dim=-2, keepdim=True).clamp_min(1.0) * float(local)
                ctx = xgm.sum(dim=(3, 4), keepdim=True) / denom.view(1, 1, 1, 1, 1, l)
                yg = xgm + self.scale[bidx] * ctx
            elif bidx == 2:
                # neighboring joints + global motion: same local offset across the full clip
                denom = valid_kl.sum(dim=-1, keepdim=True).clamp_min(1.0) * float(m)
                ctx = xgm.sum(dim=(2, 5), keepdim=True) / denom.view(1, 1, 1, 1, k, 1)
                yg = xgm + self.scale[bidx] * ctx
            else:
                # distant joints + global motion
                denom = valid_kl.sum(dim=-2, keepdim=True).clamp_min(1.0) * float(m)
                ctx = xgm.sum(dim=(2, 4), keepdim=True) / denom.view(1, 1, 1, 1, 1, l)
                yg = xgm + self.scale[bidx] * ctx

            yg = yg.view(n, self.branch_channels, tpad, k, l)
            out = self._scatter_groups(yg, original_t=t)
            outputs.append(self.branch_proj[bidx](out))

        return self.out_proj(torch.cat(outputs, dim=1))


class SkateFormerBlock(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        num_joints: int = 17,
        local_size: int = 4,
        num_heads_per_type: int = 1,
        kernel_size: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        if channels < 8 or channels % 4 != 0:
            raise ValueError('SkateFormerBlock channels must be divisible by 4')
        self.norm1 = nn.BatchNorm2d(channels)
        c_gc = channels // 4
        c_tc = channels // 4
        c_msa = channels - c_gc - c_tc
        self.c_gc, self.c_tc, self.c_msa = c_gc, c_tc, c_msa

        self.pre = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.gc = LearnableGraphConv(c_gc, num_joints=num_joints, dropout=dropout)
        self.tc = TemporalConvBranch(c_tc, kernel_size=kernel_size, dropout=dropout)
        self.msa = SkatePartitionMSA(
            channels=c_msa,
            num_joints=num_joints,
            local_size=local_size,
            num_heads_per_type=num_heads_per_type,
            dropout=dropout,
        )
        self.mix = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.BatchNorm2d(channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Dropout(dropout),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pre(self.norm1(x))
        x_gc, x_tc, x_msa = torch.split(y, [self.c_gc, self.c_tc, self.c_msa], dim=1)
        y = torch.cat([self.gc(x_gc), self.tc(x_tc), self.msa(x_msa)], dim=1)
        x = self.act(x + self.mix(y))
        x = self.act(x + self.ffn(self.norm2(x)))
        return x


class SkateFormerFusion(TaskFeatureFusion):
    """Skeleton-only SkateFormer-inspired comparator."""

    uses_shared = False
    uses_face = False

    def __init__(
        self,
        *,
        fused_channels: int,
        pose_channels: int,
        face_channels: int,
        num_joints: int = 17,
        num_layers: int = 2,
        local_size: int = 4,
        num_heads_per_type: int = 1,
        kernel_size: int = 7,
        dropout: float = 0.1,
        task_adapter: bool = True,
    ):
        super().__init__()
        del face_channels
        self.out_dim = int(fused_channels)
        self.pose_proj = nn.Sequential(
            nn.Conv2d(pose_channels, fused_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fused_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(*[
            SkateFormerBlock(
                channels=fused_channels,
                num_joints=num_joints,
                local_size=local_size,
                num_heads_per_type=num_heads_per_type,
                kernel_size=kernel_size,
                dropout=dropout,
            )
            for _ in range(int(num_layers))
        ])
        self.out_norm = nn.LayerNorm(fused_channels)
        self.task_adapter_enabled = bool(task_adapter)
        if self.task_adapter_enabled:
            self.task_adapters = nn.ModuleDict({
                name: nn.Sequential(
                    nn.LayerNorm(fused_channels),
                    nn.Linear(fused_channels, fused_channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(fused_channels, fused_channels),
                )
                for name in TASK_NAMES
            })
            for adapter in self.task_adapters.values():
                nn.init.zeros_(adapter[-1].weight)
                nn.init.zeros_(adapter[-1].bias)

    def forward(self, *, shared, pose_feat, face_feat, x_occ=None):
        del shared, face_feat, x_occ
        x = self.pose_proj(pose_feat)
        x = self.blocks(x)
        z = x.mean(dim=-1).mean(dim=-1)
        z = self.out_norm(z)
        if not self.task_adapter_enabled:
            return {name: z for name in TASK_NAMES}
        return {name: z + self.task_adapters[name](z) for name in TASK_NAMES}


class SkateFormerFaceFusion(TaskFeatureFusion):
    """SkateFormer pose stream + lightweight FaceMesh stream.

    FaceMesh is mixed as an auxiliary stream, not as part of the Skate-MSA
    skeletal partition. This keeps the original paper's skeleton-based logic in
    the pose stream while making the comparator compatible with the user's DMS
    pose+FaceMesh experimental framing. The expensive shared TGC branch is off.
    """

    uses_shared = False
    uses_face = True

    def __init__(
        self,
        *,
        fused_channels: int,
        pose_channels: int,
        face_channels: int,
        num_joints: int = 17,
        num_layers: int = 2,
        local_size: int = 4,
        num_heads_per_type: int = 1,
        kernel_size: int = 7,
        dropout: float = 0.1,
        task_adapter: bool = True,
    ):
        super().__init__()
        self.out_dim = int(fused_channels)
        self.pose_proj = nn.Sequential(
            nn.Conv2d(pose_channels, fused_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fused_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(*[
            SkateFormerBlock(
                channels=fused_channels,
                num_joints=num_joints,
                local_size=local_size,
                num_heads_per_type=num_heads_per_type,
                kernel_size=kernel_size,
                dropout=dropout,
            )
            for _ in range(int(num_layers))
        ])
        self.face_proj = nn.Sequential(
            nn.Conv2d(face_channels, fused_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fused_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.merge = nn.Sequential(
            nn.LayerNorm(fused_channels * 2),
            nn.Linear(fused_channels * 2, fused_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_channels, fused_channels),
            nn.LayerNorm(fused_channels),
        )
        self.task_adapter_enabled = bool(task_adapter)
        if self.task_adapter_enabled:
            self.task_adapters = nn.ModuleDict({
                name: nn.Sequential(
                    nn.LayerNorm(fused_channels),
                    nn.Linear(fused_channels, fused_channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(fused_channels, fused_channels),
                )
                for name in TASK_NAMES
            })
            for adapter in self.task_adapters.values():
                nn.init.zeros_(adapter[-1].weight)
                nn.init.zeros_(adapter[-1].bias)

    def forward(self, *, shared, pose_feat, face_feat, x_occ=None):
        del shared, x_occ
        if face_feat is None:
            raise ValueError('SkateFormerFaceFusion requires face_feat.')
        pose_x = self.pose_proj(pose_feat)
        pose_x = self.blocks(pose_x)
        pose_z = pose_x.mean(dim=-1).mean(dim=-1)

        face_x = self.face_proj(face_feat)
        face_z = face_x.mean(dim=-1).mean(dim=-1)

        z = self.merge(torch.cat([pose_z, face_z], dim=-1))
        if not self.task_adapter_enabled:
            return {name: z for name in TASK_NAMES}
        return {name: z + self.task_adapters[name](z) for name in TASK_NAMES}
