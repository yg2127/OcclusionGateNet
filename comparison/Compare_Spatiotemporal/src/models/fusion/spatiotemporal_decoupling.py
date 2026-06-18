"""SDA-TR-inspired skeleton-only task fusion for the existing DMS protocol.

This is not a full MMAction2 reproduction of the paper. It adapts the core
ideas of SDA-TR to the current fixed clean/masked DMS pipeline:

1) use only the YOLO pose skeleton feature stream for classification,
2) construct joint dependencies across short multi-frame tuples,
3) decouple attention by skeleton-graph distance groups,
4) aggregate temporal features at sub-action and frame levels,
5) keep the existing four-head DMS output and clean-vs-masked drop evaluation.

The default class below is skeleton-only. A second class, SpatiotemporalDecouplingFaceFusion, keeps the same SDA-TR-inspired skeleton stream and additionally fuses FaceMesh region features from the existing DMS pipeline. OCC/reliability inputs remain ignored.
"""
from __future__ import annotations

import math
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from .task_feature_fusion import TASK_NAMES, TaskFeatureFusion


# COCO/YOLO-17 skeleton edges. This is the same joint convention normally used
# by Ultralytics YOLO pose: nose, eyes, ears, shoulders, elbows, wrists, hips,
# knees, ankles.
YOLO17_EDGES = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]


def _graph_distance(num_joints: int, edges: list[tuple[int, int]]) -> torch.Tensor:
    adj = [[] for _ in range(num_joints)]
    for a, b in edges:
        if a < num_joints and b < num_joints:
            adj[a].append(b)
            adj[b].append(a)
    dist = torch.full((num_joints, num_joints), 10_000, dtype=torch.long)
    for s in range(num_joints):
        dist[s, s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[s, v] > dist[s, u] + 1:
                    dist[s, v] = dist[s, u] + 1
                    q.append(v)
    return dist


def _sinusoidal_position(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    pos = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / max(dim, 1)))
    pe = torch.zeros(length, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(pos * div)
    if dim > 1:
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe


class DecoupledSpatiotemporalAttention(nn.Module):
    """Self-attention over V*tau joints with graph-distance decoupled masks."""

    def __init__(
        self,
        *,
        channels: int,
        num_joints: int = 17,
        tau: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_position: bool = True,
    ):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels must be divisible by num_heads: {channels=} {num_heads=}")
        self.channels = int(channels)
        self.num_joints = int(num_joints)
        self.tau = int(tau)
        self.num_heads = int(num_heads)
        self.head_dim = self.channels // self.num_heads
        self.use_position = bool(use_position)

        self.q = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.v = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Dropout(dropout),
        )
        # Two-scale default from the paper: directly connected/self vs farther joints.
        self.alpha = nn.Parameter(torch.ones(2))
        dist = _graph_distance(num_joints, YOLO17_EDGES)
        near = dist <= 1
        far = dist >= 2
        self.register_buffer('near_mask_base', near, persistent=False)
        self.register_buffer('far_mask_base', far, persistent=False)

    def _make_tuple_masks(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        # Repeat the single-frame VxV mask across tau-by-tau temporal blocks so
        # the same role group can interact across frames within a sub-action tuple.
        near = self.near_mask_base.to(device=device)
        far = self.far_mask_base.to(device=device)
        near = near.repeat(self.tau, self.tau)
        far = far.repeat(self.tau, self.tau)
        return near, far

    def _partition(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        # (N,C,T,V) -> (N,C,T0,V*tau), padding T if necessary.
        n, c, t, v = x.shape
        pad = (self.tau - (t % self.tau)) % self.tau
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        tp = t + pad
        x = x.view(n, c, tp // self.tau, self.tau, v)
        x = x.permute(0, 1, 2, 4, 3).contiguous().view(n, c, tp // self.tau, v * self.tau)
        return x, t

    def _reverse(self, x: torch.Tensor, original_t: int) -> torch.Tensor:
        # (N,C,T0,V*tau) -> (N,C,T,V)
        n, c, t0, vt = x.shape
        v = self.num_joints
        x = x.view(n, c, t0, v, self.tau).permute(0, 1, 2, 4, 3).contiguous()
        x = x.view(n, c, t0 * self.tau, v)
        return x[:, :, :original_t, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, t, v = x.shape
        if v != self.num_joints:
            raise ValueError(f"expected V={self.num_joints}, got {v}")
        xt, original_t = self._partition(x)
        n, c, t0, v0 = xt.shape

        if self.use_position:
            pe = _sinusoidal_position(v0, c, xt.device, xt.dtype).transpose(0, 1).view(1, c, 1, v0)
            xt = xt + pe

        q = self.q(xt).view(n, self.num_heads, self.head_dim, t0, v0)
        k = self.k(xt).view(n, self.num_heads, self.head_dim, t0, v0)
        val = self.v(xt).view(n, self.num_heads, self.head_dim, t0, v0)

        logits = torch.einsum('nhctv,nhctw->nhtvw', q, k) / math.sqrt(self.head_dim)
        near, far = self._make_tuple_masks(xt.device)
        masks = [near, far]
        attn_sum = 0.0
        for idx, mask in enumerate(masks):
            masked_logits = logits.masked_fill(~mask.view(1, 1, 1, v0, v0), -1e4)
            attn = torch.softmax(masked_logits, dim=-1)
            attn_sum = attn_sum + self.alpha[idx] * attn

        out = torch.einsum('nhtvw,nhctw->nhctv', attn_sum, val).contiguous()
        out = out.view(n, c, t0, v0)
        out = self.proj(out)
        out = self._reverse(out, original_t)
        return out


class TemporalFeatureAggregation(nn.Module):
    """Sub-action-level + frame-level temporal aggregation."""

    def __init__(self, *, channels: int, tau: int = 4, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        self.tau = int(tau)
        k = int(kernel_size)
        pad = k // 2
        dil_pad = (k // 2) * 2

        def bbtc():
            return nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=(k, 1), padding=(pad, 0), bias=False),
                nn.BatchNorm2d(channels),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        def bbtc_dilated():
            return nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=(k, 1), padding=(dil_pad, 0), dilation=(2, 1), bias=False),
                nn.BatchNorm2d(channels),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.pattern_b1 = bbtc()
        self.pattern_b2 = bbtc_dilated()
        self.pattern_merge = nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False), nn.BatchNorm2d(channels))

        self.frame_b1 = bbtc()
        self.frame_b2 = bbtc_dilated()
        self.frame_merge = nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False), nn.BatchNorm2d(channels))
        self.act = nn.GELU()

    def _subaction_pool(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        n, c, t, v = x.shape
        pad = (self.tau - (t % self.tau)) % self.tau
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        tp = t + pad
        pooled = x.view(n, c, tp // self.tau, self.tau, v).mean(dim=3)
        return pooled, t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p, original_t = self._subaction_pool(x)
        p = self.pattern_merge(torch.cat([self.pattern_b1(p), self.pattern_b2(p)], dim=1))
        p = p.repeat_interleave(self.tau, dim=2)[:, :, :original_t, :]
        f = self.frame_merge(torch.cat([self.frame_b1(x), self.frame_b2(x)], dim=1))
        return self.act(x + p + f)


class SDATRBlock(nn.Module):
    def __init__(self, *, channels: int, num_joints: int, tau: int, num_heads: int, kernel_size: int, dropout: float):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(channels)
        self.sda = DecoupledSpatiotemporalAttention(
            channels=channels,
            num_joints=num_joints,
            tau=tau,
            num_heads=num_heads,
            dropout=dropout,
            use_position=True,
        )
        self.norm2 = nn.BatchNorm2d(channels)
        self.tfa = TemporalFeatureAggregation(channels=channels, tau=tau, kernel_size=kernel_size, dropout=dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(x + self.sda(self.norm1(x)))
        x = self.act(x + self.tfa(self.norm2(x)))
        return x


class SpatiotemporalDecouplingFusion(TaskFeatureFusion):
    """SDA-TR-inspired skeleton-only fusion used as a DMS comparison model."""

    uses_shared = False

    def __init__(
        self,
        *,
        fused_channels: int,
        pose_channels: int,
        face_channels: int,
        num_joints: int = 17,
        num_layers: int = 3,
        tau: int = 4,
        num_heads: int = 4,
        kernel_size: int = 7,
        dropout: float = 0.1,
        task_adapter: bool = True,
    ):
        super().__init__()
        del face_channels
        self.out_dim = int(fused_channels)
        self.input_proj = nn.Sequential(
            nn.Conv2d(pose_channels, fused_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fused_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(*[
            SDATRBlock(
                channels=fused_channels,
                num_joints=num_joints,
                tau=tau,
                num_heads=num_heads,
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
                last = adapter[-1]
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def forward(self, *, shared, pose_feat, face_feat, x_occ=None):
        del shared, face_feat, x_occ
        x = self.input_proj(pose_feat)
        x = self.blocks(x)
        z = x.mean(dim=-1).mean(dim=-1)
        z = self.out_norm(z)
        if not self.task_adapter_enabled:
            return {name: z for name in TASK_NAMES}
        return {name: z + self.task_adapters[name](z) for name in TASK_NAMES}


class SpatiotemporalDecouplingFaceFusion(TaskFeatureFusion):
    """SDA-TR-inspired pose stream + lightweight FaceMesh region comparator.

    This variant keeps the SDA/TFA skeleton stream from SDA-TR and adds a
    shallow FaceMesh region stream. It deliberately disables the expensive
    shared branch (ConcatJointFusion + TGCBlock stack). The model therefore uses
    FaceMesh evidence but avoids rebuilding the previous full pose-face fusion
    path. It is meant as a practical DMS comparator under the same clean/masked
    fixed-split protocol, not as a full reproduction of the original
    skeleton-only paper.
    """

    # Ask MultitaskClassifier to compute face_feat, but not shared.
    uses_shared = False
    uses_face = True

    def __init__(
        self,
        *,
        fused_channels: int,
        pose_channels: int,
        face_channels: int,
        num_joints: int = 17,
        num_layers: int = 3,
        tau: int = 4,
        num_heads: int = 4,
        kernel_size: int = 7,
        dropout: float = 0.1,
        task_adapter: bool = True,
    ):
        super().__init__()
        self.out_dim = int(fused_channels)
        self.pose_input_proj = nn.Sequential(
            nn.Conv2d(pose_channels, fused_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fused_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pose_blocks = nn.Sequential(*[
            SDATRBlock(
                channels=fused_channels,
                num_joints=num_joints,
                tau=tau,
                num_heads=num_heads,
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
                last = adapter[-1]
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def forward(self, *, shared, pose_feat, face_feat, x_occ=None):
        del shared, x_occ
        if face_feat is None:
            raise ValueError(
                'SpatiotemporalDecouplingFaceFusion requires face_feat. '
                'Check MultitaskClassifier uses_face path.'
            )

        pose_x = self.pose_input_proj(pose_feat)
        pose_x = self.pose_blocks(pose_x)
        pose_z = pose_x.mean(dim=-1).mean(dim=-1)

        face_x = self.face_proj(face_feat)
        face_z = face_x.mean(dim=-1).mean(dim=-1)

        z = self.merge(torch.cat([pose_z, face_z], dim=-1))

        if not self.task_adapter_enabled:
            return {name: z for name in TASK_NAMES}
        return {name: z + self.task_adapters[name](z) for name in TASK_NAMES}
