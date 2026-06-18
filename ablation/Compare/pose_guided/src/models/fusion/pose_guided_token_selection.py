"""PO-GUISE-inspired pose-guided token selection fusion for the existing DMS pipeline.

This module is intentionally not a full RGB VideoMAEv2 reimplementation.
It keeps the current Ablation-B data/protocol unchanged:
    - x_body: YOLO pose skeleton window
    - x_face: FaceMesh/region pooled window
    - clean+masked fixed manifest split
    - clean vs masked test and drop computation

The comparison target is the core PO-GUISE idea:
    1) represent pose and face/visual evidence as tokens,
    2) use class + pose tokens to select task-relevant face/visual tokens,
    3) merge dropped tokens so discarded information is not fully lost,
    4) classify with the same four task heads used by the project.

No reliability/occlusion confidence branch is used here. x_occ is ignored.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .task_feature_fusion import TASK_NAMES, TaskFeatureFusion


class PoseGuidedTokenSelectionFusion(TaskFeatureFusion):
    """Pose-guided token-selection fusion.

    Inputs from the existing backbone:
        shared    : (N, D)         fused global representation
        pose_feat : (N, C_pose, T, V_pose)
        face_feat : (N, C_face, T, R_face)

    The module builds a short token sequence:
        [class token, pose tokens, selected face tokens, merged dropped token]

    Face/region tokens are scored by their similarity to the class token and pose
    tokens. The top-k face tokens are retained. The remaining tokens are averaged
    into one merged residual token, mirroring the paper's motivation that pruned
    tokens should not be simply thrown away.
    """

    def __init__(
        self,
        *,
        fused_channels: int,
        pose_channels: int,
        face_channels: int,
        keep_ratio: float = 0.6,
        kappa: float = 0.5,
        num_layers: int = 1,
        num_heads: int = 4,
        ff_mult: int = 2,
        dropout: float = 0.1,
        task_adapter: bool = True,
    ):
        super().__init__()
        self.out_dim = int(fused_channels)
        self.keep_ratio = float(keep_ratio)
        self.kappa = float(kappa)
        if not (0.0 < self.keep_ratio <= 1.0):
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
        if not (0.0 <= self.kappa <= 1.0):
            raise ValueError(f"kappa must be in [0, 1], got {kappa}")
        if fused_channels % num_heads != 0:
            raise ValueError(
                f"fused_channels must be divisible by num_heads: "
                f"{fused_channels=} {num_heads=}"
            )

        self.shared_norm = nn.LayerNorm(fused_channels)
        self.pose_proj = nn.Sequential(
            nn.LayerNorm(pose_channels),
            nn.Linear(pose_channels, fused_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.face_proj = nn.Sequential(
            nn.LayerNorm(face_channels),
            nn.Linear(face_channels, fused_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.merge_proj = nn.Sequential(
            nn.LayerNorm(fused_channels),
            nn.Linear(fused_channels, fused_channels),
            nn.GELU(),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=fused_channels,
            nhead=num_heads,
            dim_feedforward=fused_channels * int(ff_mult),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.out_norm = nn.LayerNorm(fused_channels)

        self.task_adapter_enabled = bool(task_adapter)
        if self.task_adapter_enabled:
            self.task_adapters = nn.ModuleDict(
                {
                    name: nn.Sequential(
                        nn.LayerNorm(fused_channels),
                        nn.Linear(fused_channels, fused_channels),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(fused_channels, fused_channels),
                    )
                    for name in TASK_NAMES
                }
            )
            # Start as near-residual adapters to avoid destabilizing early epochs.
            for adapter in self.task_adapters.values():
                last = adapter[-1]
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    @staticmethod
    def _pool_temporal_tokens(x: torch.Tensor) -> torch.Tensor:
        """(N, C, T, V) -> (N, V, C)."""
        if x.ndim != 4:
            raise ValueError(f"expected (N,C,T,V), got {tuple(x.shape)}")
        return x.mean(dim=2).transpose(1, 2).contiguous()

    def _score_face_tokens(
        self,
        cls_token: torch.Tensor,
        pose_tokens: torch.Tensor,
        face_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Return score per face token: (N, R)."""
        cls_n = F.normalize(cls_token, dim=-1)          # (N, 1, D)
        pose_n = F.normalize(pose_tokens, dim=-1)       # (N, Vp, D)
        face_n = F.normalize(face_tokens, dim=-1)       # (N, R, D)

        cls_score = torch.matmul(face_n, cls_n.transpose(1, 2)).squeeze(-1)  # (N, R)
        pose_score = torch.matmul(face_n, pose_n.transpose(1, 2)).amax(dim=-1)  # (N, R)
        return self.kappa * cls_score + (1.0 - self.kappa) * pose_score

    def _select_and_merge(self, scores: torch.Tensor, face_tokens: torch.Tensor):
        n, r, d = face_tokens.shape
        keep = max(1, min(r, int(math.ceil(r * self.keep_ratio))))
        top_idx = scores.topk(k=keep, dim=1, largest=True, sorted=False).indices  # (N, keep)
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, d)
        selected = torch.gather(face_tokens, dim=1, index=gather_idx)

        if keep == r:
            merged = selected.mean(dim=1, keepdim=True)
            return selected, self.merge_proj(merged)

        keep_mask = torch.zeros(n, r, device=face_tokens.device, dtype=torch.bool)
        keep_mask.scatter_(1, top_idx, True)
        drop_mask = ~keep_mask
        denom = drop_mask.sum(dim=1, keepdim=True).clamp_min(1).to(face_tokens.dtype)
        merged = (face_tokens * drop_mask.unsqueeze(-1).to(face_tokens.dtype)).sum(dim=1, keepdim=True) / denom.unsqueeze(-1)
        return selected, self.merge_proj(merged)

    def forward(self, *, shared, pose_feat, face_feat, x_occ=None):
        del x_occ  # Deliberately ignored: this comparator is not reliability/OCC-conditioned.
        cls = self.shared_norm(shared).unsqueeze(1)  # (N, 1, D)
        pose_tokens = self.pose_proj(self._pool_temporal_tokens(pose_feat))
        face_tokens = self.face_proj(self._pool_temporal_tokens(face_feat))

        scores = self._score_face_tokens(cls, pose_tokens, face_tokens)
        selected, merged = self._select_and_merge(scores, face_tokens)
        tokens = torch.cat([cls, pose_tokens, selected, merged], dim=1)
        encoded = self.encoder(tokens)

        # Combine the updated class token with selected/merged evidence.
        z = encoded[:, 0] + 0.5 * encoded[:, 1:].mean(dim=1)
        z = self.out_norm(z)

        if not self.task_adapter_enabled:
            return {name: z for name in TASK_NAMES}
        return {name: z + self.task_adapters[name](z) for name in TASK_NAMES}
