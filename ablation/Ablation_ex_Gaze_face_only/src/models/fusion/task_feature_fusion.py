"""Task-level fusion modules for DMS multitask classification.

This file is the main extension point for new fusion experiments.
The backbone still produces:
    - shared fused feature: (N, fused_channels)
    - pose branch feature: (N, C, T, V_pose)
    - face branch feature: (N, C, T, R)

Each module returns task-specific feature vectors consumed by the four heads.
To add a new method, implement a class here or in a new file, then register it
in `src.models.fusion.factory.build_task_fusion`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

TASK_NAMES = ("action", "gaze", "hands", "talk")


class TaskFeatureFusion(nn.Module, ABC):
    """Common interface for swappable task-fusion modules."""

    out_dim: int

    @abstractmethod
    def forward(
        self,
        *,
        shared: torch.Tensor,
        pose_feat: torch.Tensor,
        face_feat: torch.Tensor,
        x_occ: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class IdentityTaskFusion(TaskFeatureFusion):
    """Baseline: all heads receive the same shared fused representation."""

    def __init__(self, fused_channels: int):
        super().__init__()
        self.out_dim = int(fused_channels)

    def forward(self, *, shared, pose_feat, face_feat, x_occ=None):
        del pose_feat, face_feat, x_occ
        return {name: shared for name in TASK_NAMES}


class ConcatConditionTaskFusion(TaskFeatureFusion):
    """Simple occlusion conditioning: concat(shared_feature, MLP(occ_vector))."""

    def __init__(
        self,
        *,
        fused_channels: int,
        occ_dim: int,
        occ_hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        if occ_dim <= 0:
            raise ValueError("concat_condition requires occ.enabled=true and occ.dim > 0")

        self.occ_dim = int(occ_dim)
        self.out_dim = int(fused_channels) + int(occ_hidden_dim)
        self.occ_mlp = nn.Sequential(
            nn.LayerNorm(self.occ_dim),
            nn.Linear(self.occ_dim, occ_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(occ_hidden_dim, occ_hidden_dim),
            nn.GELU(),
        )

    def forward(self, *, shared, pose_feat, face_feat, x_occ=None):
        del pose_feat, face_feat
        if x_occ is None:
            raise ValueError("concat_condition requires x_occ but received None")
        occ_emb = self.occ_mlp(x_occ)
        z = torch.cat([shared, occ_emb], dim=-1)
        return {name: z for name in TASK_NAMES}


class TaskScalarGate(nn.Module):
    """Task-specific scalar gate with optional occlusion conditioning."""

    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        init_bias: float = 0.0,
        cond_dim: int = 0,
    ):
        super().__init__()
        self.cond_dim = int(cond_dim)
        total_in_dim = int(in_dim) + self.cond_dim

        self.net = nn.Sequential(
            nn.LayerNorm(total_in_dim),
            nn.Linear(total_in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, init_bias)

    def forward(self, pose_vec: torch.Tensor, face_vec: torch.Tensor, x_occ=None):
        parts = [pose_vec, face_vec]
        if self.cond_dim > 0:
            if x_occ is None:
                x_occ = torch.zeros(
                    pose_vec.shape[0],
                    self.cond_dim,
                    device=pose_vec.device,
                    dtype=pose_vec.dtype,
                )
            parts.append(x_occ)
        return torch.sigmoid(self.net(torch.cat(parts, dim=-1)))


class TaskGatedLateFusion(TaskFeatureFusion):
    """Late task-specific pose/face scalar gate.

    Each head receives:
        shared + scale * (g_task * pose_proj + (1-g_task) * face_proj)
    """

    def __init__(
        self,
        *,
        fused_channels: int,
        pose_channels: int,
        face_channels: int,
        occ_dim: int = 0,
        gate_hidden_channels: int = 128,
        gate_dropout: float = 0.2,
        gate_feature_scale: float = 0.25,
        init_bias: dict | None = None,
    ):
        super().__init__()
        init_bias = init_bias or {}
        self.out_dim = int(fused_channels)
        self.gate_feature_scale = float(gate_feature_scale)

        gate_in_dim = int(pose_channels) + int(face_channels)
        self.gates = nn.ModuleDict(
            {
                name: TaskScalarGate(
                    in_dim=gate_in_dim,
                    hidden_dim=gate_hidden_channels,
                    dropout=gate_dropout,
                    init_bias=float(init_bias.get(name, 0.0)),
                    cond_dim=occ_dim,
                )
                for name in TASK_NAMES
            }
        )

        self.pose_proj = nn.Sequential(
            nn.LayerNorm(pose_channels),
            nn.Linear(pose_channels, fused_channels),
            nn.GELU(),
            nn.Dropout(gate_dropout),
        )
        self.face_proj = nn.Sequential(
            nn.LayerNorm(face_channels),
            nn.Linear(face_channels, fused_channels),
            nn.GELU(),
            nn.Dropout(gate_dropout),
        )

    @staticmethod
    def _pool_branch(x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=-1).mean(dim=-1)

    def forward(self, *, shared, pose_feat, face_feat, x_occ=None):
        pose_vec = self._pool_branch(pose_feat)
        face_vec = self._pool_branch(face_feat)
        pose_proj = self.pose_proj(pose_vec)
        face_proj = self.face_proj(face_vec)

        out = {}
        for name in TASK_NAMES:
            g = self.gates[name](pose_vec, face_vec, x_occ=x_occ)
            pf = g * pose_proj + (1.0 - g) * face_proj
            out[name] = shared + self.gate_feature_scale * pf
        return out


class TaskGatedLateNoGazeOccFusion(TaskFeatureFusion):
    """Task gated late fusion where OCC is disabled only for the gaze gate.

    action / hands / talk gates receive x_occ as reliability condition.
    gaze gate receives only pose_vec and face_vec, so gaze does not directly
    depend on the OCC vector.
    """

    def __init__(
        self,
        *,
        fused_channels: int,
        pose_channels: int,
        face_channels: int,
        occ_dim: int = 0,
        gate_hidden_channels: int = 128,
        gate_dropout: float = 0.2,
        gate_feature_scale: float = 0.25,
        init_bias: dict | None = None,
    ):
        super().__init__()
        init_bias = init_bias or {}
        self.out_dim = int(fused_channels)
        self.gate_feature_scale = float(gate_feature_scale)
        self.occ_dim = int(occ_dim)

        gate_in_dim = int(pose_channels) + int(face_channels)
        self.gates = nn.ModuleDict()
        for name in TASK_NAMES:
            cond_dim = 0 if name == "gaze" else self.occ_dim
            self.gates[name] = TaskScalarGate(
                in_dim=gate_in_dim,
                hidden_dim=gate_hidden_channels,
                dropout=gate_dropout,
                init_bias=float(init_bias.get(name, 0.0)),
                cond_dim=cond_dim,
            )

        self.pose_proj = nn.Sequential(
            nn.LayerNorm(pose_channels),
            nn.Linear(pose_channels, fused_channels),
            nn.GELU(),
            nn.Dropout(gate_dropout),
        )
        self.face_proj = nn.Sequential(
            nn.LayerNorm(face_channels),
            nn.Linear(face_channels, fused_channels),
            nn.GELU(),
            nn.Dropout(gate_dropout),
        )

    @staticmethod
    def _pool_branch(x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=-1).mean(dim=-1)

    def forward(self, *, shared, pose_feat, face_feat, x_occ=None):
        pose_vec = self._pool_branch(pose_feat)
        face_vec = self._pool_branch(face_feat)
        pose_proj = self.pose_proj(pose_vec)
        face_proj = self.face_proj(face_vec)

        out = {}
        for name in TASK_NAMES:
            # Gaze gate was constructed with cond_dim=0, so this x_occ is ignored there.
            g = self.gates[name](pose_vec, face_vec, x_occ=x_occ)
            pf = g * pose_proj + (1.0 - g) * face_proj
            out[name] = shared + self.gate_feature_scale * pf
        return out


class TaskGatedLateGazeFaceOnlyFusion(TaskFeatureFusion):
    """Task-gated late fusion with a strictly face-only gaze branch.

    Purpose:
        Validate whether gaze can be solved from FaceMesh/face-region features alone.

    action / hands / talk:
        Same as TaskGatedLateFusion: task-specific pose/face scalar gate, optionally OCC-conditioned.

    gaze:
        Uses only pooled face_feat -> face_proj. It does not use pose_feat, shared pose+face
        representation, OCC vector, or scalar gate.

    This isolates the gaze head from pose and OCC so the resulting gaze metric directly tests
    the quality of the face representation under clean/masked conditions.
    """

    def __init__(
        self,
        *,
        fused_channels: int,
        pose_channels: int,
        face_channels: int,
        occ_dim: int = 0,
        gate_hidden_channels: int = 128,
        gate_dropout: float = 0.2,
        gate_feature_scale: float = 0.25,
        init_bias: dict | None = None,
    ):
        super().__init__()
        init_bias = init_bias or {}
        self.out_dim = int(fused_channels)
        self.gate_feature_scale = float(gate_feature_scale)
        self.occ_dim = int(occ_dim)

        gate_in_dim = int(pose_channels) + int(face_channels)
        # Gates are only used for non-gaze heads. Keep gaze out of the gate path.
        self.gates = nn.ModuleDict(
            {
                name: TaskScalarGate(
                    in_dim=gate_in_dim,
                    hidden_dim=gate_hidden_channels,
                    dropout=gate_dropout,
                    init_bias=float(init_bias.get(name, 0.0)),
                    cond_dim=self.occ_dim,
                )
                for name in ("action", "hands", "talk")
            }
        )

        self.pose_proj = nn.Sequential(
            nn.LayerNorm(pose_channels),
            nn.Linear(pose_channels, fused_channels),
            nn.GELU(),
            nn.Dropout(gate_dropout),
        )
        self.face_proj = nn.Sequential(
            nn.LayerNorm(face_channels),
            nn.Linear(face_channels, fused_channels),
            nn.GELU(),
            nn.Dropout(gate_dropout),
        )

    @staticmethod
    def _pool_branch(x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=-1).mean(dim=-1)

    def forward(self, *, shared, pose_feat, face_feat, x_occ=None):
        pose_vec = self._pool_branch(pose_feat)
        face_vec = self._pool_branch(face_feat)
        pose_proj = self.pose_proj(pose_vec)
        face_proj = self.face_proj(face_vec)

        out = {}
        for name in ("action", "hands", "talk"):
            g = self.gates[name](pose_vec, face_vec, x_occ=x_occ)
            pf = g * pose_proj + (1.0 - g) * face_proj
            out[name] = shared + self.gate_feature_scale * pf

        # Strict face-only gaze feature: no shared feature, no pose, no OCC, no gate.
        out["gaze"] = face_proj
        return out


class FaceRegionRelationEncoder(nn.Module):
    """Small transformer encoder over semantic face-region tokens."""

    def __init__(
        self,
        *,
        channels: int,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        ff_mult: int = 2,
    ):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels must be divisible by num_heads: channels={channels}, heads={num_heads}"
            )
        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=channels * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected region tokens (N, R, C), got {tuple(x.shape)}")
        return self.encoder(x)


class TaskRegionGate(nn.Module):
    """Task-specific gate over relation-aware face region tokens."""

    def __init__(
        self,
        *,
        channels: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        init_bias: float = 0.0,
        cond_dim: int = 0,
    ):
        super().__init__()
        self.cond_dim = int(cond_dim)
        in_dim = int(channels) + self.cond_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, init_bias)

    def forward(self, region_tokens: torch.Tensor, x_occ=None) -> torch.Tensor:
        if self.cond_dim > 0:
            n, r, _ = region_tokens.shape
            if x_occ is None:
                x_occ = torch.zeros(
                    n,
                    self.cond_dim,
                    device=region_tokens.device,
                    dtype=region_tokens.dtype,
                )
            occ_tokens = x_occ[:, None, :].expand(n, r, self.cond_dim)
            region_tokens = torch.cat([region_tokens, occ_tokens], dim=-1)
        return torch.sigmoid(self.net(region_tokens))


class TaskRegionGatedLateFusion(TaskFeatureFusion):
    """Relation-aware task-specific region gated late fusion.

    This corresponds to the current 3-A style baseline:
        Transformer(face_region_tokens) -> task region gate -> weighted region pool
        -> shared residual correction.
    """

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
    ):
        super().__init__()
        init_bias = init_bias or {}
        self.out_dim = int(fused_channels)
        self.gate_feature_scale = float(gate_feature_scale)
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
                    cond_dim=occ_dim,
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
        if region_tokens.shape[:2] != gate.shape[:2]:
            raise ValueError(f"region/gate mismatch: {region_tokens.shape} vs {gate.shape}")
        weighted = region_tokens * gate
        denom = gate.sum(dim=1).clamp_min(1e-6)
        return weighted.sum(dim=1) / denom

    def forward(self, *, shared, pose_feat, face_feat, x_occ=None):
        del pose_feat
        tokens = self._face_region_tokens(face_feat)
        tokens = self.relation(tokens)
        out = {}
        for name in TASK_NAMES:
            gate = self.gates[name](tokens, x_occ=x_occ)
            face_vec = self._weighted_region_pool(tokens, gate)
            face_vec = self.face_proj(face_vec)
            out[name] = shared + self.gate_feature_scale * face_vec
        return out


class TaskRegionGatedLateNoGazeOccFusion(TaskFeatureFusion):
    """Task-region gated fusion where OCC is disabled only for the gaze region gate.

    action / hands / talk region gates receive x_occ.
    gaze region gate receives only face region tokens.
    """

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
    ):
        super().__init__()
        init_bias = init_bias or {}
        self.out_dim = int(fused_channels)
        self.gate_feature_scale = float(gate_feature_scale)
        self.occ_dim = int(occ_dim)
        self.relation = FaceRegionRelationEncoder(
            channels=face_channels,
            num_heads=region_num_heads,
            num_layers=region_num_layers,
            dropout=region_dropout,
            ff_mult=region_ff_mult,
        )
        self.gates = nn.ModuleDict()
        for name in TASK_NAMES:
            cond_dim = 0 if name == "gaze" else self.occ_dim
            self.gates[name] = TaskRegionGate(
                channels=face_channels,
                hidden_dim=gate_hidden_channels,
                dropout=gate_dropout,
                init_bias=float(init_bias.get(name, 0.0)),
                cond_dim=cond_dim,
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
        if region_tokens.shape[:2] != gate.shape[:2]:
            raise ValueError(f"region/gate mismatch: {region_tokens.shape} vs {gate.shape}")
        weighted = region_tokens * gate
        denom = gate.sum(dim=1).clamp_min(1e-6)
        return weighted.sum(dim=1) / denom

    def forward(self, *, shared, pose_feat, face_feat, x_occ=None):
        del pose_feat
        tokens = self._face_region_tokens(face_feat)
        tokens = self.relation(tokens)
        out = {}
        for name in TASK_NAMES:
            gate = self.gates[name](tokens, x_occ=x_occ)
            face_vec = self._weighted_region_pool(tokens, gate)
            face_vec = self.face_proj(face_vec)
            out[name] = shared + self.gate_feature_scale * face_vec
        return out
