"""V5 hybrid region-scalar task fusion modules.

These modules preserve the V5 model core:
    1) face region tokens are contextualized by a region Transformer,
    2) each task learns its own face-region gate,
    3) each task learns a scalar pose-vs-face gate,
    4) the resulting task-specific feature is added to the shared fused feature.

The explicit mask variant additionally injects OCC reliability into the region gate.
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
    TaskScalarGate,
)


class TaskRegionScalarGatedLateFusion(TaskFeatureFusion):
    """V5 core fusion: task-specific region gate + task-specific scalar pose/face gate.

    Per task:
        region_tokens = Transformer(face_tokens)
        face_task_vec = weighted_pool(region_tokens, task_region_gate)
        g = task_scalar_gate(pose_vec, face_task_vec)   # g = pose weight
        task_feature = shared + scale * (g * pose_proj + (1-g) * face_proj)

    By default, this reproduces the V5 idea without forcing OCC into the gates.
    Set `region_gate_condition_occ=true` and/or `scalar_gate_condition_occ=true`
    to make the V5 hybrid gate OCC-conditioned.
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
        region_num_heads: int = 4,
        region_num_layers: int = 1,
        region_dropout: float = 0.1,
        region_ff_mult: int = 2,
        region_gate_condition_occ: bool = False,
        scalar_gate_condition_occ: bool = False,
    ):
        super().__init__()
        init_bias = init_bias or {}
        self.out_dim = int(fused_channels)
        self.occ_dim = int(occ_dim)
        self.gate_feature_scale = float(gate_feature_scale)
        self.scalar_gate_condition_occ = bool(scalar_gate_condition_occ)

        self.relation = FaceRegionRelationEncoder(
            channels=face_channels,
            num_heads=region_num_heads,
            num_layers=region_num_layers,
            dropout=region_dropout,
            ff_mult=region_ff_mult,
        )
        self.region_gates = nn.ModuleDict(
            {
                name: TaskRegionGate(
                    channels=face_channels,
                    hidden_dim=gate_hidden_channels,
                    dropout=gate_dropout,
                    init_bias=float(init_bias.get(f"region_{name}", init_bias.get(name, 0.0))),
                    cond_dim=self.occ_dim if region_gate_condition_occ else 0,
                )
                for name in TASK_NAMES
            }
        )

        gate_in_dim = int(pose_channels) + int(face_channels)
        default_scalar_bias = {
            "action": 0.0,
            "gaze": -0.7,
            "hands": 0.7,
            "talk": -0.5,
        }
        self.scalar_gates = nn.ModuleDict(
            {
                name: TaskScalarGate(
                    in_dim=gate_in_dim,
                    hidden_dim=gate_hidden_channels,
                    dropout=gate_dropout,
                    init_bias=float(init_bias.get(f"scalar_{name}", default_scalar_bias[name])),
                    cond_dim=self.occ_dim if scalar_gate_condition_occ else 0,
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

    @staticmethod
    def _face_region_tokens(face_feat: torch.Tensor) -> torch.Tensor:
        if face_feat.ndim != 4:
            raise ValueError(f"expected face feature (N, C, T, R), got {tuple(face_feat.shape)}")
        return face_feat.mean(dim=2).transpose(1, 2).contiguous()

    @staticmethod
    def _weighted_region_pool(region_tokens: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if region_tokens.shape[:2] != gate.shape[:2]:
            raise ValueError(f"region/gate mismatch: {tuple(region_tokens.shape)} vs {tuple(gate.shape)}")
        weighted = region_tokens * gate
        denom = gate.sum(dim=1).clamp_min(1e-6)
        return weighted.sum(dim=1) / denom

    def _task_face_vecs(self, tokens: torch.Tensor, x_occ: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        tokens = self.relation(tokens)
        out = {}
        for name in TASK_NAMES:
            gate = self.region_gates[name](tokens, x_occ=x_occ)
            out[name] = self._weighted_region_pool(tokens, gate)
        return out

    def forward(self, *, shared, pose_feat, face_feat, x_occ=None):
        pose_vec = self._pool_branch(pose_feat)
        region_tokens = self._face_region_tokens(face_feat)
        face_vecs = self._task_face_vecs(region_tokens, x_occ=x_occ)

        pose_proj = self.pose_proj(pose_vec)
        out = {}
        for name in TASK_NAMES:
            face_vec = face_vecs[name]
            face_proj = self.face_proj(face_vec)
            g = self.scalar_gates[name](pose_vec, face_vec, x_occ=x_occ)
            pf = g * pose_proj + (1.0 - g) * face_proj
            out[name] = shared + self.gate_feature_scale * pf
        return out


class ExplicitRegionScalarMaskGateFusion(TaskRegionScalarGatedLateFusion):
    """V5 hybrid fusion + explicit OCC reliability mask on task region gates.

    This is the V5-core equivalent of V4's `explicit_region_mask_gate`:
        region_gate = task_region_gate(tokens, x_occ)
        region_gate = region_gate * occ_to_region_reliability(x_occ)
        face_task_vec = weighted_pool(tokens, region_gate)
        scalar_gate then mixes pose and face_task_vec.
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
        region_num_heads: int = 4,
        region_num_layers: int = 1,
        region_dropout: float = 0.1,
        region_ff_mult: int = 2,
        region_gate_condition_occ: bool = True,
        scalar_gate_condition_occ: bool = True,
        region_occ_indices: Sequence[int] | None = None,
        default_visible: float = 1.0,
        mask_strength: float = 1.0,
        min_reliability: float = 0.05,
    ):
        if occ_dim <= 0:
            raise ValueError("explicit_region_scalar_mask_gate requires occ.enabled=true and occ.dim > 0")
        super().__init__(
            fused_channels=fused_channels,
            pose_channels=pose_channels,
            face_channels=face_channels,
            occ_dim=occ_dim,
            gate_hidden_channels=gate_hidden_channels,
            gate_dropout=gate_dropout,
            gate_feature_scale=gate_feature_scale,
            init_bias=init_bias,
            region_num_heads=region_num_heads,
            region_num_layers=region_num_layers,
            region_dropout=region_dropout,
            region_ff_mult=region_ff_mult,
            region_gate_condition_occ=region_gate_condition_occ,
            scalar_gate_condition_occ=scalar_gate_condition_occ,
        )
        self.region_occ_indices = None if region_occ_indices is None else [int(x) for x in region_occ_indices]
        self.default_visible = float(default_visible)
        self.mask_strength = float(mask_strength)
        self.min_reliability = float(min_reliability)

    def _task_face_vecs(self, tokens: torch.Tensor, x_occ: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
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
            gate = self.region_gates[name](tokens, x_occ=x_occ)
            gate = gate * reliability
            out[name] = self._weighted_region_pool(tokens, gate)
        return out


class TaskRegionScalarGatedLateNoGazeOccFusion(TaskFeatureFusion):
    """V5 hybrid region+scalar fusion with OCC disabled only for gaze.

    action / hands / talk:
        region gate may receive OCC and scalar gate may receive OCC.
    gaze:
        both region gate and scalar gate are built with cond_dim=0, so gaze does
        not directly use x_occ. This isolates the question: does removing OCC
        from gaze while keeping OCC for the other heads improve robustness?
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
        region_num_heads: int = 4,
        region_num_layers: int = 1,
        region_dropout: float = 0.1,
        region_ff_mult: int = 2,
        region_gate_condition_occ: bool = True,
        scalar_gate_condition_occ: bool = True,
    ):
        super().__init__()
        init_bias = init_bias or {}
        self.out_dim = int(fused_channels)
        self.occ_dim = int(occ_dim)
        self.gate_feature_scale = float(gate_feature_scale)

        self.relation = FaceRegionRelationEncoder(
            channels=face_channels,
            num_heads=region_num_heads,
            num_layers=region_num_layers,
            dropout=region_dropout,
            ff_mult=region_ff_mult,
        )

        self.region_gates = nn.ModuleDict()
        for name in TASK_NAMES:
            cond_dim = self.occ_dim if (name != "gaze" and region_gate_condition_occ) else 0
            self.region_gates[name] = TaskRegionGate(
                channels=face_channels,
                hidden_dim=gate_hidden_channels,
                dropout=gate_dropout,
                init_bias=float(init_bias.get(f"region_{name}", init_bias.get(name, 0.0))),
                cond_dim=cond_dim,
            )

        gate_in_dim = int(pose_channels) + int(face_channels)
        default_scalar_bias = {
            "action": 0.0,
            "gaze": -0.7,
            "hands": 0.7,
            "talk": -0.5,
        }
        self.scalar_gates = nn.ModuleDict()
        for name in TASK_NAMES:
            cond_dim = self.occ_dim if (name != "gaze" and scalar_gate_condition_occ) else 0
            self.scalar_gates[name] = TaskScalarGate(
                in_dim=gate_in_dim,
                hidden_dim=gate_hidden_channels,
                dropout=gate_dropout,
                init_bias=float(init_bias.get(f"scalar_{name}", default_scalar_bias[name])),
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

    @staticmethod
    def _face_region_tokens(face_feat: torch.Tensor) -> torch.Tensor:
        if face_feat.ndim != 4:
            raise ValueError(f"expected face feature (N, C, T, R), got {tuple(face_feat.shape)}")
        return face_feat.mean(dim=2).transpose(1, 2).contiguous()

    @staticmethod
    def _weighted_region_pool(region_tokens: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if region_tokens.shape[:2] != gate.shape[:2]:
            raise ValueError(f"region/gate mismatch: {tuple(region_tokens.shape)} vs {tuple(gate.shape)}")
        weighted = region_tokens * gate
        denom = gate.sum(dim=1).clamp_min(1e-6)
        return weighted.sum(dim=1) / denom

    def _task_face_vecs(self, tokens: torch.Tensor, x_occ: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        tokens = self.relation(tokens)
        out = {}
        for name in TASK_NAMES:
            gate = self.region_gates[name](tokens, x_occ=x_occ)
            out[name] = self._weighted_region_pool(tokens, gate)
        return out

    def forward(self, *, shared, pose_feat, face_feat, x_occ=None):
        pose_vec = self._pool_branch(pose_feat)
        region_tokens = self._face_region_tokens(face_feat)
        face_vecs = self._task_face_vecs(region_tokens, x_occ=x_occ)

        pose_proj = self.pose_proj(pose_vec)
        out = {}
        for name in TASK_NAMES:
            face_vec = face_vecs[name]
            face_proj = self.face_proj(face_vec)
            # Gaze scalar gate was constructed with cond_dim=0, so x_occ is ignored for gaze.
            g = self.scalar_gates[name](pose_vec, face_vec, x_occ=x_occ)
            pf = g * pose_proj + (1.0 - g) * face_proj
            out[name] = shared + self.gate_feature_scale * pf
        return out
