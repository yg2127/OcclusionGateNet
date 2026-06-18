"""DMD-paper-inspired multi-task baseline adapted to the current DMD label space.

The original DMD paper evaluates a lightweight driver-behaviour recognizer on
the dBehaviourMD subset with several fusion strategies:

- single stream
- early fusion
- late fusion
- score fusion

This file adapts that comparison direction to the current project setup:

- input windows are the same preprocessed DMD pose / face / optional head-pose
  tensors used by the existing multitask pipeline
- outputs follow the current 4-head DMD labels:
  action / gaze / hands / talk

The goal is not a byte-for-byte reproduction of the ECCV 2020 demo system.
Instead, it keeps the original paper's *fusion-study structure* while making it
directly comparable against the current multi-task models on the same split.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _normalize_score_weights(
    body_weight: float,
    face_weight: float,
    head_pose_weight: float,
    use_head_pose: bool,
) -> torch.Tensor:
    weights = torch.tensor(
        [
            float(body_weight),
            float(face_weight),
            float(head_pose_weight if use_head_pose else 0.0),
        ],
        dtype=torch.float32,
    )
    if not use_head_pose:
        weights = weights[:2]
    total = float(weights.sum().item())
    if total <= 1e-8:
        weights = torch.ones_like(weights) / float(weights.numel())
    else:
        weights = weights / total
    return weights


class _PerFrameProjector(nn.Module):
    """Flatten (C, T, V) per frame, project, and summarize with a BiGRU."""

    def __init__(self, in_channels: int, num_nodes: int, proj_dim: int, stream_dim: int, dropout: float):
        super().__init__()
        self.in_dim = int(in_channels) * int(num_nodes)
        self.input_norm = nn.LayerNorm(self.in_dim)
        self.project = nn.Sequential(
            nn.Linear(self.in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        hidden_size = max(32, int(stream_dim) // 2)
        self.gru = nn.GRU(
            input_size=proj_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.output_norm = nn.LayerNorm(hidden_size * 2)

    def flatten_temporal(self, x: torch.Tensor) -> torch.Tensor:
        n, c, t, v = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(n, t, c * v)

    def project_sequence(self, x: torch.Tensor) -> torch.Tensor:
        seq = self.flatten_temporal(x)
        return self.project(self.input_norm(seq))

    def encode_projected_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(seq)
        return self.output_norm(out.mean(dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode_projected_sequence(self.project_sequence(x))


class _TaskHeads(nn.Module):
    def __init__(
        self,
        in_dim: int,
        dropout: float,
        num_action: int,
        num_gaze: int,
        num_hands: int,
        num_talk: int,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.action_head = nn.Linear(in_dim, num_action)
        self.gaze_head = nn.Linear(in_dim, num_gaze)
        self.hands_head = nn.Linear(in_dim, num_hands)
        self.talk_head = nn.Linear(in_dim, num_talk)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.dropout(x)
        return {
            "action": self.action_head(z),
            "gaze": self.gaze_head(z),
            "hands": self.hands_head(z),
            "talk": self.talk_head(z),
        }


class DMDOriginalMultitaskClassifier(nn.Module):
    """Fusion-study baseline aligned with the DMD paper's comparison protocol.

    Supported fusion modes:
    - `single_body`
    - `single_face`
    - `single_head_pose` (requires `use_head_pose=True`)
    - `early_fusion`
    - `late_fusion`
    - `score_fusion`
    """

    VALID_FUSION_KINDS = {
        "single_body",
        "single_face",
        "single_head_pose",
        "early_fusion",
        "late_fusion",
        "score_fusion",
    }

    def __init__(
        self,
        pose_in_channels: int,
        face_in_channels: int,
        num_pose_joints: int = 17,
        num_face_regions: int = 478,
        proj_dim: int = 128,
        stream_dim: int = 256,
        fusion_dim: int = 512,
        dropout: float = 0.3,
        num_action: int = 11,
        num_gaze: int = 9,
        num_hands: int = 4,
        num_talk: int = 2,
        fusion_kind: str = "late_fusion",
        use_head_pose: bool = False,
        head_pose_in_channels: int = 2,
        num_head_pose_axes: int = 3,
        score_fusion_body_weight: float = 1.0,
        score_fusion_face_weight: float = 1.0,
        score_fusion_head_pose_weight: float = 0.5,
    ):
        super().__init__()
        if fusion_kind not in self.VALID_FUSION_KINDS:
            raise ValueError(f"Unsupported fusion_kind={fusion_kind!r}")
        if fusion_kind == "single_head_pose" and not use_head_pose:
            raise ValueError("single_head_pose requires use_head_pose=True")

        self.fusion_kind = fusion_kind
        self.use_head_pose = use_head_pose
        self.pose_stream = _PerFrameProjector(
            in_channels=pose_in_channels,
            num_nodes=num_pose_joints,
            proj_dim=proj_dim,
            stream_dim=stream_dim,
            dropout=dropout,
        )
        self.face_stream = _PerFrameProjector(
            in_channels=face_in_channels,
            num_nodes=num_face_regions,
            proj_dim=proj_dim,
            stream_dim=stream_dim,
            dropout=dropout,
        )
        if use_head_pose:
            self.head_pose_stream = _PerFrameProjector(
                in_channels=head_pose_in_channels,
                num_nodes=num_head_pose_axes,
                proj_dim=proj_dim,
                stream_dim=stream_dim,
                dropout=dropout,
            )

        active_streams = 2 + int(use_head_pose)
        self.early_temporal = nn.GRU(
            input_size=proj_dim * active_streams,
            hidden_size=max(32, stream_dim // 2),
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.early_norm = nn.LayerNorm(stream_dim)

        self.late_fusion = nn.Sequential(
            nn.Linear(stream_dim * active_streams, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, stream_dim),
            nn.LayerNorm(stream_dim),
            nn.GELU(),
        )
        self.shared_heads = _TaskHeads(
            in_dim=stream_dim,
            dropout=dropout,
            num_action=num_action,
            num_gaze=num_gaze,
            num_hands=num_hands,
            num_talk=num_talk,
        )

        self.body_heads = _TaskHeads(stream_dim, dropout, num_action, num_gaze, num_hands, num_talk)
        self.face_heads = _TaskHeads(stream_dim, dropout, num_action, num_gaze, num_hands, num_talk)
        if use_head_pose:
            self.head_pose_heads = _TaskHeads(stream_dim, dropout, num_action, num_gaze, num_hands, num_talk)

        score_weights = _normalize_score_weights(
            body_weight=score_fusion_body_weight,
            face_weight=score_fusion_face_weight,
            head_pose_weight=score_fusion_head_pose_weight,
            use_head_pose=use_head_pose,
        )
        self.register_buffer("score_fusion_weights", score_weights, persistent=True)

    def _make_head_pose_input(self, x_body: torch.Tensor, x_head_pose: Optional[torch.Tensor]) -> torch.Tensor:
        if x_head_pose is not None:
            return x_head_pose
        n, _, t, _ = x_body.shape
        return x_body.new_zeros((n, 2, t, 3))

    def _collect_stream_features(
        self,
        x_body: torch.Tensor,
        x_face: torch.Tensor,
        x_head_pose: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        features = {
            "body": self.pose_stream(x_body),
            "face": self.face_stream(x_face),
        }
        if self.use_head_pose:
            hp = self._make_head_pose_input(x_body, x_head_pose)
            features["head_pose"] = self.head_pose_stream(hp)
        return features

    def _collect_projected_sequences(
        self,
        x_body: torch.Tensor,
        x_face: torch.Tensor,
        x_head_pose: Optional[torch.Tensor],
    ) -> list[torch.Tensor]:
        sequences = [
            self.pose_stream.project_sequence(x_body),
            self.face_stream.project_sequence(x_face),
        ]
        if self.use_head_pose:
            hp = self._make_head_pose_input(x_body, x_head_pose)
            sequences.append(self.head_pose_stream.project_sequence(hp))
        return sequences

    def _score_fuse(self, stream_logits: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        weights = self.score_fusion_weights
        out: dict[str, torch.Tensor] = {}
        for head in ("action", "gaze", "hands", "talk"):
            fused = None
            for idx, logits in enumerate(stream_logits):
                term = logits[head] * weights[idx]
                fused = term if fused is None else fused + term
            out[head] = fused
        return out

    def forward(
        self,
        x_body: torch.Tensor,
        x_face: torch.Tensor,
        x_head_pose: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if self.fusion_kind == "single_body":
            return self.shared_heads(self.pose_stream(x_body))
        if self.fusion_kind == "single_face":
            return self.shared_heads(self.face_stream(x_face))
        if self.fusion_kind == "single_head_pose":
            hp = self._make_head_pose_input(x_body, x_head_pose)
            return self.shared_heads(self.head_pose_stream(hp))

        if self.fusion_kind == "early_fusion":
            projected = self._collect_projected_sequences(x_body, x_face, x_head_pose)
            fused_seq = torch.cat(projected, dim=-1)
            out, _ = self.early_temporal(fused_seq)
            pooled = self.early_norm(out.mean(dim=1))
            return self.shared_heads(pooled)

        features = self._collect_stream_features(x_body, x_face, x_head_pose)
        if self.fusion_kind == "late_fusion":
            ordered = [features["body"], features["face"]]
            if self.use_head_pose:
                ordered.append(features["head_pose"])
            fused = self.late_fusion(torch.cat(ordered, dim=1))
            return self.shared_heads(fused)

        if self.fusion_kind == "score_fusion":
            logits = [
                self.body_heads(features["body"]),
                self.face_heads(features["face"]),
            ]
            if self.use_head_pose:
                logits.append(self.head_pose_heads(features["head_pose"]))
            return self._score_fuse(logits)

        raise RuntimeError(f"Unhandled fusion_kind={self.fusion_kind!r}")
