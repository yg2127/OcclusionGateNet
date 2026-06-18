"""Multi-task DMS classifier with swappable task-fusion modules.

Backbone flow is fixed:
    body/pose -> PoseBranch
    face      -> optional FaceRegionPool -> FaceBranch
    pose+face joint graph -> TGC post blocks -> temporal -> shared feature

Experiment flow is swappable:
    shared/pose_feat/face_feat/x_occ -> src.models.fusion.* -> task features

To test a new fusion method, add/register a fusion module. The training loop and
this classifier should not need structural changes.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.models.backbones.face_branch import FaceBranch
from src.models.backbones.face_region_pool import FaceRegionPool
from src.models.backbones.pose_branch import PoseBranch, TGCBlock
from src.models.fusion.concat_joint import ConcatJointFusion, build_fused_adjacency
from src.models.fusion.factory import build_task_fusion
from src.models.temporal.tsam import build_temporal


class MultitaskClassifier(nn.Module):
    """Gaze-only classifier using the same pose+face shared encoder."""

    def __init__(
        self,
        pose_in_channels: int,
        face_in_channels: int,
        pose_A: np.ndarray,
        num_pose_joints: int = 17,
        num_face_regions: int = 478,
        pose_mid_channels: int = 128,
        face_mid_channels: int = 128,
        fused_channels: int = 256,
        num_action: int = 11,
        num_gaze: int = 10,
        num_hands: int = 4,
        num_talk: int = 2,
        temporal_kind: str = "identity",
        temporal_num_heads: int = 4,
        temporal_dropout: float = 0.1,
        temporal_max_len: int = 8192,
        dropout_backbone: float = 0.1,
        dropout_head: float = 0.3,
        fusion_cfg: dict | None = None,
        occ_cfg: dict | None = None,
        face_encoder: str = "full",
        face_region_scheme: str = "dms_7",
        face_region_reduce: str = "mean",
    ):
        super().__init__()

        if pose_mid_channels != face_mid_channels:
            raise ValueError(
                "Current fusion modules assume pose_mid_channels == face_mid_channels. "
                f"Got pose={pose_mid_channels}, face={face_mid_channels}."
            )

        occ_cfg = occ_cfg or {}
        self.occ_enabled = bool(occ_cfg.get("enabled", False))
        self.occ_dim = int(occ_cfg.get("dim", 5)) if self.occ_enabled else 0
        self.face_encoder = face_encoder
        self.num_face_regions = int(num_face_regions)

        if face_encoder == "full":
            self.face_region_pool = nn.Identity()
        elif face_encoder == "region_pool":
            self.face_region_pool = FaceRegionPool(
                scheme=face_region_scheme,
                reduce=face_region_reduce,
            )
        else:
            raise ValueError(f"Unknown face_encoder={face_encoder!r}. Use 'full' or 'region_pool'.")

        self.pose_branch = PoseBranch(
            in_channels=pose_in_channels,
            mid_channels=pose_mid_channels,
            A=pose_A,
            num_joints=num_pose_joints,
            dropout=dropout_backbone,
        )
        self.face_branch = FaceBranch(
            in_channels=face_in_channels,
            mid_channels=face_mid_channels,
            num_regions=num_face_regions,
            dropout=dropout_backbone,
        )

        self.joint_fusion = ConcatJointFusion()
        fused_A = build_fused_adjacency(pose_A, num_face_regions=num_face_regions)

        self.post1 = TGCBlock(
            pose_mid_channels,
            fused_channels,
            fused_A,
            stride=2,
            dropout=dropout_backbone,
        )
        self.post2 = TGCBlock(
            fused_channels,
            fused_channels,
            fused_A,
            stride=1,
            dropout=dropout_backbone,
        )
        self.temporal = build_temporal(
            kind=temporal_kind,
            channels=fused_channels,
            num_heads=temporal_num_heads,
            dropout=temporal_dropout,
            max_len=temporal_max_len,
        )
        self.head_dropout = nn.Dropout(dropout_head)

        self.task_fusion = build_task_fusion(
            fusion_cfg,
            fused_channels=fused_channels,
            pose_channels=pose_mid_channels,
            face_channels=face_mid_channels,
            occ_dim=self.occ_dim,
        )
        head_dim = int(self.task_fusion.out_dim)

        # Gaze-only experiment: action/hands/talk heads are intentionally removed.
        # The shared pose+face encoder is unchanged; only the gaze head receives loss and is evaluated.
        del num_action, num_hands, num_talk
        self.gaze_head = nn.Linear(head_dim, num_gaze)

    def _encode_shared(self, pose_feat: torch.Tensor, face_feat: torch.Tensor) -> torch.Tensor:
        x = self.joint_fusion(pose_feat, face_feat)
        x = self.post1(x)
        x = self.post2(x)
        x = self.temporal(x)
        x = x.mean(dim=-1).mean(dim=-1)
        return self.head_dropout(x)

    def forward(
        self,
        x_body: torch.Tensor,
        x_face: torch.Tensor,
        x_occ: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.occ_dim > 0 and x_occ is not None:
            x_occ = x_occ.to(device=x_body.device, dtype=x_body.dtype)

        pose_feat = self.pose_branch(x_body)
        x_face = self.face_region_pool(x_face)
        face_feat = self.face_branch(x_face)
        shared = self._encode_shared(pose_feat, face_feat)

        task_features = self.task_fusion(
            shared=shared,
            pose_feat=pose_feat,
            face_feat=face_feat,
            x_occ=x_occ,
        )

        return {
            "gaze": self.gaze_head(task_features["gaze"]),
        }
