from __future__ import annotations

from .spatiotemporal_decoupling import SpatiotemporalDecouplingFaceFusion, SpatiotemporalDecouplingFusion
from .task_feature_fusion import IdentityTaskFusion

SUPPORTED_FUSION_KINDS = (
    "spatiotemporal_decoupling",
    "sda_tr_skeleton",
    "spatiotemporal_decoupling_face",
    "sda_tr_pose_face",
    "identity",
    "concat",
)


def build_task_fusion(
    fusion_cfg: dict | None,
    *,
    fused_channels: int,
    pose_channels: int,
    face_channels: int,
    occ_dim: int,
):
    """Build only the fusion modules needed for this comparator package."""
    del occ_dim
    fusion_cfg = fusion_cfg or {"kind": "spatiotemporal_decoupling"}
    kind = fusion_cfg.get("kind", "spatiotemporal_decoupling")

    if kind in ("identity", "concat"):
        return IdentityTaskFusion(fused_channels=fused_channels)

    if kind in ("spatiotemporal_decoupling", "sda_tr_skeleton"):
        return SpatiotemporalDecouplingFusion(
            fused_channels=fused_channels,
            pose_channels=pose_channels,
            face_channels=face_channels,
            num_joints=int(fusion_cfg.get("num_joints", 17)),
            num_layers=int(fusion_cfg.get("num_layers", 3)),
            tau=int(fusion_cfg.get("tau", 4)),
            num_heads=int(fusion_cfg.get("num_heads", 4)),
            kernel_size=int(fusion_cfg.get("kernel_size", 7)),
            dropout=float(fusion_cfg.get("dropout", 0.1)),
            task_adapter=bool(fusion_cfg.get("task_adapter", True)),
        )



    if kind in ("spatiotemporal_decoupling_face", "sda_tr_pose_face"):
        return SpatiotemporalDecouplingFaceFusion(
            fused_channels=fused_channels,
            pose_channels=pose_channels,
            face_channels=face_channels,
            num_joints=int(fusion_cfg.get("num_joints", 17)),
            num_layers=int(fusion_cfg.get("num_layers", 3)),
            tau=int(fusion_cfg.get("tau", 4)),
            num_heads=int(fusion_cfg.get("num_heads", 4)),
            kernel_size=int(fusion_cfg.get("kernel_size", 7)),
            dropout=float(fusion_cfg.get("dropout", 0.1)),
            task_adapter=bool(fusion_cfg.get("task_adapter", True)),
        )

    raise ValueError(
        f"Unknown fusion kind={kind!r}. Supported: " + ", ".join(SUPPORTED_FUSION_KINDS)
    )
