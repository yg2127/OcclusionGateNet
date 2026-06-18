from __future__ import annotations

from .skateformer import SkateFormerFaceFusion, SkateFormerFusion
from .task_feature_fusion import IdentityTaskFusion

SUPPORTED_FUSION_KINDS = (
    "skateformer",
    "skateformer_skeleton",
    "skateformer_face",
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
    fusion_cfg = fusion_cfg or {"kind": "skateformer_face"}
    kind = fusion_cfg.get("kind", "skateformer_face")

    if kind in ("identity", "concat"):
        return IdentityTaskFusion(fused_channels=fused_channels)

    common = dict(
        fused_channels=fused_channels,
        pose_channels=pose_channels,
        face_channels=face_channels,
        num_joints=int(fusion_cfg.get("num_joints", 17)),
        num_layers=int(fusion_cfg.get("num_layers", 2)),
        local_size=int(fusion_cfg.get("local_size", 4)),
        num_heads_per_type=int(fusion_cfg.get("num_heads_per_type", 1)),
        kernel_size=int(fusion_cfg.get("kernel_size", 7)),
        dropout=float(fusion_cfg.get("dropout", 0.1)),
        task_adapter=bool(fusion_cfg.get("task_adapter", True)),
    )

    if kind in ("skateformer", "skateformer_skeleton"):
        return SkateFormerFusion(**common)

    if kind == "skateformer_face":
        return SkateFormerFaceFusion(**common)

    raise ValueError(
        f"Unknown fusion kind={kind!r}. Supported: " + ", ".join(SUPPORTED_FUSION_KINDS)
    )
