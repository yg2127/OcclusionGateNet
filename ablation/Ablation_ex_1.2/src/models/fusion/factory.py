from __future__ import annotations

from .task_feature_fusion import IdentityTaskFusion

SUPPORTED_FUSION_KINDS = ("concat", "identity")


def build_task_fusion(
    fusion_cfg: dict | None,
    *,
    fused_channels: int,
    pose_channels: int,
    face_channels: int,
    occ_dim: int,
):
    del pose_channels, face_channels, occ_dim
    fusion_cfg = fusion_cfg or {"kind": "concat"}
    kind = fusion_cfg.get("kind", "concat")
    if kind in ("concat", "identity"):
        return IdentityTaskFusion(fused_channels=fused_channels)
    raise ValueError(f"This minimal V1-style package only supports fusion.kind='concat'. Got {kind!r}.")
