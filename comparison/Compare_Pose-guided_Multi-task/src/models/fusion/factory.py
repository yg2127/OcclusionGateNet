from __future__ import annotations

from .pose_guided_token_selection import PoseGuidedTokenSelectionFusion
from .task_feature_fusion import IdentityTaskFusion

SUPPORTED_FUSION_KINDS = (
    "pose_guided_token_selection",
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
    fusion_cfg = fusion_cfg or {"kind": "pose_guided_token_selection"}
    kind = fusion_cfg.get("kind", "pose_guided_token_selection")

    # Keep identity/concat only as a safe fallback for debugging. The intended
    # comparator is pose_guided_token_selection.
    if kind in ("identity", "concat"):
        return IdentityTaskFusion(fused_channels=fused_channels)

    if kind == "pose_guided_token_selection":
        return PoseGuidedTokenSelectionFusion(
            fused_channels=fused_channels,
            pose_channels=pose_channels,
            face_channels=face_channels,
            keep_ratio=float(fusion_cfg.get("keep_ratio", 0.6)),
            kappa=float(fusion_cfg.get("kappa", 0.5)),
            num_layers=int(fusion_cfg.get("num_layers", 1)),
            num_heads=int(fusion_cfg.get("num_heads", 4)),
            ff_mult=int(fusion_cfg.get("ff_mult", 2)),
            dropout=float(fusion_cfg.get("dropout", 0.1)),
            task_adapter=bool(fusion_cfg.get("task_adapter", True)),
        )

    raise ValueError(
        f"Unknown fusion kind={kind!r}. Supported: " + ", ".join(SUPPORTED_FUSION_KINDS)
    )
