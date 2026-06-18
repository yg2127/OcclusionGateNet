from __future__ import annotations

from .explicit_region_mask_gate import ExplicitRegionMaskGateFusion
from .occ_attention_bias import OccAttentionBiasFusion
from .occ_token_region_transformer import OccTokenRegionTransformerFusion
from .task_region_scalar_fusion import (
    ExplicitRegionScalarMaskGateFusion,
    TaskRegionScalarGatedLateFusion,
    TaskRegionScalarGatedLateNoGazeOccFusion,
)
from .task_feature_fusion import (
    ConcatConditionTaskFusion,
    IdentityTaskFusion,
    TaskGatedLateFusion,
    TaskGatedLateNoGazeOccFusion,
    TaskGatedLateGazeFaceOnlyFusion,
    TaskRegionGatedLateFusion,
    TaskRegionGatedLateNoGazeOccFusion,
)

SUPPORTED_FUSION_KINDS = (
    "concat",
    "identity",
    "concat_condition",
    "task_gated_late",
    "task_gated_late_no_gaze_occ",
    "task_gated_late_gaze_face_only",
    "task_region_gated_late",
    "task_region_gated_late_no_gaze_occ",
    "explicit_region_mask_gate",
    "occ_token_region_transformer",
    "occ_attention_bias",
    "task_region_scalar_gated_late",
    "task_region_scalar_gated_late_no_gaze_occ",
    "explicit_region_scalar_mask_gate",
)


def _region_common_kwargs(fusion_cfg: dict) -> dict:
    return {
        "gate_hidden_channels": int(fusion_cfg.get("gate_hidden_channels", 128)),
        "gate_dropout": float(fusion_cfg.get("gate_dropout", 0.2)),
        "gate_feature_scale": float(fusion_cfg.get("gate_feature_scale", 0.25)),
        "init_bias": fusion_cfg.get("init_bias", {}),
        "region_num_heads": int(fusion_cfg.get("region_num_heads", 4)),
        "region_num_layers": int(fusion_cfg.get("region_num_layers", 1)),
        "region_dropout": float(fusion_cfg.get("region_dropout", 0.1)),
        "region_ff_mult": int(fusion_cfg.get("region_ff_mult", 2)),
    }


def _region_occ_kwargs(fusion_cfg: dict) -> dict:
    return {
        "region_occ_indices": fusion_cfg.get("region_occ_indices", None),
        "default_visible": float(fusion_cfg.get("default_visible", 1.0)),
        "mask_strength": float(fusion_cfg.get("mask_strength", 1.0)),
        "min_reliability": float(fusion_cfg.get("min_reliability", 0.05)),
    }


def build_task_fusion(
    fusion_cfg: dict | None,
    *,
    fused_channels: int,
    pose_channels: int,
    face_channels: int,
    occ_dim: int,
):
    """Build the swappable task-level fusion module.

    New fusion experiments should be registered here. The training loop and
    MultitaskClassifier do not need to change when switching methods.
    """
    fusion_cfg = fusion_cfg or {"kind": "concat"}
    kind = fusion_cfg.get("kind", "concat")

    if kind in ("concat", "identity"):
        return IdentityTaskFusion(fused_channels=fused_channels)

    if kind == "concat_condition":
        return ConcatConditionTaskFusion(
            fused_channels=fused_channels,
            occ_dim=occ_dim,
            occ_hidden_dim=int(fusion_cfg.get("occ_hidden_dim", 64)),
            dropout=float(fusion_cfg.get("occ_dropout", 0.1)),
        )

    if kind == "task_gated_late":
        return TaskGatedLateFusion(
            fused_channels=fused_channels,
            pose_channels=pose_channels,
            face_channels=face_channels,
            occ_dim=occ_dim,
            gate_hidden_channels=int(fusion_cfg.get("gate_hidden_channels", 128)),
            gate_dropout=float(fusion_cfg.get("gate_dropout", 0.2)),
            gate_feature_scale=float(fusion_cfg.get("gate_feature_scale", 0.25)),
            init_bias=fusion_cfg.get("init_bias", {}),
        )

    if kind == "task_gated_late_no_gaze_occ":
        return TaskGatedLateNoGazeOccFusion(
            fused_channels=fused_channels,
            pose_channels=pose_channels,
            face_channels=face_channels,
            occ_dim=occ_dim,
            gate_hidden_channels=int(fusion_cfg.get("gate_hidden_channels", 128)),
            gate_dropout=float(fusion_cfg.get("gate_dropout", 0.2)),
            gate_feature_scale=float(fusion_cfg.get("gate_feature_scale", 0.25)),
            init_bias=fusion_cfg.get("init_bias", {}),
        )


    if kind == "task_gated_late_gaze_face_only":
        return TaskGatedLateGazeFaceOnlyFusion(
            fused_channels=fused_channels,
            pose_channels=pose_channels,
            face_channels=face_channels,
            occ_dim=occ_dim,
            gate_hidden_channels=int(fusion_cfg.get("gate_hidden_channels", 128)),
            gate_dropout=float(fusion_cfg.get("gate_dropout", 0.2)),
            gate_feature_scale=float(fusion_cfg.get("gate_feature_scale", 0.25)),
            init_bias=fusion_cfg.get("init_bias", {}),
        )

    if kind == "task_region_gated_late":
        return TaskRegionGatedLateFusion(
            fused_channels=fused_channels,
            face_channels=face_channels,
            occ_dim=occ_dim,
            **_region_common_kwargs(fusion_cfg),
        )

    if kind == "task_region_gated_late_no_gaze_occ":
        return TaskRegionGatedLateNoGazeOccFusion(
            fused_channels=fused_channels,
            face_channels=face_channels,
            occ_dim=occ_dim,
            **_region_common_kwargs(fusion_cfg),
        )

    if kind == "explicit_region_mask_gate":
        return ExplicitRegionMaskGateFusion(
            fused_channels=fused_channels,
            face_channels=face_channels,
            occ_dim=occ_dim,
            gate_condition_occ=bool(fusion_cfg.get("gate_condition_occ", True)),
            **_region_common_kwargs(fusion_cfg),
            **_region_occ_kwargs(fusion_cfg),
        )

    if kind == "occ_token_region_transformer":
        return OccTokenRegionTransformerFusion(
            fused_channels=fused_channels,
            face_channels=face_channels,
            occ_dim=occ_dim,
            occ_token_hidden_dim=int(fusion_cfg.get("occ_token_hidden_dim", 128)),
            gate_condition_occ=bool(fusion_cfg.get("gate_condition_occ", True)),
            use_occ_token_residual=bool(fusion_cfg.get("use_occ_token_residual", True)),
            **_region_common_kwargs(fusion_cfg),
        )



    if kind == "task_region_scalar_gated_late":
        return TaskRegionScalarGatedLateFusion(
            fused_channels=fused_channels,
            pose_channels=pose_channels,
            face_channels=face_channels,
            occ_dim=occ_dim,
            region_gate_condition_occ=bool(fusion_cfg.get("region_gate_condition_occ", False)),
            scalar_gate_condition_occ=bool(fusion_cfg.get("scalar_gate_condition_occ", False)),
            **_region_common_kwargs(fusion_cfg),
        )

    if kind == "task_region_scalar_gated_late_no_gaze_occ":
        return TaskRegionScalarGatedLateNoGazeOccFusion(
            fused_channels=fused_channels,
            pose_channels=pose_channels,
            face_channels=face_channels,
            occ_dim=occ_dim,
            region_gate_condition_occ=bool(fusion_cfg.get("region_gate_condition_occ", True)),
            scalar_gate_condition_occ=bool(fusion_cfg.get("scalar_gate_condition_occ", True)),
            **_region_common_kwargs(fusion_cfg),
        )

    if kind == "explicit_region_scalar_mask_gate":
        return ExplicitRegionScalarMaskGateFusion(
            fused_channels=fused_channels,
            pose_channels=pose_channels,
            face_channels=face_channels,
            occ_dim=occ_dim,
            region_gate_condition_occ=bool(fusion_cfg.get("region_gate_condition_occ", True)),
            scalar_gate_condition_occ=bool(fusion_cfg.get("scalar_gate_condition_occ", True)),
            **_region_common_kwargs(fusion_cfg),
            **_region_occ_kwargs(fusion_cfg),
        )

    if kind == "occ_attention_bias":
        return OccAttentionBiasFusion(
            fused_channels=fused_channels,
            face_channels=face_channels,
            occ_dim=occ_dim,
            attention_bias_strength=float(fusion_cfg.get("attention_bias_strength", 2.0)),
            gate_condition_occ=bool(fusion_cfg.get("gate_condition_occ", True)),
            **_region_common_kwargs(fusion_cfg),
            **_region_occ_kwargs(fusion_cfg),
        )

    raise ValueError(
        f"Unknown fusion kind={kind!r}. Supported: " + ", ".join(SUPPORTED_FUSION_KINDS)
    )
