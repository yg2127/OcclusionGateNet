"""Builders for data, model, criterion, optimizer, and scheduler.

The training entrypoint should stay thin. These functions collect the wiring
that should not be duplicated across experiments.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from constants.gaze_zones import NUM_GAZE_ZONES
from src.data.clip_builder import (
    NUM_ACTION_CLASSES,
    NUM_HANDS_CLASSES,
    NUM_TALK_CLASSES,
    build_all_clips,
    save_clip_manifest,
)
from src.data.dataset import MemoryMultitaskDataset, preload_multitask_windows
from src.data.dmd_paths import discover_all
from src.data.fixed_manifest_split import (
    _load_fixed_items_manifest,
    _log_clip_variant_counts,
    _log_duplicate_clip_ids,
    _log_video_variant_counts,
    _log_window_variant_counts,
    build_manifest_split_videos,
)
from src.data.frame_shifts import load_frame_shifts
from src.data.preprocess_pose import build_coco_adjacency
from src.data.split import (
    filter_by_split_info,
    load_split_info,
    save_split_info,
    split_single_fold,
)
from src.experiment.face_shape import resolve_face_shape, resolve_loader_face_v
from src.models.multitask_classifier import MultitaskClassifier
from src.training.loops import MultitaskCriterion, make_class_weights
from src.utils.io import save_json


def build_clip_splits(cfg: dict, save_root: Path, log):
    """Discover videos and build train/val/test clip splits."""
    shifts = load_frame_shifts(cfg["paths"]["frame_shifts"])
    videos = discover_all(
        pose_root=cfg["paths"]["pose_root"],
        face_root=cfg["paths"]["face_root"],
        face5pt_root=cfg["paths"]["face5pt_root"],
        dmd_root=cfg["paths"]["dmd_root"],
        use_distraction=cfg["data"]["use_distraction"],
        use_gaze=cfg["data"]["use_gaze"],
        distraction_sessions=cfg["data"]["distraction_sessions"],
        gaze_sessions=cfg["data"]["gaze_sessions"],
        logger=log,
    )

    if bool(cfg["data"].get("use_fixed_items_manifest", False)):
        fixed_path = cfg["paths"].get("fixed_items_json")
        if not fixed_path:
            raise ValueError("use_fixed_items_manifest=true requires paths.fixed_items_json")

        manifest = _load_fixed_items_manifest(fixed_path)
        train_variants = cfg["data"].get("train_variants", ["clean", "masked"])
        val_variants = cfg["data"].get("val_variants", ["clean", "masked"])
        test_variants = ["clean", "masked"]

        log.info(f"[fixed manifest] loaded={fixed_path}")
        log.info(
            f"[fixed manifest] split_name={manifest.get('split_name')} "
            f"version={manifest.get('version')}"
        )

        train_videos, val_videos, test_clean_videos, test_masked_videos = build_manifest_split_videos(
            manifest=manifest,
            discovered_videos=videos,
            clean_face_root=cfg["paths"]["face_root"],
            train_variants=train_variants,
            val_variants=val_variants,
            test_variants=test_variants,
            logger=log,
        )
        for name, split_videos in (
            ("train", train_videos),
            ("val", val_videos),
            ("test_clean", test_clean_videos),
            ("test_masked", test_masked_videos),
        ):
            _log_video_variant_counts(name, split_videos, log)

        train_clips = _build_split_clips(train_videos, shifts, cfg, log)
        val_clips = _build_split_clips(val_videos, shifts, cfg, log)
        test_clean_clips = _build_split_clips(test_clean_videos, shifts, cfg, log)
        test_masked_clips = _build_split_clips(test_masked_videos, shifts, cfg, log)

        test_eval_clips = {
            "test_clean": test_clean_clips,
            "test_masked": test_masked_clips,
        }

        for name, split_clips in (
            ("train", train_clips),
            ("val", val_clips),
            ("test_clean", test_clean_clips),
            ("test_masked", test_masked_clips),
        ):
            _log_clip_variant_counts(name, split_clips, log)
            _log_duplicate_clip_ids(name, split_clips, log)

        save_clip_manifest(train_clips, save_root / "clip_manifest_train.json")
        save_clip_manifest(val_clips, save_root / "clip_manifest_val.json")
        save_clip_manifest(test_clean_clips, save_root / "clip_manifest_test_clean.json")
        save_clip_manifest(test_masked_clips, save_root / "clip_manifest_test_masked.json")

        save_json(
            {
                "mode": "fixed_items_manifest",
                "fixed_items_json": str(fixed_path),
                "train_variants": train_variants,
                "val_variants": val_variants,
                "test_variants": test_variants,
                "n_train_videos": len(train_videos),
                "n_val_videos": len(val_videos),
                "n_test_clean_videos": len(test_clean_videos),
                "n_test_masked_videos": len(test_masked_videos),
                "n_train_clips": len(train_clips),
                "n_val_clips": len(val_clips),
                "n_test_clean_clips": len(test_clean_clips),
                "n_test_masked_clips": len(test_masked_clips),
            },
            save_root / "fixed_manifest_split_info.json",
        )
        return train_clips, val_clips, test_eval_clips

    clips = _build_split_clips(videos, shifts, cfg, log)
    save_clip_manifest(clips, save_root / "clip_manifest.json")

    split_path = save_root / "split_info.json"
    if split_path.exists():
        log.info(f"[split] reuse={split_path}")
        train_clips, val_clips, test_clips = filter_by_split_info(clips, load_split_info(split_path))
    else:
        train_clips, val_clips, test_clips = split_single_fold(
            clips,
            train_ratio=cfg["split"]["train_ratio"],
            val_ratio=cfg["split"]["val_ratio"],
            test_ratio=cfg["split"]["test_ratio"],
            seed=cfg["seed"],
            logger=log,
        )
        save_split_info(train_clips, val_clips, test_clips, seed=cfg["seed"], path=split_path)
    return train_clips, val_clips, {"test": test_clips}


def _build_split_clips(videos: list, shifts: dict, cfg: dict, log):
    return build_all_clips(
        videos,
        shifts,
        clip_len=cfg["data"]["clip_len"],
        action_stride=cfg["data"]["action_stride"],
        gaze_stride=cfg["data"]["gaze_stride"],
        normal_ratio=cfg["data"]["normal_ratio"],
        seed=cfg["seed"],
        logger=log,
    )


def build_loaders(cfg: dict, train_clips: list, val_clips: list, test_eval_clips: dict, device: str, log):
    """Preload windows and return DataLoaders plus item metadata."""
    face_cfg = cfg["face"]
    common = dict(
        window_size=cfg["window"]["size"],
        window_stride=cfg["window"]["stride"],
        max_windows_per_clip=cfg["window"]["max_per_clip"],
        pose_min_valid_frames=cfg["window"]["pose_min_valid_frames"],
        pose_min_valid_ratio=cfg["window"]["pose_min_valid_ratio"],
        pose_min_valid_joint_ratio=cfg["window"]["pose_min_valid_joint_ratio"],
        face_min_detected_ratio=cfg["window"]["face_min_detected_ratio"],
        joint_conf_thres=cfg["pose"]["joint_conf_thres"],
        face_mode=face_cfg["mode"],
        face_use_z=face_cfg.get("use_z", True),
        face_use_detected_channel=face_cfg.get("use_detected_channel", True),
        face_use_det_score_channel=face_cfg.get("use_det_score_channel", True),
        face_bbox_det_thres=face_cfg.get("bbox_det_thres", 0.25),
        occ_cfg=cfg.get("occ", {}),
        logger=log,
    )

    train_items = preload_multitask_windows(train_clips, desc="preload train", **common)
    val_items = preload_multitask_windows(val_clips, desc="preload val", **common)
    test_eval_items = {
        name: preload_multitask_windows(clips, desc=f"preload {name}", **common)
        for name, clips in test_eval_clips.items()
    }

    _log_window_variant_counts("train", train_items, log)
    _log_window_variant_counts("val", val_items, log)
    for name, items in test_eval_items.items():
        _log_window_variant_counts(name, items, log)

    pin = device != "cpu"
    bs = cfg["train"]["batch_size"]
    nw = cfg["train"]["num_workers"]

    train_loader = DataLoader(
        MemoryMultitaskDataset(train_items),
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=pin,
        drop_last=False,
    )
    val_loader = DataLoader(
        MemoryMultitaskDataset(val_items),
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
    )
    test_eval_loaders = {
        name: DataLoader(
            MemoryMultitaskDataset(items),
            batch_size=bs,
            shuffle=False,
            num_workers=nw,
            pin_memory=pin,
        )
        for name, items in test_eval_items.items()
    }
    return train_loader, val_loader, test_eval_loaders, train_items, val_items, test_eval_items


def build_model(cfg: dict, device: str):
    pose_A = build_coco_adjacency(num_joints=cfg["pose"]["num_joints"], self_link=True)
    pose_in_ch = (
        2
        + (2 if cfg["pose"].get("use_bone", False) else 0)
        + (2 if cfg["pose"].get("use_velocity", False) else 0)
        + (1 if cfg["pose"].get("use_conf_channel", False) else 0)
    )
    face_in_ch, num_face_regions, face_encoder = resolve_face_shape(cfg["face"], cfg["face"]["mode"])
    model = MultitaskClassifier(
        pose_in_channels=pose_in_ch,
        face_in_channels=face_in_ch,
        pose_A=pose_A,
        num_pose_joints=cfg["pose"]["num_joints"],
        num_face_regions=num_face_regions,
        pose_mid_channels=cfg["model"]["pose_mid_channels"],
        face_mid_channels=cfg["model"]["face_mid_channels"],
        fused_channels=cfg["model"]["fused_channels"],
        num_action=NUM_ACTION_CLASSES,
        num_gaze=NUM_GAZE_ZONES,
        num_hands=NUM_HANDS_CLASSES,
        num_talk=NUM_TALK_CLASSES,
        temporal_kind=cfg["model"]["temporal"]["kind"],
        temporal_num_heads=cfg["model"]["temporal"]["num_heads"],
        temporal_dropout=cfg["model"]["temporal"]["dropout"],
        temporal_max_len=cfg["model"]["temporal"]["max_len"],
        dropout_backbone=cfg["model"]["dropout_backbone"],
        dropout_head=cfg["model"]["dropout_head"],
        fusion_cfg=cfg["model"].get("fusion", {"kind": "concat"}),
        occ_cfg=cfg.get("occ", {}),
        face_encoder=face_encoder,
        face_region_scheme=cfg["face"].get("region_scheme", "dms_10"),
        face_region_reduce=cfg["face"].get("region_reduce", "mean"),
    ).to(device)
    meta = {
        "pose_in_ch": pose_in_ch,
        "face_in_ch": face_in_ch,
        "num_face_regions": num_face_regions,
        "face_encoder": face_encoder,
        "loader_face_v": resolve_loader_face_v(cfg["face"], cfg["face"]["mode"]),
    }
    return model, meta


def build_train_objects(cfg: dict, model, train_items: list, device: str):
    ucw = cfg["train"].get("use_class_weight", {})
    act_w = make_class_weights([it.y_action for it in train_items], NUM_ACTION_CLASSES).to(device) if ucw.get("action", True) else None
    gaze_w = make_class_weights([it.y_gaze_fine for it in train_items], NUM_GAZE_ZONES).to(device) if ucw.get("gaze", True) else None
    hands_w = make_class_weights([it.y_hands for it in train_items], NUM_HANDS_CLASSES).to(device) if ucw.get("hands", True) else None
    talk_w = make_class_weights([it.y_talk for it in train_items], NUM_TALK_CLASSES).to(device) if ucw.get("talk", True) else None

    criterion = MultitaskCriterion(
        alpha_action=cfg["loss"]["alpha_action"],
        alpha_gaze=cfg["loss"]["alpha_gaze"],
        alpha_hands=cfg["loss"]["alpha_hands"],
        alpha_talk=cfg["loss"]["alpha_talk"],
        gaze_weak_weight=cfg["loss"]["gaze_weak_weight"],
        action_class_weights=act_w,
        gaze_class_weights=gaze_w,
        hands_class_weights=hands_w,
        talk_class_weights=talk_w,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )
    return criterion, optimizer, scheduler
