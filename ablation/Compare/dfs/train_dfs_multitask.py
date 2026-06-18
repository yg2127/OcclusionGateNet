#!/usr/bin/env python3
"""Train the DMD-paper-inspired multitask baseline on the current DMD labels."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parents[2] if len(THIS_DIR.parents) >= 3 else THIS_DIR.parent.parent
for path in (THIS_DIR, ROOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from dfs_multitask import DFSDMDMultitaskClassifier as DMDOriginalMultitaskClassifier
except ModuleNotFoundError:
    from .dfs_multitask import DFSDMDMultitaskClassifier as DMDOriginalMultitaskClassifier


DEFAULT_CLASSIFICATION_ROOT = "/data/shared/scuppy/Classification_model_V1"
DEFAULT_CONFIG = str(THIS_DIR / "dmd_original_v12_template.yaml")


def _add_classification_root(classification_root: str | Path) -> Path:
    root = Path(classification_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _import_shared_modules():
    from constants.gaze_zones import GAZE_ZONES, NUM_GAZE_ZONES
    from src.data.clip_builder import (
        ClipLabels,
        ClipRecord,
        NUM_ACTION_CLASSES,
        NUM_HANDS_CLASSES,
        NUM_TALK_CLASSES,
        build_all_clips,
        save_clip_manifest,
    )
    from src.data.dataset import MemoryMultitaskDataset, preload_multitask_windows
    from src.data.dmd_paths import discover_all
    from src.data.frame_shifts import load_frame_shifts
    from src.data.preprocess_pose import build_coco_adjacency
    from src.data.split import filter_by_split_info, load_split_info, save_split_info, split_single_fold
    from src.training.loops import HEAD_NAMES, MultitaskCriterion, make_class_weights, run_one_epoch, weighted_score
    from src.utils.io import ensure_dir, load_json, load_yaml, save_json
    from src.utils.logging import get_logger
    from src.utils.seed import set_seed

    return {
        "GAZE_ZONES": GAZE_ZONES,
        "NUM_GAZE_ZONES": NUM_GAZE_ZONES,
        "ClipLabels": ClipLabels,
        "ClipRecord": ClipRecord,
        "NUM_ACTION_CLASSES": NUM_ACTION_CLASSES,
        "NUM_HANDS_CLASSES": NUM_HANDS_CLASSES,
        "NUM_TALK_CLASSES": NUM_TALK_CLASSES,
        "build_all_clips": build_all_clips,
        "save_clip_manifest": save_clip_manifest,
        "MemoryMultitaskDataset": MemoryMultitaskDataset,
        "preload_multitask_windows": preload_multitask_windows,
        "discover_all": discover_all,
        "load_frame_shifts": load_frame_shifts,
        "build_coco_adjacency": build_coco_adjacency,
        "filter_by_split_info": filter_by_split_info,
        "load_split_info": load_split_info,
        "save_split_info": save_split_info,
        "split_single_fold": split_single_fold,
        "HEAD_NAMES": HEAD_NAMES,
        "MultitaskCriterion": MultitaskCriterion,
        "make_class_weights": make_class_weights,
        "run_one_epoch": run_one_epoch,
        "weighted_score": weighted_score,
        "ensure_dir": ensure_dir,
        "load_json": load_json,
        "load_yaml": load_yaml,
        "save_json": save_json,
        "get_logger": get_logger,
        "set_seed": set_seed,
    }


def notify(cfg: dict, message: str) -> None:
    nf = cfg.get("notify") or {}
    if not nf.get("enabled"):
        return
    script = nf.get("script")
    if not script or not Path(script).exists():
        return
    tag = nf.get("tag") or ""
    text = f"[{tag}] {message}" if tag else message
    try:
        subprocess.run(
            [script, text],
            check=False,
            timeout=15,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def save_ckpt(path, model, optimizer, scheduler, epoch, best_score, history, cfg):
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_score": float(best_score),
            "history": history,
            "config": cfg,
        },
        path,
    )


def _save_confusion(cm: np.ndarray, labels: list[str], path: Path) -> None:
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(path, encoding="utf-8-sig")


def measure_window_latency(
    model,
    device: str,
    pose_shape: tuple[int, ...],
    face_shape: tuple[int, ...],
    head_pose_shape: tuple[int, ...] | None = None,
    repeats: int = 30,
) -> float:
    model.eval()
    xb = torch.randn(*pose_shape, device=device)
    xf = torch.randn(*face_shape, device=device)
    xhp = torch.randn(*head_pose_shape, device=device) if head_pose_shape else None

    def _forward():
        return model(xb, xf, xhp) if xhp is not None else model(xb, xf)

    with torch.no_grad():
        for _ in range(5):
            _forward()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            _forward()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return float((time.perf_counter() - t0) / repeats * 1000.0)


def load_clip_manifest_records(manifest_path: str | Path, ClipLabels, ClipRecord, load_json) -> list:
    raw_records = load_json(manifest_path)
    records = []
    for raw in raw_records:
        item = dict(raw)
        labels = ClipLabels(**item.pop("labels"))
        records.append(ClipRecord(labels=labels, **item))
    return records


def _relative_dmd_face_key(path: str | Path) -> str:
    text = str(path)
    if text.endswith(".npz"):
        text = text[:-4]
    for token in ("distraction/dmd/", "gaze/dmd/"):
        idx = text.find(token)
        if idx >= 0:
            return text[idx:]
    return text


def _clone_clip_for_fixed_item(base_clip, item: dict, ClipRecord):
    variant = item.get("variant", "unknown")
    mask_region = item.get("mask_region", "unknown")
    mask_appearance = item.get("mask_appearance", "unknown")
    clip_id = f"{variant}::{mask_region}::{mask_appearance}::{base_clip.clip_id}"
    return ClipRecord(
        clip_id=clip_id,
        subject_key=base_clip.subject_key,
        source=base_clip.source,
        video_prefix=base_clip.video_prefix,
        body_npz=base_clip.body_npz,
        face_npz=item["face_path"],
        face5pt_npz=item["face5pt_path"],
        mosaic_start=base_clip.mosaic_start,
        mosaic_end=base_clip.mosaic_end,
        body_start=base_clip.body_start,
        body_end=base_clip.body_end,
        face_start=base_clip.face_start,
        face_end=base_clip.face_end,
        labels=base_clip.labels,
        head_pose_npz=getattr(base_clip, "head_pose_npz", None),
    )


def _sanitize_clip_label_space(clip, shared: dict):
    labels = clip.labels

    def _valid(value, num_classes: int):
        if value is None:
            return None
        value = int(value)
        return value if 0 <= value < int(num_classes) else None

    clean_labels = shared["ClipLabels"](
        action=_valid(labels.action, shared["NUM_ACTION_CLASSES"]),
        gaze_fine=_valid(labels.gaze_fine, shared["NUM_GAZE_ZONES"]),
        gaze_weak=_valid(labels.gaze_weak, 2),
        hands=_valid(labels.hands, shared["NUM_HANDS_CLASSES"]),
        talk=_valid(labels.talk, shared["NUM_TALK_CLASSES"]),
    )
    return shared["ClipRecord"](
        clip_id=clip.clip_id,
        subject_key=clip.subject_key,
        source=clip.source,
        video_prefix=clip.video_prefix,
        body_npz=clip.body_npz,
        face_npz=clip.face_npz,
        face5pt_npz=clip.face5pt_npz,
        mosaic_start=clip.mosaic_start,
        mosaic_end=clip.mosaic_end,
        body_start=clip.body_start,
        body_end=clip.body_end,
        face_start=clip.face_start,
        face_end=clip.face_end,
        labels=clean_labels,
        head_pose_npz=getattr(clip, "head_pose_npz", None),
    )


def build_fixed_split_clips(cfg: dict, clips: list, shared: dict, log):
    fixed_cfg = cfg.get("fixed_split") or {}
    split_json_path = Path(fixed_cfg["json_path"])
    protocol_name = fixed_cfg.get("protocol", "clean_masked_augmentation_baseline")
    fixed = shared["load_json"](split_json_path)
    protocol = fixed["protocols"][protocol_name]

    by_face_key: dict[str, list] = {}
    for clip in clips:
        key = _relative_dmd_face_key(clip.face_npz)
        by_face_key.setdefault(key, []).append(clip)

    for fallback_path in fixed_cfg.get("fallback_clip_manifest_paths", []) or []:
        fallback_by_key: dict[str, list] = {}
        fallback_clips = load_clip_manifest_records(
            manifest_path=fallback_path,
            ClipLabels=shared["ClipLabels"],
            ClipRecord=shared["ClipRecord"],
            load_json=shared["load_json"],
        )
        for clip in fallback_clips:
            key = _relative_dmd_face_key(clip.face_npz)
            fallback_by_key.setdefault(key, []).append(_sanitize_clip_label_space(clip, shared))
        added_keys = 0
        added_clips = 0
        for key, fallback_records in fallback_by_key.items():
            if key in by_face_key:
                continue
            by_face_key[key] = fallback_records
            added_keys += 1
            added_clips += len(fallback_records)
        log.info(
            f"[fixed split] fallback manifest {fallback_path}:"
            f" added_keys={added_keys} added_clips={added_clips}"
        )

    def expand(item_list_key: str) -> tuple[list, dict]:
        expanded = []
        missing = []
        item_count = 0
        for item in fixed["items"][item_list_key]:
            item_count += 1
            matches = by_face_key.get(item["sample_key"], [])
            if not matches:
                missing.append(item["sample_key"])
                continue
            expanded.extend(_clone_clip_for_fixed_item(base_clip, item, shared["ClipRecord"]) for base_clip in matches)
        stats = {
            "item_key": item_list_key,
            "n_items": item_count,
            "n_clips": len(expanded),
            "n_missing_items": len(missing),
            "missing_items": missing[:20],
        }
        return expanded, stats

    train_clips, train_stats = expand(protocol["train"])
    val_clips, val_stats = expand(protocol["val"])
    test_clean_clips, test_clean_stats = expand(protocol["test_clean"])
    test_masked_clips, test_masked_stats = expand(protocol["test_masked"])

    split_info = {
        "mode": "fixed_split_json",
        "json_path": str(split_json_path),
        "split_name": fixed.get("split_name"),
        "version": fixed.get("version"),
        "protocol": protocol_name,
        "protocol_definition": protocol,
        "stats": {
            "train": train_stats,
            "val": val_stats,
            "test_clean": test_clean_stats,
            "test_masked": test_masked_stats,
        },
        "subjects": fixed.get("subjects", {}),
        "notes": fixed.get("notes", []),
    }
    for name, stats in split_info["stats"].items():
        log.info(
            f"[fixed split] {name}: items={stats['n_items']} clips={stats['n_clips']}"
            f" missing_items={stats['n_missing_items']}"
        )
        if stats["missing_items"]:
            log.warning(f"[fixed split] {name} missing examples: {stats['missing_items'][:3]}")
    return train_clips, val_clips, {"test_clean": test_clean_clips, "test_masked": test_masked_clips}, split_info


def build_or_load_clips(cfg: dict, shared: dict, log) -> list:
    clip_manifest_path = cfg["paths"].get("clip_manifest_path")
    if clip_manifest_path and Path(clip_manifest_path).exists():
        log.info(f"[clips] reusing manifest: {clip_manifest_path}")
        return load_clip_manifest_records(
            manifest_path=clip_manifest_path,
            ClipLabels=shared["ClipLabels"],
            ClipRecord=shared["ClipRecord"],
            load_json=shared["load_json"],
        )

    log.info("[clips] rebuilding from raw discovery roots")
    shifts = shared["load_frame_shifts"](cfg["paths"]["frame_shifts"])
    use_head_pose = bool(cfg.get("baseline", {}).get("use_head_pose", False))
    videos = shared["discover_all"](
        pose_root=cfg["paths"]["pose_root"],
        face_root=cfg["paths"]["face_root"],
        face5pt_root=cfg["paths"]["face5pt_root"],
        dmd_root=cfg["paths"]["dmd_root"],
        use_distraction=cfg["data"]["use_distraction"],
        use_gaze=cfg["data"]["use_gaze"],
        distraction_sessions=cfg["data"]["distraction_sessions"],
        gaze_sessions=cfg["data"]["gaze_sessions"],
        head_pose_root=cfg["paths"].get("head_pose_root") if use_head_pose else None,
        require_head_pose=use_head_pose,
        logger=log,
    )
    clips = shared["build_all_clips"](
        videos,
        shifts,
        clip_len=cfg["data"]["clip_len"],
        action_stride=cfg["data"]["action_stride"],
        gaze_stride=cfg["data"]["gaze_stride"],
        normal_ratio=cfg["data"]["normal_ratio"],
        seed=cfg["seed"],
        logger=log,
    )
    return clips


def main(config_path: str):
    classification_root = _add_classification_root(DEFAULT_CLASSIFICATION_ROOT)
    shared = _import_shared_modules()

    cfg = shared["load_yaml"](config_path)
    classification_root = _add_classification_root(cfg["paths"].get("classification_root", classification_root))
    shared = _import_shared_modules()

    save_root = shared["ensure_dir"](cfg["paths"]["save_root"])
    log = shared["get_logger"]("dfs_baseline_train", log_file=save_root / "train.log")
    log.info(f"config: {config_path}")
    log.info(f"classification_root: {classification_root}")

    shared["set_seed"](cfg["seed"])
    device = cfg["device"] if (cfg["device"] == "cpu" or torch.cuda.is_available()) else "cpu"
    log.info(f"device: {device}")
    shared["save_json"](cfg, save_root / "config.json")

    clips = build_or_load_clips(cfg, shared, log)
    shared["save_clip_manifest"](clips, save_root / "clip_manifest.json")

    fixed_split_cfg = cfg.get("fixed_split") or {}
    if fixed_split_cfg.get("enabled", False):
        log.info(f"[fixed split] using {fixed_split_cfg['json_path']}")
        train_clips, val_clips, test_clip_sets, split_info = build_fixed_split_clips(cfg, clips, shared, log)
        shared["save_json"](split_info, save_root / "split_info.json")
    else:
        split_path = Path(cfg["paths"].get("split_info_path", "")) if cfg["paths"].get("split_info_path") else save_root / "split_info.json"
        if split_path.exists():
            log.info(f"[split] reusing: {split_path}")
            split_info = shared["load_split_info"](split_path)
            train_clips, val_clips, test_clips = shared["filter_by_split_info"](clips, split_info)
            if split_path != save_root / "split_info.json":
                shared["save_json"](split_info, save_root / "split_info.json")
        else:
            train_clips, val_clips, test_clips = shared["split_single_fold"](
                clips,
                train_ratio=cfg["split"]["train_ratio"],
                val_ratio=cfg["split"]["val_ratio"],
                test_ratio=cfg["split"]["test_ratio"],
                seed=cfg["seed"],
                logger=log,
            )
            shared["save_split_info"](train_clips, val_clips, test_clips, seed=cfg["seed"], path=save_root / "split_info.json")
        test_clip_sets = {"test": test_clips}

    face_cfg = cfg["face"]
    face_mode = face_cfg["mode"]
    use_head_pose = bool(cfg.get("baseline", {}).get("use_head_pose", False))
    common = dict(
        window_size=cfg["window"]["size"],
        window_stride=cfg["window"]["stride"],
        max_windows_per_clip=cfg["window"]["max_per_clip"],
        pose_min_valid_frames=cfg["window"]["pose_min_valid_frames"],
        pose_min_valid_ratio=cfg["window"]["pose_min_valid_ratio"],
        pose_min_valid_joint_ratio=cfg["window"]["pose_min_valid_joint_ratio"],
        face_min_detected_ratio=cfg["window"]["face_min_detected_ratio"],
        joint_conf_thres=cfg["pose"]["joint_conf_thres"],
        face_mode=face_mode,
        face_use_z=face_cfg.get("use_z", True),
        face_use_detected_channel=face_cfg.get("use_detected_channel", True),
        face_use_det_score_channel=face_cfg.get("use_det_score_channel", True),
        face_bbox_det_thres=face_cfg.get("bbox_det_thres", 0.25),
        use_head_pose=use_head_pose,
        logger=log,
    )
    train_items = shared["preload_multitask_windows"](train_clips, desc="preload train", **common)
    val_items = shared["preload_multitask_windows"](val_clips, desc="preload val", **common)
    test_items_by_split = {
        name: shared["preload_multitask_windows"](split_clips, desc=f"preload {name}", **common)
        for name, split_clips in test_clip_sets.items()
    }
    test_window_counts = {name: len(items) for name, items in test_items_by_split.items()}
    log.info(f"windows: train={len(train_items)} val={len(val_items)} test={test_window_counts}")

    train_ds = shared["MemoryMultitaskDataset"](train_items)
    val_ds = shared["MemoryMultitaskDataset"](val_items)

    pin = device != "cpu"
    bs = cfg["train"]["batch_size"]
    nw = cfg["train"]["num_workers"]
    nw_eval = min(nw, 2)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pin, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw_eval, pin_memory=pin)
    test_loaders = {
        name: DataLoader(
            shared["MemoryMultitaskDataset"](items),
            batch_size=bs,
            shuffle=False,
            num_workers=nw_eval,
            pin_memory=pin,
        )
        for name, items in test_items_by_split.items()
    }

    pose_in_ch = (
        2
        + (2 if cfg["pose"]["use_bone"] else 0)
        + (2 if cfg["pose"]["use_velocity"] else 0)
        + (1 if cfg["pose"]["use_conf_channel"] else 0)
    )
    if face_mode in ("facemesh", "facemesh_full"):
        face_in_ch = (3 if face_cfg.get("use_z", True) else 2) + (1 if face_cfg.get("use_detected_channel", True) else 0)
        num_face_regions = face_cfg.get("num_landmarks", 478) if face_mode == "facemesh_full" else face_cfg.get("num_regions", 10)
    else:
        face_in_ch = (
            2
            + (1 if face_cfg.get("use_detected_channel", True) else 0)
            + (1 if face_cfg.get("use_det_score_channel", True) else 0)
        )
        num_face_regions = face_cfg.get("num_regions", 5)

    baseline_cfg = cfg["baseline"]
    model = DMDOriginalMultitaskClassifier(
        pose_in_channels=pose_in_ch,
        face_in_channels=face_in_ch,
        num_pose_joints=cfg["pose"]["num_joints"],
        num_face_regions=num_face_regions,
        proj_dim=baseline_cfg["proj_dim"],
        stream_dim=baseline_cfg["stream_dim"],
        fusion_dim=baseline_cfg["fusion_dim"],
        dropout=baseline_cfg["dropout"],
        num_action=shared["NUM_ACTION_CLASSES"],
        num_gaze=shared["NUM_GAZE_ZONES"],
        num_hands=shared["NUM_HANDS_CLASSES"],
        num_talk=shared["NUM_TALK_CLASSES"],
        fusion_kind=baseline_cfg["fusion_kind"],
        use_head_pose=use_head_pose,
        head_pose_in_channels=2,
        num_head_pose_axes=3,
        score_fusion_body_weight=baseline_cfg.get("score_fusion_body_weight", 1.0),
        score_fusion_face_weight=baseline_cfg.get("score_fusion_face_weight", 1.0),
        score_fusion_head_pose_weight=baseline_cfg.get("score_fusion_head_pose_weight", 0.5),
    ).to(device)
    n_params = sum(parameter.numel() for parameter in model.parameters())
    log.info(
        f"model params: {n_params/1e6:.3f}M"
        f" pose_ch={pose_in_ch} face_ch={face_in_ch} face_mode={face_mode} face_V={num_face_regions}"
        f" fusion={baseline_cfg['fusion_kind']}"
    )

    ucw = cfg["train"]["use_class_weight"]
    act_w = shared["make_class_weights"]([it.y_action for it in train_items], shared["NUM_ACTION_CLASSES"]).to(device) if ucw.get("action", True) else None
    gaze_w = shared["make_class_weights"]([it.y_gaze_fine for it in train_items], shared["NUM_GAZE_ZONES"]).to(device) if ucw.get("gaze", True) else None
    hands_w = shared["make_class_weights"]([it.y_hands for it in train_items], shared["NUM_HANDS_CLASSES"]).to(device) if ucw.get("hands", True) else None
    talk_w = shared["make_class_weights"]([it.y_talk for it in train_items], shared["NUM_TALK_CLASSES"]).to(device) if ucw.get("talk", True) else None

    criterion = shared["MultitaskCriterion"](
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    score_weights = cfg["best_score_weights"]
    epochs = cfg["train"]["epochs"]
    patience = cfg["train"]["patience"]
    grad_clip = cfg["train"]["grad_clip_norm"]
    save_every = cfg["train"]["save_every_epoch"]

    history: list[dict] = []
    best_score = -1.0
    best_epoch = -1
    no_improve = 0
    start_epoch = 1

    last_ckpt = save_root / "last.pt"
    resume_path = Path(cfg["train"]["resume_path"]) if cfg["train"].get("resume_path") else last_ckpt
    if cfg["train"]["resume"] and resume_path.exists():
        log.info(f"[resume] {resume_path}")
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        if ck.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ck["scheduler_state_dict"])
        start_epoch = int(ck["epoch"]) + 1
        best_score = float(ck.get("best_score", -1.0))
        history = ck.get("history", [])

    log.info("==== training start ====")
    log.info(
        (
            f"classes(action={shared['NUM_ACTION_CLASSES']}, gaze={shared['NUM_GAZE_ZONES']}, "
            f"hands={shared['NUM_HANDS_CLASSES']}, talk={shared['NUM_TALK_CLASSES']})\n"
            f"windows train={len(train_items)} val={len(val_items)} test={test_window_counts}\n"
            f"fusion={baseline_cfg['fusion_kind']} face_mode={face_mode} bs={bs} lr={cfg['train']['lr']}"
        )
    )

    ablation_cfg = cfg.get("ablation", {}) or {}
    if ablation_cfg:
        log.info(f"ablation: {ablation_cfg}")

    for epoch in range(start_epoch, epochs + 1):
        train_out = shared["run_one_epoch"](
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            train=True,
            grad_clip_norm=grad_clip,
            epoch_idx=epoch,
            total_epochs=epochs,
            agg_mode=cfg["eval"]["clip_agg_mode"],
            topk=cfg["eval"]["clip_topk"],
            ablation_cfg=ablation_cfg,
        )
        val_out = shared["run_one_epoch"](
            model,
            val_loader,
            optimizer,
            criterion,
            device,
            train=False,
            grad_clip_norm=None,
            epoch_idx=epoch,
            total_epochs=epochs,
            agg_mode=cfg["eval"]["clip_agg_mode"],
            topk=cfg["eval"]["clip_topk"],
            ablation_cfg=ablation_cfg,
        )

        val_score = shared["weighted_score"](val_out, score_weights)
        scheduler.step(val_score)
        cur_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": cur_lr,
            "train_loss": train_out["loss"],
            "val_loss": val_out["loss"],
            "val_weighted_score": val_score,
        }
        for head in shared["HEAD_NAMES"]:
            th = train_out["heads"][head]
            vh = val_out["heads"][head]
            row[f"train_{head}_w_f1"] = th["window_f1_macro"]
            row[f"val_{head}_w_f1"] = vh["window_f1_macro"]
            row[f"val_{head}_c_f1"] = vh.get("clip_f1_macro", 0.0)
            row[f"val_{head}_c_acc"] = vh.get("clip_acc", 0.0)
        vgb = val_out.get("gaze_binary_on_distraction", {})
        row["val_gaze_bin_w_f1"] = vgb.get("window_f1_macro", 0.0)
        row["val_gaze_bin_c_f1"] = vgb.get("clip_f1_macro", 0.0)
        row["val_gaze_bin_c_acc"] = vgb.get("clip_acc", 0.0)
        history.append(row)

        log.info(
            f"[{epoch}/{epochs}] lr={cur_lr:.2e} train_loss={train_out['loss']:.4f} val_loss={val_out['loss']:.4f}"
            f" | action c_f1={val_out['heads']['action']['clip_f1_macro']:.4f}"
            f" | gaze c_f1={val_out['heads']['gaze']['clip_f1_macro']:.4f}"
            f" | hands c_f1={val_out['heads']['hands']['clip_f1_macro']:.4f}"
            f" | talk c_f1={val_out['heads']['talk']['clip_f1_macro']:.4f}"
            f" | score={val_score:.4f}"
        )

        pd.DataFrame(history).to_csv(save_root / "metrics.csv", index=False, encoding="utf-8-sig")
        shared["save_json"](history, save_root / "history.json")

        if save_every:
            save_ckpt(save_root / "last.pt", model, optimizer, scheduler, epoch, best_score, history, cfg)

        for head in shared["HEAD_NAMES"]:
            vh = val_out["heads"][head]
            if not vh.get("clip_targets"):
                continue
            if head == "action":
                num = shared["NUM_ACTION_CLASSES"]
            elif head == "gaze":
                num = shared["NUM_GAZE_ZONES"]
            elif head == "hands":
                num = shared["NUM_HANDS_CLASSES"]
            else:
                num = shared["NUM_TALK_CLASSES"]
            cm = confusion_matrix(vh["clip_targets"], vh["clip_preds"], labels=list(range(num)))
            _save_confusion(cm, [str(i) for i in range(num)], save_root / f"val_{head}_clip_confusion_epoch_{epoch:03d}.csv")

        is_best = val_score > best_score
        if is_best:
            best_score = val_score
            best_epoch = epoch
            no_improve = 0
            save_ckpt(save_root / "best.pt", model, optimizer, scheduler, epoch, best_score, history, cfg)
            log.info(f"[best] epoch={epoch} score={best_score:.4f}")
        else:
            no_improve += 1

        nf = cfg.get("notify") or {}
        if nf.get("enabled") and (nf.get("every_epoch") or is_best):
            star = " ★best" if is_best else ""
            msg = (
                f"ep {epoch}/{epochs}{star}  lr={cur_lr:.2e}  score={val_score:.3f}\n"
                f"  action c_f1={val_out['heads']['action']['clip_f1_macro']:.3f}\n"
                f"  gaze   c_f1={val_out['heads']['gaze']['clip_f1_macro']:.3f}\n"
                f"  hands  c_f1={val_out['heads']['hands']['clip_f1_macro']:.3f}\n"
                f"  talk   c_f1={val_out['heads']['talk']['clip_f1_macro']:.3f}\n"
                f"  best={best_score:.3f}@{best_epoch}  no_improve={no_improve}/{patience}"
            )
            notify(cfg, msg)

        if no_improve >= patience:
            log.info(f"[early stop] no improve {patience} epochs")
            notify(cfg, f"⏹ early stop ep {epoch} (best={best_score:.4f}@{best_epoch})")
            break

    log.info(f"==== done. best_epoch={best_epoch} best_score={best_score:.4f} ====")

    best_path = save_root / "best.pt"
    if best_path.exists():
        ck = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])

    def _num_classes_for_head(head: str) -> int:
        if head == "action":
            return shared["NUM_ACTION_CLASSES"]
        if head == "gaze":
            return shared["NUM_GAZE_ZONES"]
        if head == "hands":
            return shared["NUM_HANDS_CLASSES"]
        return shared["NUM_TALK_CLASSES"]

    def _head_metrics(head_out: dict) -> dict:
        return {
            "window_f1_macro": head_out["window_f1_macro"],
            "window_acc": head_out["window_acc"],
            "clip_f1_macro": head_out.get("clip_f1_macro", 0.0),
            "clip_acc": head_out.get("clip_acc", 0.0),
        }

    test_outputs = {}
    test_summaries = {}
    for split_name, test_loader in test_loaders.items():
        out = shared["run_one_epoch"](
            model,
            test_loader,
            optimizer,
            criterion,
            device,
            train=False,
            grad_clip_norm=None,
            agg_mode=cfg["eval"]["clip_agg_mode"],
            topk=cfg["eval"]["clip_topk"],
            ablation_cfg=ablation_cfg,
        )
        test_outputs[split_name] = out
        for head in shared["HEAD_NAMES"]:
            th = out["heads"][head]
            if not th.get("clip_targets"):
                continue
            num = _num_classes_for_head(head)
            cm = confusion_matrix(th["clip_targets"], th["clip_preds"], labels=list(range(num)))
            _save_confusion(cm, [str(i) for i in range(num)], save_root / f"{split_name}_{head}_clip_confusion.csv")
            log.info(
                f"[{split_name} {head}] c_f1={th['clip_f1_macro']:.4f}"
                f" c_acc={th['clip_acc']:.4f}"
                f" w_f1={th['window_f1_macro']:.4f}"
            )

        gaze_bin = out.get("gaze_binary_on_distraction", {})
        if gaze_bin:
            log.info(
                f"[{split_name} gaze_bin(dist)] window acc={gaze_bin.get('window_acc', 0):.4f}"
                f" f1={gaze_bin.get('window_f1_macro', 0):.4f}"
                f" | clip acc={gaze_bin.get('clip_acc', 0):.4f}"
                f" f1={gaze_bin.get('clip_f1_macro', 0):.4f}"
                f" | n={gaze_bin.get('n')}"
            )
            if gaze_bin.get("clip_targets"):
                cm = confusion_matrix(gaze_bin["clip_targets"], gaze_bin["clip_preds"], labels=[0, 1])
                _save_confusion(cm, ["not_front", "front"], save_root / f"{split_name}_gaze_binary_on_distraction.csv")

        test_summaries[split_name] = {
            "loss": out["loss"],
            "per_head": {head: _head_metrics(out["heads"][head]) for head in shared["HEAD_NAMES"]},
            "gaze_binary_on_distraction": {
                "window_acc": gaze_bin.get("window_acc", 0.0),
                "window_f1_macro": gaze_bin.get("window_f1_macro", 0.0),
                "clip_acc": gaze_bin.get("clip_acc", 0.0),
                "clip_f1_macro": gaze_bin.get("clip_f1_macro", 0.0),
                "n_windows": gaze_bin.get("n", 0),
                "support_front": gaze_bin.get("support_front", 0),
                "support_not_front": gaze_bin.get("support_not_front", 0),
            }
            if gaze_bin
            else None,
            "n_clips": len(test_clip_sets[split_name]),
            "n_windows": test_window_counts.get(split_name, 0),
        }

    primary_test_name = "test_clean" if "test_clean" in test_outputs else next(iter(test_outputs))
    test_out = test_outputs[primary_test_name]
    tgb = test_out.get("gaze_binary_on_distraction", {})

    masked_drop = {}
    if "test_clean" in test_summaries and "test_masked" in test_summaries:
        clean_heads = test_summaries["test_clean"]["per_head"]
        masked_heads = test_summaries["test_masked"]["per_head"]
        for head in shared["HEAD_NAMES"]:
            masked_drop[head] = {}
            for metric in ("clip_f1_macro", "clip_acc", "window_f1_macro", "window_acc"):
                clean_value = float(clean_heads[head].get(metric, 0.0))
                masked_value = float(masked_heads[head].get(metric, 0.0))
                masked_drop[head][metric] = {
                    "clean": clean_value,
                    "masked": masked_value,
                    "drop_abs": clean_value - masked_value,
                    "drop_rel": (clean_value - masked_value) / clean_value if abs(clean_value) > 1e-12 else None,
                }

    lat_ms = measure_window_latency(
        model,
        device,
        pose_shape=(1, pose_in_ch, cfg["window"]["size"], cfg["pose"]["num_joints"]),
        face_shape=(1, face_in_ch, cfg["window"]["size"], num_face_regions),
        head_pose_shape=(1, 2, cfg["window"]["size"], 3) if use_head_pose else None,
    )
    log.info(f"[latency] {lat_ms:.2f} ms/window")

    summary = {
        "baseline_name": "dmd_original_multitask",
        "baseline_fusion_kind": baseline_cfg["fusion_kind"],
        "best_epoch": best_epoch,
        "best_score": best_score,
        "primary_test_split": primary_test_name,
        "test_loss": test_out["loss"],
        "test_per_head": {
            head: {
                "window_f1_macro": test_out["heads"][head]["window_f1_macro"],
                "window_acc": test_out["heads"][head]["window_acc"],
                "clip_f1_macro": test_out["heads"][head].get("clip_f1_macro", 0.0),
                "clip_acc": test_out["heads"][head].get("clip_acc", 0.0),
            }
            for head in shared["HEAD_NAMES"]
        },
        "test_gaze_binary_on_distraction": {
            "window_acc": tgb.get("window_acc", 0.0),
            "window_f1_macro": tgb.get("window_f1_macro", 0.0),
            "clip_acc": tgb.get("clip_acc", 0.0),
            "clip_f1_macro": tgb.get("clip_f1_macro", 0.0),
            "n_windows": tgb.get("n", 0),
            "support_front": tgb.get("support_front", 0),
            "support_not_front": tgb.get("support_not_front", 0),
        }
        if tgb
        else None,
        "test_splits": test_summaries,
        "masked_drop": masked_drop,
        "n_action_classes": shared["NUM_ACTION_CLASSES"],
        "n_gaze_classes": shared["NUM_GAZE_ZONES"],
        "n_hands_classes": shared["NUM_HANDS_CLASSES"],
        "n_talk_classes": shared["NUM_TALK_CLASSES"],
        "n_train_clips": len(train_clips),
        "n_val_clips": len(val_clips),
        "n_test_clips": {name: len(split_clips) for name, split_clips in test_clip_sets.items()},
        "n_train_windows": len(train_items),
        "n_val_windows": len(val_items),
        "n_test_windows": test_window_counts,
        "inference_ms_per_window": lat_ms,
        "model_params": n_params,
        "face_mode": face_mode,
        "score_weights": score_weights,
    }
    shared["save_json"](summary, save_root / "summary.json")
    log.info(f"[done] summary -> {save_root / 'summary.json'}")
    notify(
        cfg,
        (
            f"done best ep {best_epoch} score={best_score:.4f}\n"
            + " | ".join(
                f"{head} c_f1={test_out['heads'][head].get('clip_f1_macro', 0):.3f}"
                for head in shared["HEAD_NAMES"]
            )
            + f"\nlatency {lat_ms:.2f} ms"
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    main(args.config)
