"""Compact training runner for V5 DMS multitask experiments."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from src.evaluation.v1_eval import (
    _gaze_binary_summary,
    _head_metrics_for_summary,
    _save_confusion,
    build_clean_masked_drop_summary,
    evaluate_test_split,
)
from src.experiment.runtime import measure_window_latency, notify, save_ckpt
from src.training.builders import (
    build_clip_splits,
    build_eval_loader,
    build_model,
    build_train_objects,
    build_train_val_loaders,
)
from src.training.loops import HEAD_NAMES, run_one_epoch, weighted_score
from src.utils.finalize import derive_results_name, finalize_results
from src.utils.io import ensure_dir, load_yaml, save_json
from src.utils.logging import get_logger
from src.utils.seed import set_seed


def _num_classes_for_confusion(cfg: dict, head: str) -> int:
    from constants.gaze_zones import NUM_GAZE_ZONES
    from src.data.clip_builder import NUM_ACTION_CLASSES, NUM_HANDS_CLASSES, NUM_TALK_CLASSES

    if head == "action":
        return NUM_ACTION_CLASSES
    if head == "gaze":
        return NUM_GAZE_ZONES
    if head == "hands":
        return NUM_HANDS_CLASSES
    if head == "talk":
        return NUM_TALK_CLASSES
    raise ValueError(head)


def _save_val_confusions(val_out: dict, save_root: Path, epoch: int, cfg: dict) -> None:
    for head in HEAD_NAMES:
        vh = val_out["heads"][head]
        if not vh.get("clip_targets"):
            continue
        num = _num_classes_for_confusion(cfg, head)
        cm = confusion_matrix(vh["clip_targets"], vh["clip_preds"], labels=list(range(num)))
        _save_confusion(cm, [str(i) for i in range(num)], save_root / f"val_{head}_clip_confusion_epoch_{epoch:03d}.csv")


def _resume_if_needed(cfg: dict, model, optimizer, scheduler, save_root: Path, device: str, log):
    history = []
    best_score = -1.0
    start_epoch = 1

    last_ckpt = save_root / "last.pt"
    resume_path = Path(cfg["train"].get("resume_path") or last_ckpt)
    if cfg["train"].get("resume", False) and resume_path.exists():
        log.info(f"[resume] {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_score = float(ckpt.get("best_score", -1.0))
        history = ckpt.get("history", [])
    return start_epoch, best_score, history


def train_and_evaluate(config_path: str) -> dict:
    cfg = load_yaml(config_path)
    save_root = ensure_dir(cfg["paths"]["save_root"])
    log = get_logger("v5_train", log_file=save_root / "train.log")
    log.info(f"config={config_path}")

    set_seed(cfg["seed"])
    device = cfg["device"] if (cfg["device"] == "cpu" or torch.cuda.is_available()) else "cpu"
    save_json(cfg, save_root / "config.json")
    log.info(f"device={device}")

    train_clips, val_clips, test_eval_clips = build_clip_splits(cfg, save_root, log)

    # IMPORTANT: build and move the full model to GPU before preloading window tensors.
    # This keeps the requested V1/full-FaceMesh architecture unchanged while avoiding
    # the previous order: preload all train/val/test items first, then allocate CUDA model.
    model, model_meta = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(
        f"model params={n_params / 1e6:.3f}M "
        f"fusion={cfg['model'].get('fusion', {}).get('kind', 'concat')} "
        f"face_mode={cfg['face']['mode']} face_encoder={model_meta['face_encoder']} "
        f"loader_face_V={model_meta['loader_face_v']} model_face_V={model_meta['num_face_regions']} "
        f"occ_enabled={bool(cfg.get('occ', {}).get('enabled', False))} "
        f"occ_dim={int(cfg.get('occ', {}).get('dim', 0)) if cfg.get('occ', {}).get('enabled', False) else 0}"
    )

    train_loader, val_loader, train_items, val_items = build_train_val_loaders(
        cfg, train_clips, val_clips, device, log
    )

    criterion, optimizer, scheduler = build_train_objects(cfg, model, train_items, device)
    start_epoch, best_score, history = _resume_if_needed(cfg, model, optimizer, scheduler, save_root, device, log)

    epochs = int(cfg["train"]["epochs"])
    patience = int(cfg["train"]["patience"])
    grad_clip = cfg["train"].get("grad_clip_norm")
    save_every = bool(cfg["train"].get("save_every_epoch", True))
    score_weights = cfg["best_score_weights"]
    ablation_cfg = cfg.get("ablation", {})

    best_epoch = -1
    no_improve = 0
    notify(
        cfg,
        f"start fusion={cfg['model'].get('fusion', {}).get('kind', 'concat')} "
        f"windows train={len(train_items)} val={len(val_items)} "
        f"test_preload=delayed_one_split_at_a_time",
    )

    for epoch in range(start_epoch, epochs + 1):
        ablation_cfg = cfg.get("ablation", {})
        train_out = run_one_epoch(
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
        val_out = run_one_epoch(
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

        val_score = weighted_score(val_out, score_weights)
        scheduler.step(val_score)
        cur_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": cur_lr,
            "train_loss": train_out["loss"],
            "val_loss": val_out["loss"],
            "val_weighted_score": val_score,
        }
        for head in HEAD_NAMES:
            row[f"train_{head}_w_f1"] = train_out["heads"][head]["window_f1_macro"]
            row[f"val_{head}_w_f1"] = val_out["heads"][head]["window_f1_macro"]
            row[f"val_{head}_c_f1"] = val_out["heads"][head].get("clip_f1_macro", 0.0)
            row[f"val_{head}_c_acc"] = val_out["heads"][head].get("clip_acc", 0.0)
        vgb = val_out.get("gaze_binary_on_distraction", {})
        row["val_gaze_bin_w_f1"] = vgb.get("window_f1_macro", 0.0)
        row["val_gaze_bin_c_f1"] = vgb.get("clip_f1_macro", 0.0)
        row["val_gaze_bin_c_acc"] = vgb.get("clip_acc", 0.0)
        history.append(row)

        log.info(
            f"[{epoch}/{epochs}] lr={cur_lr:.2e} train_loss={train_out['loss']:.4f} "
            f"val_loss={val_out['loss']:.4f} "
            f"| action={val_out['heads']['action'].get('clip_f1_macro', 0):.4f} "
            f"gaze={val_out['heads']['gaze'].get('clip_f1_macro', 0):.4f} "
            f"hands={val_out['heads']['hands'].get('clip_f1_macro', 0):.4f} "
            f"talk={val_out['heads']['talk'].get('clip_f1_macro', 0):.4f} "
            f"| score={val_score:.4f}"
        )

        pd.DataFrame(history).to_csv(save_root / "metrics.csv", index=False, encoding="utf-8-sig")
        save_json(history, save_root / "history.json")
        if save_every:
            save_ckpt(save_root / "last.pt", model, optimizer, scheduler, epoch, best_score, history, cfg)
        _save_val_confusions(val_out, save_root, epoch, cfg)

        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            no_improve = 0
            save_ckpt(save_root / "best.pt", model, optimizer, scheduler, epoch, best_score, history, cfg)
            log.info(f"[best] epoch={epoch} score={best_score:.4f}")
        else:
            no_improve += 1

        if no_improve >= patience:
            log.info(f"[early stop] no improve {patience} epochs")
            notify(cfg, f"early stop ep {epoch} best={best_score:.4f}@{best_epoch}")
            break

    log.info(f"==== training done best_epoch={best_epoch} best_score={best_score:.4f} ====")

    best_path = save_root / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    # Release train/val DataLoader references before test preloading.
    # The train_items list is kept only long enough for summary counts.
    import gc

    n_train_windows = len(train_items)
    n_val_windows = len(val_items)
    del train_loader, val_loader, train_items, val_items
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()

    test_outputs = {}
    n_test_windows_by_split = {}
    for name, clips in test_eval_clips.items():
        test_loader, test_items = build_eval_loader(cfg, name, clips, device, log)
        n_test_windows_by_split[name] = len(test_items)
        test_outputs[name] = evaluate_test_split(
            name=name,
            model=model,
            loader=test_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            cfg=cfg,
            ablation_cfg=ablation_cfg,
            save_root=save_root,
            log=log,
        )
        del test_loader, test_items
        gc.collect()
        if device != "cpu":
            torch.cuda.empty_cache()

    test_drop = None
    if "test_clean" in test_outputs and "test_masked" in test_outputs:
        test_drop = build_clean_masked_drop_summary(test_outputs["test_clean"], test_outputs["test_masked"])
        save_json(test_drop, save_root / "test_clean_vs_masked_drop.json")
        pd.DataFrame([{"head": h, **d} for h, d in test_drop.items()]).to_csv(
            save_root / "test_clean_vs_masked_drop.csv",
            index=False,
            encoding="utf-8-sig",
        )

    occ_dim = int(cfg.get("occ", {}).get("dim", 0)) if cfg.get("occ", {}).get("enabled", False) else 0
    lat_ms = measure_window_latency(
        model,
        device,
        pose_shape=(1, model_meta["pose_in_ch"], cfg["window"]["size"], cfg["pose"]["num_joints"]),
        face_shape=(1, model_meta["face_in_ch"], cfg["window"]["size"], model_meta["loader_face_v"]),
        occ_dim=occ_dim,
    )

    test_summary = {
        name: {
            "loss": out["loss"],
            "per_head": _head_metrics_for_summary(out),
            "gaze_binary_on_distraction": _gaze_binary_summary(out),
        }
        for name, out in test_outputs.items()
    }
    summary = {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "test_splits": test_summary,
        "test_clean_vs_masked_drop": test_drop,
        "n_train_clips": len(train_clips),
        "n_val_clips": len(val_clips),
        "n_test_clips_by_split": {name: len(clips) for name, clips in test_eval_clips.items()},
        "n_train_windows": n_train_windows,
        "n_val_windows": n_val_windows,
        "n_test_windows_by_split": n_test_windows_by_split,
        "inference_ms_per_window": lat_ms,
        "model_params": n_params,
        "fusion_kind": cfg["model"].get("fusion", {}).get("kind", "concat"),
        **model_meta,
    }
    if "test" in test_outputs:
        summary["test_loss"] = test_outputs["test"]["loss"]
        summary["test_per_head"] = _head_metrics_for_summary(test_outputs["test"])
        summary["test_gaze_binary_on_distraction"] = _gaze_binary_summary(test_outputs["test"])

    save_json(summary, save_root / "summary.json")
    log.info(f"[latency] {lat_ms:.2f} ms/window")
    log.info(f"[done] summary -> {save_root / 'summary.json'}")

    try:
        results_root = Path(cfg.get("paths", {}).get("results_root", "results"))
        results_name = derive_results_name(save_root)
        tag = (cfg.get("notify") or {}).get("tag", results_name)
        finalize_results(save_root=save_root, results_root=results_root, results_name=results_name, title=f"V5 Multitask — {tag}", logger=log)
    except Exception as exc:
        log.warning(f"[finalize] failed: {exc}")

    notify(cfg, f"done best={best_score:.4f}@{best_epoch} latency={lat_ms:.2f}ms")
    return summary
