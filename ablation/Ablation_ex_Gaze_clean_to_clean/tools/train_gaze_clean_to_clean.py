from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import pandas as pd
import torch

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dmd_paths import discover_all
from src.data.frame_shifts import load_frame_shifts
from src.data.fixed_manifest_split import (
    _load_fixed_items_manifest,
    build_manifest_split_videos,
    _log_video_variant_counts,
    _log_clip_variant_counts,
    _log_duplicate_clip_ids,
    _log_window_variant_counts,
)
from src.data.clip_builder import build_all_clips, save_clip_manifest
from src.training.builders import build_loaders, build_model, build_train_objects
from src.training.loops import run_one_epoch, weighted_score
from src.evaluation.v1_eval import evaluate_test_split, _head_metrics_for_summary
from src.experiment.runtime import save_ckpt, measure_window_latency
from src.utils.io import ensure_dir, load_yaml, save_json
from src.utils.logging import get_logger
from src.utils.seed import set_seed


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


def _log_cuda_mem(tag: str, log):
    if not torch.cuda.is_available():
        return

    try:
        free, total = torch.cuda.mem_get_info()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()

        log.info(
            f"[cuda mem] {tag} | "
            f"free={free / 1024**3:.2f}GB "
            f"total={total / 1024**3:.2f}GB "
            f"allocated={allocated / 1024**3:.4f}GB "
            f"reserved={reserved / 1024**3:.4f}GB"
        )
    except Exception as e:
        log.warning(f"[cuda mem] failed at {tag}: {e}")


def build_clean_only_clip_splits(cfg: dict, save_root: Path, log):
    """
    Clean FaceMesh upper-bound split builder.

    Train: clean FaceMesh only
    Val  : clean FaceMesh only
    Test : clean FaceMesh only

    This intentionally discards masked test videos.
    """
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

    fixed_path = cfg["paths"]["fixed_items_json"]
    manifest = _load_fixed_items_manifest(fixed_path)

    train_variants = cfg["data"].get("train_variants", ["clean"])
    val_variants = cfg["data"].get("val_variants", ["clean"])

    log.info(f"[fixed manifest] loaded={fixed_path}")
    log.info(
        f"[fixed manifest] split_name={manifest.get('split_name')} "
        f"version={manifest.get('version')}"
    )
    log.info("[clean-to-clean] train=clean only, val=clean only, test=clean only")

    train_videos, val_videos, test_clean_videos, _test_masked_videos = build_manifest_split_videos(
        manifest=manifest,
        discovered_videos=videos,
        clean_face_root=cfg["paths"]["face_root"],
        train_variants=train_variants,
        val_variants=val_variants,
        test_variants=["clean"],
        logger=log,
    )

    for name, split_videos in (
        ("train", train_videos),
        ("val", val_videos),
        ("test_clean", test_clean_videos),
    ):
        _log_video_variant_counts(name, split_videos, log)

    train_clips = _build_split_clips(train_videos, shifts, cfg, log)
    val_clips = _build_split_clips(val_videos, shifts, cfg, log)
    test_clean_clips = _build_split_clips(test_clean_videos, shifts, cfg, log)

    for name, split_clips in (
        ("train", train_clips),
        ("val", val_clips),
        ("test_clean", test_clean_clips),
    ):
        _log_clip_variant_counts(name, split_clips, log)
        _log_duplicate_clip_ids(name, split_clips, log)

    save_clip_manifest(train_clips, save_root / "clip_manifest_train_clean.json")
    save_clip_manifest(val_clips, save_root / "clip_manifest_val_clean.json")
    save_clip_manifest(test_clean_clips, save_root / "clip_manifest_test_clean.json")

    save_json(
        {
            "mode": "gaze_clean_to_clean",
            "fixed_items_json": str(fixed_path),
            "train_variants": train_variants,
            "val_variants": val_variants,
            "test_variants": ["clean"],
            "n_train_videos": len(train_videos),
            "n_val_videos": len(val_videos),
            "n_test_clean_videos": len(test_clean_videos),
            "n_train_clips": len(train_clips),
            "n_val_clips": len(val_clips),
            "n_test_clean_clips": len(test_clean_clips),
        },
        save_root / "clean_to_clean_split_info.json",
    )

    return train_clips, val_clips, {"test_clean": test_clean_clips}


def train_gaze_clean_to_clean(config_path: str) -> dict:
    cfg = load_yaml(config_path)

    save_root = ensure_dir(cfg["paths"]["save_root"])
    log = get_logger("gaze_clean_to_clean", log_file=save_root / "train.log")

    log.info("=" * 80)
    log.info("[EXPERIMENT] Clean FaceMesh -> Clean Gaze Test")
    log.info("[PURPOSE] FaceMesh-based upper-bound for Gaze task")
    log.info("=" * 80)
    log.info(f"config={config_path}")

    set_seed(int(cfg["seed"]))

    device = cfg["device"] if (cfg["device"] == "cpu" or torch.cuda.is_available()) else "cpu"
    log.info(f"device={device}")

    if device == "cuda":
        torch.cuda.empty_cache()
        _log_cuda_mem("start", log)

    # Hard guards: this must remain the clean-to-clean Gaze upper-bound experiment.
    assert cfg["data"]["train_variants"] == ["clean"], cfg["data"]["train_variants"]
    assert cfg["data"]["val_variants"] == ["clean"], cfg["data"]["val_variants"]
    assert cfg["data"]["test_variants"] == ["clean"], cfg["data"]["test_variants"]

    assert cfg["loss"]["alpha_action"] == 0.0
    assert cfg["loss"]["alpha_gaze"] == 1.0
    assert cfg["loss"]["alpha_hands"] == 0.0
    assert cfg["loss"]["alpha_talk"] == 0.0

    assert cfg["best_score_weights"]["action"] == 0.0
    assert cfg["best_score_weights"]["gaze"] == 1.0
    assert cfg["best_score_weights"]["hands"] == 0.0
    assert cfg["best_score_weights"]["talk"] == 0.0

    assert not bool(cfg.get("occ", {}).get("enabled", False))
    assert cfg["model"].get("fusion", {}).get("kind") == "concat"

    save_json(cfg, save_root / "config.json")

    # ------------------------------------------------------------
    # Split clips first.
    # This stage only builds metadata lists. It should not allocate CUDA tensors.
    # ------------------------------------------------------------
    train_clips, val_clips, test_eval_clips = build_clean_only_clip_splits(
        cfg,
        save_root,
        log,
    )

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        _log_cuda_mem("after clip split", log)

    # ------------------------------------------------------------
    # IMPORTANT FIX:
    # Build and move model to CUDA BEFORE build_loaders/preload.
    #
    # In the previous version, build_loaders() ran first, then model.to(cuda)
    # failed with CUDA OOM on GB10. Model-only test showed the model is tiny
    # and can be moved to CUDA successfully, so this order avoids loader/preload
    # or unified-memory pressure interfering with model allocation.
    # ------------------------------------------------------------
    model, model_meta = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())

    log.info(
        f"model params={n_params / 1e6:.3f}M "
        f"fusion={cfg['model'].get('fusion', {}).get('kind', 'concat')} "
        f"face_mode={cfg['face']['mode']} "
        f"face_encoder={model_meta['face_encoder']} "
        f"occ_enabled={bool(cfg.get('occ', {}).get('enabled', False))}"
    )

    if device == "cuda":
        _log_cuda_mem("after model build", log)

    # ------------------------------------------------------------
    # Now build loaders.
    # ------------------------------------------------------------
    train_loader, val_loader, test_eval_loaders, train_items, val_items, test_eval_items = build_loaders(
        cfg,
        train_clips,
        val_clips,
        test_eval_clips,
        device,
        log,
    )

    _log_window_variant_counts("train", train_items, log)
    _log_window_variant_counts("val", val_items, log)
    for name, items in test_eval_items.items():
        _log_window_variant_counts(name, items, log)

    if device == "cuda":
        _log_cuda_mem("after build_loaders", log)

    criterion, optimizer, scheduler = build_train_objects(
        cfg,
        model,
        train_items,
        device,
    )

    history = []
    best_score = -1.0
    best_epoch = -1
    no_improve = 0

    epochs = int(cfg["train"]["epochs"])
    patience = int(cfg["train"]["patience"])
    grad_clip = cfg["train"].get("grad_clip_norm")
    save_every = bool(cfg["train"].get("save_every_epoch", True))
    score_weights = cfg["best_score_weights"]

    for epoch in range(1, epochs + 1):
        train_out = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            train=True,
            grad_clip_norm=grad_clip,
            epoch_idx=epoch,
            total_epochs=epochs,
            agg_mode=cfg["eval"]["clip_agg_mode"],
            topk=cfg["eval"]["clip_topk"],
            ablation_cfg={},
        )

        val_out = run_one_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            train=False,
            grad_clip_norm=None,
            epoch_idx=epoch,
            total_epochs=epochs,
            agg_mode=cfg["eval"]["clip_agg_mode"],
            topk=cfg["eval"]["clip_topk"],
            ablation_cfg={},
        )

        val_score = weighted_score(val_out, score_weights)
        scheduler.step(val_score)
        cur_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": cur_lr,
            "train_loss": float(train_out["loss"]),
            "val_loss": float(val_out["loss"]),
            "val_gaze_clip_f1_macro": float(
                val_out["heads"]["gaze"].get("clip_f1_macro", 0.0)
            ),
            "val_gaze_clip_acc": float(
                val_out["heads"]["gaze"].get("clip_acc", 0.0)
            ),
            "val_gaze_window_f1_macro": float(
                val_out["heads"]["gaze"].get("window_f1_macro", 0.0)
            ),
            "val_gaze_window_acc": float(
                val_out["heads"]["gaze"].get("window_acc", 0.0)
            ),
            "val_score": float(val_score),
        }
        history.append(row)

        pd.DataFrame(history).to_csv(
            save_root / "metrics_gaze_only.csv",
            index=False,
            encoding="utf-8-sig",
        )
        save_json(history, save_root / "history_gaze_only.json")

        log.info(
            f"[{epoch}/{epochs}] "
            f"lr={cur_lr:.2e} "
            f"train_loss={train_out['loss']:.4f} "
            f"val_loss={val_out['loss']:.4f} "
            f"| val_gaze_clip_f1={row['val_gaze_clip_f1_macro']:.4f} "
            f"val_gaze_clip_acc={row['val_gaze_clip_acc']:.4f} "
            f"| score={val_score:.4f}"
        )

        if save_every:
            save_ckpt(
                save_root / "last.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_score,
                history,
                cfg,
            )

        if val_score > best_score:
            best_score = float(val_score)
            best_epoch = int(epoch)
            no_improve = 0
            save_ckpt(
                save_root / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_score,
                history,
                cfg,
            )
            log.info(f"[best] epoch={epoch} gaze_score={best_score:.4f}")
        else:
            no_improve += 1

        if no_improve >= patience:
            log.info(f"[early stop] no improve {patience} epochs")
            break

    log.info(
        f"==== training done best_epoch={best_epoch} "
        f"best_gaze_score={best_score:.4f} ===="
    )

    best_path = save_root / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        log.info(f"[load best] {best_path}")

    test_clean_out = evaluate_test_split(
        name="test_clean",
        model=model,
        loader=test_eval_loaders["test_clean"],
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        cfg=cfg,
        ablation_cfg={},
        save_root=save_root,
        log=log,
    )

    gaze_summary = _head_metrics_for_summary(test_clean_out)["gaze"]

    summary = {
        "experiment": "gaze_clean_to_clean",
        "purpose": "Clean FaceMesh upper-bound for Gaze task",
        "best_epoch": best_epoch,
        "best_score_gaze_clip_f1": best_score,
        "test_clean_gaze": gaze_summary,
        "test_clean_loss": float(test_clean_out["loss"]),
        "n_train_clips": len(train_clips),
        "n_val_clips": len(val_clips),
        "n_test_clean_clips": len(test_eval_clips["test_clean"]),
        "n_train_windows": len(train_items),
        "n_val_windows": len(val_items),
        "n_test_clean_windows": len(test_eval_items["test_clean"]),
        "model_params": int(n_params),
        "model_meta": model_meta,
        "config_path": str(config_path),
    }

    lat_ms = measure_window_latency(
        model,
        device,
        pose_shape=(
            1,
            model_meta["pose_in_ch"],
            cfg["window"]["size"],
            cfg["pose"]["num_joints"],
        ),
        face_shape=(
            1,
            model_meta["face_in_ch"],
            cfg["window"]["size"],
            model_meta["loader_face_v"],
        ),
        occ_dim=0,
    )
    summary["inference_ms_per_window"] = float(lat_ms)

    save_json(summary, save_root / "summary_gaze_clean_to_clean.json")

    pd.DataFrame(
        [
            {
                "split": "test_clean",
                "gaze_clip_f1_macro": gaze_summary.get("clip_f1_macro", 0.0),
                "gaze_clip_acc": gaze_summary.get("clip_acc", 0.0),
                "gaze_window_f1_macro": gaze_summary.get("window_f1_macro", 0.0),
                "gaze_window_acc": gaze_summary.get("window_acc", 0.0),
                "loss": float(test_clean_out["loss"]),
                "best_epoch": best_epoch,
                "best_score": best_score,
                "n_test_windows": len(test_eval_items["test_clean"]),
            }
        ]
    ).to_csv(
        save_root / "result_gaze_clean_to_clean.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log.info("=" * 80)
    log.info("[FINAL | Clean FaceMesh -> Clean Test | Gaze only]")
    log.info(f"gaze_clip_f1_macro   = {gaze_summary.get('clip_f1_macro', 0.0):.4f}")
    log.info(f"gaze_clip_acc        = {gaze_summary.get('clip_acc', 0.0):.4f}")
    log.info(f"gaze_window_f1_macro = {gaze_summary.get('window_f1_macro', 0.0):.4f}")
    log.info(f"gaze_window_acc      = {gaze_summary.get('window_acc', 0.0):.4f}")
    log.info(f"summary -> {save_root / 'summary_gaze_clean_to_clean.json'}")
    log.info("=" * 80)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/gaze_clean_to_clean.yaml"),
    )
    args = parser.parse_args()
    train_gaze_clean_to_clean(args.config)


if __name__ == "__main__":
    main()