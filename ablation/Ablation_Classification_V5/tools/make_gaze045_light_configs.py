#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate light V5 configs for gaze/action-balanced training.

This script reads the existing V5 yaml configs and writes modified configs to:
  <root>/configs_gaze045_light/*.yaml

Main changes:
  loss.alpha_action = 0.45
  loss.alpha_gaze   = 0.45
  loss.alpha_hands  = 0.05
  loss.alpha_talk   = 0.05

  best_score_weights uses the same 0.45/0.45/0.05/0.05 weights.

  train is lighter than the original:
    epochs = 20
    patience = 4
    save_every_epoch = false
    resume = false

The original configs are not overwritten.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from copy import deepcopy

try:
    import yaml
except ImportError as e:
    raise SystemExit("PyYAML is required. Try: pip install pyyaml") from e


DEFAULT_CONFIGS = [
    "v5_no_occ_original_mediapipe_seed42.yaml",
    "v5_task_gated_late.yaml",
    "v5_task_region_gated_late.yaml",
    "v5_task_region_scalar_gated_late.yaml",
    "v5_explicit_region_mask_gate.yaml",
    "v5_explicit_region_scalar_mask_gate.yaml",
    "v5_occ_token_region_transformer.yaml",
    "v5_occ_attention_bias.yaml",
]


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def update_config(cfg: dict, root: Path, src_name: str, args: argparse.Namespace) -> dict:
    cfg = deepcopy(cfg)

    stem = Path(src_name).stem
    exp_name = f"{stem}_gaze045_light"

    cfg["seed"] = int(args.seed)
    cfg["device"] = args.device

    cfg.setdefault("paths", {})
    cfg["paths"]["results_root"] = str(root / "results_gaze045_light")
    cfg["paths"]["save_root"] = str(root / "artifacts_gaze045_light" / exp_name)

    # Keep data roots as they are, but make project-local frame_shifts robust after copying code.
    if (root / "constants" / "frame_shifts.json").exists():
        cfg["paths"]["frame_shifts"] = str(root / "constants" / "frame_shifts.json")

    cfg.setdefault("train", {})
    cfg["train"]["epochs"] = int(args.epochs)
    cfg["train"]["patience"] = int(args.patience)
    cfg["train"]["save_every_epoch"] = bool(args.save_every_epoch)
    cfg["train"]["resume"] = False

    if args.batch_size is not None:
        cfg["train"]["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        cfg["train"]["num_workers"] = int(args.num_workers)
    if args.lr is not None:
        cfg["train"]["lr"] = float(args.lr)

    cfg.setdefault("loss", {})
    cfg["loss"]["alpha_action"] = float(args.action_weight)
    cfg["loss"]["alpha_gaze"] = float(args.gaze_weight)
    cfg["loss"]["alpha_hands"] = float(args.hands_weight)
    cfg["loss"]["alpha_talk"] = float(args.talk_weight)
    cfg["loss"]["gaze_weak_weight"] = float(args.gaze_weak_weight)

    cfg["best_score_weights"] = {
        "action": float(args.action_weight),
        "gaze": float(args.gaze_weight),
        "hands": float(args.hands_weight),
        "talk": float(args.talk_weight),
    }

    cfg.setdefault("notify", {})
    cfg["notify"]["enabled"] = False
    cfg["notify"]["tag"] = exp_name

    # Add explicit experiment metadata for later checking.
    cfg["gaze045_light_experiment"] = {
        "source_config": src_name,
        "experiment_name": exp_name,
        "purpose": "action/gaze-balanced lightweight V5 retraining",
        "loss_weights": cfg["loss"],
        "best_score_weights": cfg["best_score_weights"],
        "light_train": {
            "epochs": cfg["train"].get("epochs"),
            "patience": cfg["train"].get("patience"),
            "save_every_epoch": cfg["train"].get("save_every_epoch"),
            "batch_size": cfg["train"].get("batch_size"),
            "num_workers": cfg["train"].get("num_workers"),
            "lr": cfg["train"].get("lr"),
        },
    }

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/data/shared/scuppy/hyi/Ablation/Ablation_Classification_V5")
    parser.add_argument("--src_config_dir", type=str, default=None)
    parser.add_argument("--out_config_dir", type=str, default=None)
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--action_weight", type=float, default=0.45)
    parser.add_argument("--gaze_weight", type=float, default=0.45)
    parser.add_argument("--hands_weight", type=float, default=0.05)
    parser.add_argument("--talk_weight", type=float, default=0.05)
    parser.add_argument("--gaze_weak_weight", type=float, default=0.0)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=None, help="Default: keep source config value")
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=None, help="Default: keep source config value")
    parser.add_argument("--save_every_epoch", action="store_true", help="Default false. If passed, saves every epoch.")

    args = parser.parse_args()

    root = Path(args.root).resolve()
    src_dir = Path(args.src_config_dir).resolve() if args.src_config_dir else root / "configs"
    out_dir = Path(args.out_config_dir).resolve() if args.out_config_dir else root / "configs_gaze045_light"

    if not src_dir.exists():
        raise FileNotFoundError(f"Source config dir not found: {src_dir}")

    print(f"[INFO] root          = {root}")
    print(f"[INFO] src_config_dir= {src_dir}")
    print(f"[INFO] out_config_dir= {out_dir}")
    print(f"[INFO] weights       = action={args.action_weight} gaze={args.gaze_weight} hands={args.hands_weight} talk={args.talk_weight}")
    print(f"[INFO] light train   = epochs={args.epochs} patience={args.patience} num_workers={args.num_workers} save_every_epoch={args.save_every_epoch}")

    written = []
    for name in args.configs:
        src = src_dir / name
        if not src.exists():
            print(f"[WARN] missing config, skip: {src}")
            continue
        cfg = load_yaml(src)
        new_cfg = update_config(cfg, root=root, src_name=name, args=args)
        dst = out_dir / name.replace(".yaml", "_gaze045_light.yaml")
        save_yaml(new_cfg, dst)
        written.append(dst)
        print(f"[WRITE] {dst}")

    if not written:
        raise RuntimeError("No configs were written.")

    list_path = out_dir / "config_list.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for p in written:
            f.write(str(p.relative_to(root)) + "\n")
    print(f"[WRITE] {list_path}")
    print("[DONE]")


if __name__ == "__main__":
    main()
