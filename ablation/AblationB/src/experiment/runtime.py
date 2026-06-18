# Auto-split from Pasted code(257).py
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import torch


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
            "scheduler_state_dict": scheduler.state_dict()
            if scheduler is not None
            else None,
            "best_score": float(best_score),
            "history": history,
            "config": cfg,
        },
        path,
    )


def measure_window_latency(
    model,
    device,
    pose_shape,
    face_shape,
    occ_dim: int = 0,
    repeats: int = 30,
) -> float:
    model.eval()

    xb = torch.randn(*pose_shape, device=device)
    xf = torch.randn(*face_shape, device=device)
    xo = None
    if occ_dim > 0:
        xo = torch.randn(pose_shape[0], occ_dim, device=device)

    with torch.no_grad():
        for _ in range(5):
            model(xb, xf, x_occ=xo)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    with torch.no_grad():
        for _ in range(repeats):
            model(xb, xf, x_occ=xo)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    return float((time.perf_counter() - t0) / repeats * 1000.0)
