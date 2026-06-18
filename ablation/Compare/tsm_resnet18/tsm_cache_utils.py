from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm


CACHE_TENSOR_KEYS = ("x_body", "x_face", "x_hands")


def _safe_name(split_name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(split_name))


def _tensor_to_cache_dtype(x: torch.Tensor, dtype: str) -> torch.Tensor:
    if not torch.is_tensor(x):
        return x
    if not torch.is_floating_point(x):
        return x
    if dtype == "float16":
        return x.half()
    if dtype == "bfloat16":
        return x.bfloat16()
    return x.float()


def _tensor_from_cache_dtype(x: torch.Tensor, load_as_float32: bool) -> torch.Tensor:
    if not torch.is_tensor(x):
        return x
    if load_as_float32 and torch.is_floating_point(x):
        return x.float()
    return x


class CachedSampleDataset(Dataset):
    def __init__(self, cache_dir: str | Path, load_as_float32: bool = True):
        self.cache_dir = Path(cache_dir)
        index_path = self.cache_dir / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(index_path)
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        self.files = [self.cache_dir / x for x in payload["files"]]
        self.load_as_float32 = bool(load_as_float32)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = torch.load(self.files[idx], map_location="cpu", weights_only=False)
        for key in CACHE_TENSOR_KEYS:
            if key in sample and torch.is_tensor(sample[key]):
                sample[key] = _tensor_from_cache_dtype(sample[key], self.load_as_float32)
        return sample


def build_cache_if_needed(
    dataset: Dataset,
    split_name: str,
    cache_root: str | Path,
    *,
    dtype: str = "float16",
    rebuild: bool = False,
    log_every: int = 200,
) -> Path:
    split_name = _safe_name(split_name)
    cache_dir = Path(cache_root) / split_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "index.json"

    expected_n = len(dataset)

    if index_path.exists() and not rebuild:
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            files = payload.get("files", [])
            if int(payload.get("n", -1)) == expected_n and all((cache_dir / f).exists() for f in files):
                print(f"[TSM_CACHE] reuse split={split_name} n={expected_n} dir={cache_dir}", flush=True)
                return cache_dir
        except Exception as e:
            print(f"[TSM_CACHE] index read failed, rebuild split={split_name}: {e}", flush=True)

    print(f"[TSM_CACHE] build split={split_name} n={expected_n} dir={cache_dir} dtype={dtype}", flush=True)

    files = []
    start = time.time()

    for i in tqdm(range(expected_n), desc=f"cache {split_name}", ncols=120, ascii=True):
        out_name = f"{i:08d}.pt"
        out_path = cache_dir / out_name
        tmp_path = cache_dir / f"{out_name}.tmp"

        if out_path.exists() and not rebuild:
            files.append(out_name)
            continue

        sample = dataset[i]

        # float32 영상 tensor를 half로 저장해서 디스크 사용량을 줄인다.
        for key in CACHE_TENSOR_KEYS:
            if key in sample and torch.is_tensor(sample[key]):
                sample[key] = _tensor_to_cache_dtype(sample[key].cpu(), dtype)

        torch.save(sample, tmp_path)
        os.replace(tmp_path, out_path)
        files.append(out_name)

        if (i + 1) % int(log_every) == 0:
            elapsed = time.time() - start
            print(f"[TSM_CACHE] split={split_name} cached={i+1}/{expected_n} elapsed={elapsed/60:.1f}min", flush=True)

    payload = {
        "split": split_name,
        "n": expected_n,
        "dtype": dtype,
        "files": files,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[TSM_CACHE] done split={split_name} n={expected_n} index={index_path}", flush=True)
    return cache_dir


def maybe_cache_dataset(dataset: Dataset, split_name: str, cfg: dict[str, Any]) -> Dataset:
    cache_cfg = cfg.get("cache", {}) or {}
    if not bool(cache_cfg.get("enabled", False)):
        return dataset

    cache_root = cache_cfg.get("root")
    if not cache_root:
        raise ValueError("cache.enabled=true 이면 cache.root가 필요합니다.")

    cache_dir = build_cache_if_needed(
        dataset,
        split_name,
        cache_root,
        dtype=str(cache_cfg.get("dtype", "float16")),
        rebuild=bool(cache_cfg.get("rebuild", False)),
        log_every=int(cache_cfg.get("log_every", 200)),
    )

    return CachedSampleDataset(
        cache_dir,
        load_as_float32=bool(cache_cfg.get("load_as_float32", True)),
    )
