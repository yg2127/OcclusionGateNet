"""constants/frame_shifts.json 로더 + mosaic → camera frame 변환 유틸."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.utils.io import load_json


class FrameShiftTable:
    def __init__(self, table: dict[str, dict[str, int]]):
        self._t = table

    def __contains__(self, prefix: str) -> bool:
        return prefix in self._t

    def __len__(self) -> int:
        return len(self._t)

    def get(self, prefix: str) -> dict[str, int]:
        if prefix not in self._t:
            raise KeyError(f"frame_shifts 테이블에 없는 prefix: {prefix}")
        return self._t[prefix]

    def shift(self, prefix: str, camera: str) -> int:
        s = self.get(prefix)
        if camera not in s:
            raise KeyError(f"{prefix}에 camera={camera} shift 없음, 보유: {list(s.keys())}")
        return int(s[camera])

    def to_body_frame(self, mosaic_frame: int, prefix: str) -> int:
        return mosaic_frame + self.shift(prefix, "body")

    def to_face_frame(self, mosaic_frame: int, prefix: str) -> int:
        return mosaic_frame + self.shift(prefix, "face")


_cache: dict[str, FrameShiftTable] = {}


def load_frame_shifts(path: str | Path, cache: bool = True) -> FrameShiftTable:
    key = str(Path(path).resolve())
    if cache and key in _cache:
        return _cache[key]
    table = load_json(path)
    if not isinstance(table, dict):
        raise ValueError(f"frame_shifts.json 형식 이상: {path}")
    fst = FrameShiftTable(table)
    if cache:
        _cache[key] = fst
    return fst
