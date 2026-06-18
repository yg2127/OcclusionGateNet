"""OpenLABEL annotation 로더 + multi-task 라벨 추출 (distraction + gaze)."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterator

from src.utils.io import load_json


def load_openlabel(path: str | Path) -> dict:
    d = load_json(path)
    root = d.get("openlabel") or d.get("vcd")
    if root is None:
        raise ValueError(f"openlabel/vcd 루트 키 없음: {path}")
    return root


def iter_action_intervals_by_type(ann_root: dict, prefix_filter: str | None = None) -> Iterator[tuple[str, int, int]]:
    """annotation 의 모든 action 을 순회해 (action_type, frame_start, frame_end) yield.

    prefix_filter 가 주어지면 해당 prefix 로 시작하는 action 만 yield.
    프레임 인덱스는 mosaic 기준 (face 카메라).
    """
    actions = ann_root.get("actions", {})
    for aid, a in actions.items():
        t = a.get("type", "").strip()
        if not t:
            continue
        if prefix_filter is not None and not t.startswith(prefix_filter):
            continue
        for fi in a.get("frame_intervals", []):
            fs = fi.get("frame_start")
            fe = fi.get("frame_end")
            if fs is None or fe is None:
                continue
            yield t, int(fs), int(fe)


def collect_intervals(ann_root: dict) -> dict[str, list[tuple[int, int]]]:
    """action_type → list of (frame_start, frame_end) 딕트. 전체 action 수집."""
    out: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for t, fs, fe in iter_action_intervals_by_type(ann_root):
        out[t].append((fs, fe))
    return dict(out)


def total_frames_annotated(ann_root: dict) -> int | None:
    """최대 프레임 인덱스 + 1 추정. frame_intervals 의 최대값을 스캔."""
    max_fe = 0
    for _, intervals in collect_intervals(ann_root).items():
        for _, fe in intervals:
            if fe > max_fe:
                max_fe = fe
    return max_fe + 1 if max_fe > 0 else None
