"""Distraction + Gaze 양쪽 annotation 전체에서 frame_shift 를 미리 추출해 JSON 으로 고정.

실행:
    python scripts/build_frame_shifts.py

결과:
    constants/frame_shifts.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dmd_paths import prefix_from_ann
from src.utils.io import load_json, save_json


def extract_shifts(ann_path: Path) -> dict[str, int] | None:
    try:
        d = load_json(ann_path)
    except Exception as e:
        print(f"[warn] JSON 로드 실패: {ann_path} | {e}", file=sys.stderr)
        return None

    root = d.get("openlabel") or d.get("vcd")
    if root is None:
        return None

    streams = root.get("streams", {})
    out: dict[str, int] = {}
    for cam in ("face", "body", "hands"):
        s = streams.get(f"{cam}_camera", {}).get("stream_properties", {}).get("sync", {})
        fs = s.get("frame_shift")
        if fs is None:
            continue
        out[cam] = int(fs)
    if "face" not in out or "body" not in out:
        return None
    return out


def build(dmd_root: Path) -> dict[str, dict[str, int]]:
    """distraction 과 gaze 양쪽 스캔.

    gaze 는 `_rgb_ann_gaze.json`, distraction 은 `_rgb_ann_distraction.json`.
    prefix 충돌 시 값이 일치해야 함 (같은 카메라 · 같은 영상이면 shift 동일).
    """
    table: dict[str, dict[str, int]] = {}

    patterns = [
        dmd_root / "distraction" / "dmd" / "*" / "*" / "*" / "*_rgb_ann_distraction.json",
        dmd_root / "gaze" / "dmd" / "*" / "*" / "*" / "*_rgb_ann_gaze.json",
    ]

    for pat in patterns:
        ann_files = sorted(Path().glob(str(pat).replace(str(dmd_root) + "/", "")))
        # glob from repo root won't work with absolute. use direct glob on absolute parts:
        # easier to just use Path.glob from a relative pattern. Re-do:
        label = "distraction" if "distraction" in str(pat) else "gaze"
        parent = dmd_root / label / "dmd"
        pattern_tail = f"*/*/*/*_rgb_ann_{label}.json"
        ann_files = sorted(parent.glob(pattern_tail))

        print(f"[scan] {label} | annotations: {len(ann_files)}")
        for ann in ann_files:
            try:
                prefix = prefix_from_ann(ann)
            except ValueError:
                continue
            shifts = extract_shifts(ann)
            if shifts is None:
                print(f"[warn] skip no-sync: {ann.name}", file=sys.stderr)
                continue

            if prefix in table:
                old = table[prefix]
                # 같은 영상이 distraction + gaze 에 모두 있으면 값 충돌 가능 → 로그
                for k in set(old) | set(shifts):
                    if old.get(k) != shifts.get(k):
                        print(f"[warn] shift conflict {prefix} {k}: {old.get(k)} vs {shifts.get(k)}", file=sys.stderr)
                old.update(shifts)
                table[prefix] = old
            else:
                table[prefix] = shifts
    return table


def diff_tables(old: dict, new: dict) -> list[str]:
    lines: list[str] = []
    ok, nk = set(old.keys()), set(new.keys())
    for k in sorted(nk - ok):
        lines.append(f"+ {k}: {new[k]}")
    for k in sorted(ok - nk):
        lines.append(f"- {k}: {old[k]}")
    for k in sorted(ok & nk):
        if old[k] != new[k]:
            lines.append(f"~ {k}: {old[k]} -> {new[k]}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dmd_root", default="/data/shared/DMD")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "constants" / "frame_shifts.json"))
    args = ap.parse_args()

    dmd_root = Path(args.dmd_root)
    out_path = Path(args.out)

    table = build(dmd_root)
    print(f"[done] total entries: {len(table)}")

    if out_path.exists():
        old = load_json(out_path)
        changes = diff_tables(old, table)
        if changes:
            print(f"[diff] {len(changes)} changes vs existing:")
            for line in changes[:30]:
                print(f"  {line}")
            if len(changes) > 30:
                print(f"  ... (+{len(changes) - 30} more)")
        else:
            print("[diff] no changes")

    save_json(table, out_path)
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
