"""DMD 파일 경로, prefix 파싱, VideoRecord — distraction + gaze s6 통합."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


_PREFIX_RE = re.compile(r"^(?P<group>g[A-Z])_(?P<subject>\d+)_(?P<session>s\d+)_(?P<timestamp>.+)$")

SourceType = Literal["distraction", "gaze"]


@dataclass
class VideoRecord:
    prefix: str
    group: str
    subject: int
    session: str
    timestamp: str
    subject_key: str
    source: SourceType
    body_npz: Path
    face_npz: Path
    face5pt_npz: Path
    ann_path: Path                            # primary annotation (distraction/gaze)
    hands_ann_path: Optional[Path] = None     # gaze s6 전용 (별도 파일일 경우)
    extras: dict = field(default_factory=dict)


def parse_prefix(prefix: str) -> dict:
    m = _PREFIX_RE.match(prefix)
    if not m:
        raise ValueError(f"prefix 파싱 실패: {prefix}")
    d = m.groupdict()
    d["subject"] = int(d["subject"])
    d["subject_key"] = f"{d['group']}_{d['subject']}"
    return d


def prefix_from_ann(ann_path: str | os.PathLike) -> str:
    name = Path(ann_path).name
    for suffix in (
        "_rgb_ann_distraction.json",
        "_rgb_ann_gaze.json",
        "_rgb_ann_hands.json",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    raise ValueError(f"annotation 파일명 규칙 불일치: {name}")


def _discover_distraction(
    pose_root: Path, face_root: Path, face5pt_root: Path, dmd_root: Path,
    sessions: Optional[list[str]], logger=None,
) -> list[VideoRecord]:
    pose = pose_root / "distraction" / "dmd"
    face = face_root / "distraction" / "dmd"
    f5pt = face5pt_root / "distraction" / "dmd"
    ann_root = dmd_root / "distraction" / "dmd"

    body_files = sorted(pose.glob("*/*/*/*_ir_body_skeleton.npz"))
    records: list[VideoRecord] = []
    miss = {"face": 0, "face5pt": 0, "ann": 0}

    for body_npz in body_files:
        rel = body_npz.relative_to(pose)
        group, subject, session = rel.parts[0], rel.parts[1], rel.parts[2]
        if sessions is not None and session not in sessions:
            continue
        prefix = body_npz.name[: -len("_ir_body_skeleton.npz")]
        face_npz = face / group / subject / session / f"{prefix}_ir_face_facemesh.npz"
        face5pt_npz = f5pt / group / subject / session / f"{prefix}_ir_face_face5pt.npz"
        ann_path = ann_root / group / subject / session / f"{prefix}_rgb_ann_distraction.json"

        missing = []
        if not face_npz.exists(): missing.append("face"); miss["face"] += 1
        if not face5pt_npz.exists(): missing.append("face5pt"); miss["face5pt"] += 1
        if not ann_path.exists(): missing.append("ann"); miss["ann"] += 1
        if missing:
            if logger is not None:
                logger.warning(f"[skip distraction] {prefix} missing={missing}")
            continue

        meta = parse_prefix(prefix)
        records.append(VideoRecord(
            prefix=prefix, group=meta["group"], subject=meta["subject"],
            session=meta["session"], timestamp=meta["timestamp"],
            subject_key=meta["subject_key"], source="distraction",
            body_npz=body_npz, face_npz=face_npz, face5pt_npz=face5pt_npz,
            ann_path=ann_path,
        ))

    if logger is not None:
        logger.info(f"distraction discovered: {len(records)} (missing_counts={miss})")
    return records


def _discover_gaze(
    pose_root: Path, face_root: Path, face5pt_root: Path, dmd_root: Path,
    sessions: Optional[list[str]], logger=None,
) -> list[VideoRecord]:
    pose = pose_root / "gaze" / "dmd"
    face = face_root / "gaze" / "dmd"
    f5pt = face5pt_root / "gaze" / "dmd"
    ann_root = dmd_root / "gaze" / "dmd"

    body_files = sorted(pose.glob("*/*/*/*_ir_body_skeleton.npz"))
    records: list[VideoRecord] = []
    miss = {"face": 0, "face5pt": 0, "ann": 0}

    for body_npz in body_files:
        rel = body_npz.relative_to(pose)
        group, subject, session = rel.parts[0], rel.parts[1], rel.parts[2]
        if sessions is not None and session not in sessions:
            continue
        prefix = body_npz.name[: -len("_ir_body_skeleton.npz")]

        face_npz = face / group / subject / session / f"{prefix}_ir_face_facemesh.npz"
        face5pt_npz = f5pt / group / subject / session / f"{prefix}_ir_face_face5pt.npz"
        ann_path = ann_root / group / subject / session / f"{prefix}_rgb_ann_gaze.json"
        hands_ann_path = ann_root / group / subject / session / f"{prefix}_rgb_ann_hands.json"

        missing = []
        if not face_npz.exists(): missing.append("face"); miss["face"] += 1
        if not face5pt_npz.exists(): missing.append("face5pt"); miss["face5pt"] += 1
        if not ann_path.exists(): missing.append("ann"); miss["ann"] += 1
        if missing:
            if logger is not None:
                logger.warning(f"[skip gaze] {prefix} missing={missing}")
            continue

        meta = parse_prefix(prefix)
        records.append(VideoRecord(
            prefix=prefix, group=meta["group"], subject=meta["subject"],
            session=meta["session"], timestamp=meta["timestamp"],
            subject_key=meta["subject_key"], source="gaze",
            body_npz=body_npz, face_npz=face_npz, face5pt_npz=face5pt_npz,
            ann_path=ann_path,
            hands_ann_path=hands_ann_path if hands_ann_path.exists() else None,
        ))

    if logger is not None:
        logger.info(f"gaze discovered: {len(records)} (missing_counts={miss})")
    return records


def discover_all(
    pose_root: str | os.PathLike,
    face_root: str | os.PathLike,
    face5pt_root: str | os.PathLike,
    dmd_root: str | os.PathLike,
    use_distraction: bool = True,
    use_gaze: bool = True,
    distraction_sessions: Optional[list[str]] = None,
    gaze_sessions: Optional[list[str]] = None,
    logger=None,
) -> list[VideoRecord]:
    pose_root = Path(pose_root); face_root = Path(face_root)
    face5pt_root = Path(face5pt_root); dmd_root = Path(dmd_root)
    out: list[VideoRecord] = []
    if use_distraction:
        out.extend(_discover_distraction(
            pose_root, face_root, face5pt_root, dmd_root,
            distraction_sessions, logger=logger,
        ))
    if use_gaze:
        out.extend(_discover_gaze(
            pose_root, face_root, face5pt_root, dmd_root,
            gaze_sessions, logger=logger,
        ))
    if logger is not None:
        logger.info(f"discover_all: total={len(out)} "
                    f"(distraction={sum(1 for r in out if r.source=='distraction')}, "
                    f"gaze={sum(1 for r in out if r.source=='gaze')})")
    return out
