"""Multi-task ClipRecord 빌더.

Action (11-class) + Gaze (10-class fine + binary weak) + Hands (4-class) + Talk (binary).

- distraction → action · gaze_weak · hands · talk 라벨, gaze_fine 은 None
- gaze       → gaze_fine · hands 라벨, action/gaze_weak/talk 은 None
"""
from __future__ import annotations

import random
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.data.annotation import collect_intervals, load_openlabel
from src.data.dmd_paths import VideoRecord
from src.data.frame_shifts import FrameShiftTable
from constants.gaze_zones import (GAZE_WEAK_FRONT, GAZE_WEAK_NOT_FRONT,
                                  raw_label_to_id)


# ---------- Action head 라벨 ----------
# safe_drive 에 흡수되는 원본 driver_actions
SAFE_DRIVE_MERGED = {
    "driver_actions/safe_drive",
    "driver_actions/standstill_or_waiting",
    "driver_actions/change_gear",
}

# 학습에서 완전 drop (clip 생성 안 함)
DROPPED_ACTION = {
    "driver_actions/unclassified",
}

ACTION_CLASSES = [
    "safe_drive",              # 0
    "texting_right",           # 1
    "texting_left",            # 2
    "phonecall_right",         # 3
    "phonecall_left",          # 4
    "radio",                   # 5
    "drinking",                # 6
    "reach_side",              # 7
    "reach_backseat",          # 8
    "hair_and_makeup",         # 9
    "talking_to_passenger",    # 10
]

ACTION_TO_ID = {name: i for i, name in enumerate(ACTION_CLASSES)}
NUM_ACTION_CLASSES = len(ACTION_CLASSES)
SAFE_DRIVE_ID = ACTION_TO_ID["safe_drive"]


def action_raw_to_id(raw_label: str) -> int | None:
    """raw `driver_actions/*` → action class id. None 이면 drop (unclassified 등)."""
    if raw_label in DROPPED_ACTION:
        return None
    if raw_label in SAFE_DRIVE_MERGED:
        return SAFE_DRIVE_ID
    if raw_label.startswith("driver_actions/"):
        short = raw_label.split("/", 1)[1]
        return ACTION_TO_ID.get(short)
    return None


# ---------- Hands head 라벨 ----------
HANDS_CLASSES = ["both", "only_left", "only_right", "none"]
HANDS_TO_ID = {name: i for i, name in enumerate(HANDS_CLASSES)}
NUM_HANDS_CLASSES = len(HANDS_CLASSES)


def hands_raw_to_id(raw_label: str) -> int | None:
    """`hands_using_wheel/both` / `hands_on_wheel/both` 둘 다 허용."""
    if "/" not in raw_label:
        return None
    cat, short = raw_label.split("/", 1)
    if cat not in ("hands_using_wheel", "hands_on_wheel"):
        return None
    return HANDS_TO_ID.get(short)


# ---------- Talking head 라벨 ----------
NUM_TALK_CLASSES = 2


# =========================================================
# ClipRecord
# =========================================================
@dataclass
class ClipLabels:
    """각 head 에 대한 정수 라벨 또는 None (= 지도 없음)."""
    action: Optional[int] = None           # 0~10 or None
    gaze_fine: Optional[int] = None        # 0~9 or None
    gaze_weak: Optional[int] = None        # 0 or 1 (distraction only)
    hands: Optional[int] = None            # 0~3 or None
    talk: Optional[int] = None             # 0 or 1 (distraction only)


@dataclass
class ClipRecord:
    clip_id: str
    subject_key: str
    source: str                            # "distraction" | "gaze"
    video_prefix: str
    body_npz: str
    face_npz: str
    face5pt_npz: str
    mosaic_start: int
    mosaic_end: int
    body_start: int
    body_end: int
    face_start: int
    face_end: int
    labels: ClipLabels

    def as_dict(self) -> dict:
        d = asdict(self)
        return d


# =========================================================
# Helpers
# =========================================================
def _read_npz_total(npz_path: Path, key_hint: str = "detected") -> int:
    with np.load(npz_path, allow_pickle=True) as d:
        for k in (key_hint, "keypoints", "landmarks", "detected"):
            if k in d.files:
                return int(d[k].shape[0])
    raise ValueError(f"시간축 키 없음: {npz_path}")


def _slide(start: int, end: int, clip_len: int, stride: int) -> list[tuple[int, int]]:
    if end - start + 1 < clip_len:
        return []
    out: list[tuple[int, int]] = []
    s = start
    while s + clip_len - 1 <= end:
        out.append((s, s + clip_len - 1))
        s += stride
    last = (end - clip_len + 1, end)
    if not out or out[-1] != last:
        out.append(last)
    return out


def _interval_overlap(a_start: int, a_end: int, intervals: list[tuple[int, int]]) -> bool:
    """[a_start, a_end] 와 intervals 중 하나라도 겹치면 True."""
    for s, e in intervals:
        if not (a_end < s or e < a_start):
            return True
    return False


def _majority_label_in_range(
    start: int, end: int,
    type_intervals: dict[str, list[tuple[int, int]]],
) -> str | None:
    """[start, end] 구간에서 가장 오래 active 된 type 반환. 없으면 None.

    DMD 는 한 카테고리 내 mutually exclusive 이므로 같은 카테고리의 type 들만 전달할 것.
    """
    best_type, best_cov = None, 0
    for t, intervals in type_intervals.items():
        cov = 0
        for s, e in intervals:
            os_, oe_ = max(s, start), min(e, end)
            if oe_ >= os_:
                cov += oe_ - os_ + 1
        if cov > best_cov:
            best_cov = cov
            best_type = t
    return best_type if best_cov > 0 else None


def _any_overlap(start: int, end: int, intervals: list[tuple[int, int]]) -> bool:
    for s, e in intervals:
        if not (end < s or e < start):
            return True
    return False

def _get_variant(v: VideoRecord) -> str:
    """Return clean/masked/orig variant label stored in VideoRecord.

    The fixed clean/masked manifest pipeline attaches the variant to either
    v.extras["variant"] or v.variant. We inject this variant into clip_id so
    downstream logs/evaluation can separate clean and masked clips/windows.
    """
    extras = getattr(v, "extras", None)
    if isinstance(extras, dict) and extras.get("variant"):
        return str(extras["variant"])

    variant = getattr(v, "variant", None)
    if variant:
        return str(variant)

    return "orig"


# =========================================================
# Distraction clip builder
# =========================================================
def _build_distraction_clips(
    v: VideoRecord,
    shifts: FrameShiftTable,
    clip_len: int,
    action_stride: int,
    normal_ratio: float,
    rng: random.Random,
    logger=None,
) -> tuple[list[ClipRecord], dict]:
    stats = {
        "action_clips": 0, "safe_drive_clips": 0,
        "dropped_unclassified": 0, "out_of_range": 0,
    }
    try:
        sv = shifts.get(v.prefix)
    except KeyError:
        if logger is not None: logger.warning(f"[skip distraction no-shift] {v.prefix}")
        return [], stats

    body_shift = int(sv["body"])
    face_shift = int(sv["face"])
    T_body = _read_npz_total(v.body_npz)
    T_face = _read_npz_total(v.face_npz)

    try:
        ann_root = load_openlabel(v.ann_path)
    except Exception as e:
        if logger is not None: logger.warning(f"[distraction ann fail] {v.prefix}: {e}")
        return [], stats

    intervals = collect_intervals(ann_root)

    # 각 카테고리별 프레임 매핑
    action_intervals = {t: iv for t, iv in intervals.items()
                        if t.startswith("driver_actions/") and t not in DROPPED_ACTION}
    # unclassified 는 따로 기록 (drop 집계)
    unclassified_intervals = intervals.get("driver_actions/unclassified", [])

    gaze_looking_intervals = intervals.get("gaze_on_road/looking_road", [])
    gaze_notlooking_intervals = intervals.get("gaze_on_road/not_looking_road", [])

    hands_intervals = {t: iv for t, iv in intervals.items() if t.startswith("hands_using_wheel/")}

    talk_intervals = intervals.get("talking/talking", [])

    # 각 action 인터벌을 슬라이드해 clip 후보 생성
    # safe_drive (및 merge 된 것) 는 normal_ratio 로 다운샘플
    normal_type_set = SAFE_DRIVE_MERGED
    abnormal_candidates: list[tuple[int, int, int]] = []   # (start, end, action_id)
    normal_candidates: list[tuple[int, int, int]] = []

    for raw_label, ivs in action_intervals.items():
        aid = action_raw_to_id(raw_label)
        if aid is None:
            continue
        is_normal = raw_label in normal_type_set
        for s, e in ivs:
            for cs, ce in _slide(s, e, clip_len, action_stride):
                (normal_candidates if is_normal else abnormal_candidates).append((cs, ce, aid))

    if normal_candidates and normal_ratio < 1.0:
        k = max(1, int(round(len(normal_candidates) * normal_ratio)))
        normal_candidates = rng.sample(normal_candidates, k=min(k, len(normal_candidates)))

    # unclassified 구간에 완전히 들어가는 clip 은 drop
    candidates = abnormal_candidates + normal_candidates
    candidates.sort(key=lambda x: (x[0], x[1]))

    records: list[ClipRecord] = []
    local_idx = 0
    variant = _get_variant(v)

    for cs, ce, aid in candidates:
        # unclassified 과 겹치는지 — 부분 겹침은 통과 (해당 clip 주요 라벨은 이미 aid 로 결정)
        # 그러나 unclassified 만 존재하는 clip 은 aid 가 애초에 할당 안 됐을 테니 여기 안 들어옴
        body_start = cs + body_shift
        body_end = ce + body_shift
        face_start = cs + face_shift
        face_end = ce + face_shift
        if body_start < 0 or body_end >= T_body or face_start < 0 or face_end >= T_face:
            stats["out_of_range"] += 1
            continue

        # Gaze weak: clip 구간에서 looking 과 not_looking 어느 쪽이 더 길게 cover 하나
        look_cov = sum(max(0, min(e, ce) - max(s, cs) + 1) for s, e in gaze_looking_intervals
                        if not (e < cs or ce < s))
        notlook_cov = sum(max(0, min(e, ce) - max(s, cs) + 1) for s, e in gaze_notlooking_intervals
                          if not (e < cs or ce < s))
        if look_cov == 0 and notlook_cov == 0:
            gaze_weak = None
        else:
            gaze_weak = GAZE_WEAK_FRONT if look_cov >= notlook_cov else GAZE_WEAK_NOT_FRONT

        # Hands: 카테고리 내 가장 긴 type
        hands_type = _majority_label_in_range(cs, ce, hands_intervals)
        hands_id = hands_raw_to_id(hands_type) if hands_type else None

        # Talk: talking/talking 과 겹치면 1, 아니면 0
        talk = 1 if _any_overlap(cs, ce, talk_intervals) else 0

        records.append(ClipRecord(
            clip_id=f"{v.prefix}__{variant}__dist__{local_idx:05d}",
            subject_key=v.subject_key, source="distraction",
            video_prefix=v.prefix,
            body_npz=str(v.body_npz), face_npz=str(v.face_npz), face5pt_npz=str(v.face5pt_npz),
            mosaic_start=cs, mosaic_end=ce,
            body_start=body_start, body_end=body_end,
            face_start=face_start, face_end=face_end,
            labels=ClipLabels(
                action=aid, gaze_fine=None, gaze_weak=gaze_weak,
                hands=hands_id, talk=talk,
            ),
        ))
        local_idx += 1
        stats["action_clips"] += 1
        if aid == SAFE_DRIVE_ID:
            stats["safe_drive_clips"] += 1

    stats["dropped_unclassified"] = len(unclassified_intervals)
    return records, stats


# =========================================================
# Gaze clip builder
# =========================================================
def _build_gaze_clips(
    v: VideoRecord,
    shifts: FrameShiftTable,
    clip_len: int,
    stride: int,
    rng: random.Random,
    logger=None,
) -> tuple[list[ClipRecord], dict]:
    stats = {"gaze_clips": 0, "out_of_range": 0}
    try:
        sv = shifts.get(v.prefix)
    except KeyError:
        if logger is not None: logger.warning(f"[skip gaze no-shift] {v.prefix}")
        return [], stats

    body_shift = int(sv["body"])
    face_shift = int(sv["face"])
    T_body = _read_npz_total(v.body_npz)
    T_face = _read_npz_total(v.face_npz)

    try:
        ann_root = load_openlabel(v.ann_path)
    except Exception as e:
        if logger is not None: logger.warning(f"[gaze ann fail] {v.prefix}: {e}")
        return [], stats

    gaze_intervals_raw = collect_intervals(ann_root)

    gaze_zone_intervals = {t: iv for t, iv in gaze_intervals_raw.items()
                           if t.startswith("gaze_zone/")}
    # Hands 는 별도 파일 가능
    if v.hands_ann_path is not None:
        try:
            hands_ann = load_openlabel(v.hands_ann_path)
            hands_raw = collect_intervals(hands_ann)
        except Exception as e:
            if logger is not None: logger.warning(f"[gaze hands ann fail] {v.prefix}: {e}")
            hands_raw = {}
    else:
        hands_raw = gaze_intervals_raw

    hands_intervals = {t: iv for t, iv in hands_raw.items() if t.startswith("hands_on_wheel/")}

    # Gaze zone interval 별 슬라이드
    records: list[ClipRecord] = []
    local_idx = 0
    variant = _get_variant(v)

    for raw_label, ivs in gaze_zone_intervals.items():
        zone_id = raw_label_to_id(raw_label)
        if zone_id is None:
            continue
        for s, e in ivs:
            for cs, ce in _slide(s, e, clip_len, stride):
                body_start = cs + body_shift
                body_end = ce + body_shift
                face_start = cs + face_shift
                face_end = ce + face_shift
                if body_start < 0 or body_end >= T_body or face_start < 0 or face_end >= T_face:
                    stats["out_of_range"] += 1
                    continue

                # Hands majority
                hands_type = _majority_label_in_range(cs, ce, hands_intervals)
                hands_id = hands_raw_to_id(hands_type) if hands_type else None

                records.append(ClipRecord(
                    clip_id=f"{v.prefix}__{variant}__gaze__{local_idx:05d}",
                    subject_key=v.subject_key, source="gaze",
                    video_prefix=v.prefix,
                    body_npz=str(v.body_npz), face_npz=str(v.face_npz), face5pt_npz=str(v.face5pt_npz),
                    mosaic_start=cs, mosaic_end=ce,
                    body_start=body_start, body_end=body_end,
                    face_start=face_start, face_end=face_end,
                    labels=ClipLabels(
                        action=None, gaze_fine=zone_id, gaze_weak=None,
                        hands=hands_id, talk=None,
                    ),
                ))
                local_idx += 1
                stats["gaze_clips"] += 1
    return records, stats


def build_all_clips(
    videos: list[VideoRecord],
    shifts: FrameShiftTable,
    clip_len: int = 50,
    action_stride: int = 25,
    gaze_stride: int = 25,
    normal_ratio: float = 0.3,
    seed: int = 42,
    logger=None,
) -> list[ClipRecord]:
    rng = random.Random(seed)
    all_records: list[ClipRecord] = []
    totals = {"dist": 0, "gaze": 0, "out_of_range": 0, "dropped_uncl": 0}

    for v in videos:
        if v.source == "distraction":
            recs, st = _build_distraction_clips(
                v, shifts, clip_len, action_stride, normal_ratio, rng, logger=logger,
            )
            totals["dist"] += len(recs)
            totals["out_of_range"] += st["out_of_range"]
            totals["dropped_uncl"] += st["dropped_unclassified"]
        else:
            recs, st = _build_gaze_clips(
                v, shifts, clip_len, gaze_stride, rng, logger=logger,
            )
            totals["gaze"] += len(recs)
            totals["out_of_range"] += st["out_of_range"]
        all_records.extend(recs)

    if logger is not None:
        logger.info(f"clips built: distraction={totals['dist']} gaze={totals['gaze']} "
                    f"out_of_range={totals['out_of_range']} dropped_unclassified={totals['dropped_uncl']}")
        # 클래스 분포 요약
        from collections import Counter
        ac = Counter(r.labels.action for r in all_records if r.labels.action is not None)
        gc = Counter(r.labels.gaze_fine for r in all_records if r.labels.gaze_fine is not None)
        hc = Counter(r.labels.hands for r in all_records if r.labels.hands is not None)
        tc = Counter(r.labels.talk for r in all_records if r.labels.talk is not None)
        logger.info(f"  action dist: {dict(sorted(ac.items()))}")
        logger.info(f"  gaze_fine:   {dict(sorted(gc.items()))}")
        logger.info(f"  hands:       {dict(sorted(hc.items()))}")
        logger.info(f"  talk:        {dict(sorted(tc.items()))}")

    return all_records


def save_clip_manifest(records: list[ClipRecord], out_path: str | Path) -> None:
    from src.utils.io import save_json
    save_json([r.as_dict() for r in records], out_path)
