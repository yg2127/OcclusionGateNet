#!/usr/bin/env python3
"""
Full DMS warning system runner.

Adds real warning logic on top of FullDMSSystem:
  - sustained abnormal behavior detection
  - visual warning overlay
  - JSONL warning event logging
  - edge-tts based embedded audio warning

Expected project layout:
  /data/shared/scuppy/Full_System
    full_dms_system/full_system.py
    configs/full_dms_config_template.yaml

Run example:
  cd /data/shared/scuppy/Full_System
  python scripts/run_full_system_warning_overlay.py \
    --config configs/full_dms_config_template.yaml \
    --face-video /path/to/ir_face.mp4 \
    --body-video /path/to/ir_body.mp4 \
    --out-video outputs/warning_overlay.mp4 \
    --out-jsonl outputs/warning_predictions.jsonl \
    --out-events outputs/warning_events.jsonl

Embedded TTS audio:
  pip install edge-tts
  python scripts/run_full_system_warning_overlay_edge_tts_hardcoded.py

Notes:
  - This version uses edge-tts only for embedded audio generation. No pyttsx3/espeak/beep fallback.
  - If your classifier label ids differ, edit ACTION_LABELS / GAZE_LABELS / HANDS_LABELS / TALK_LABELS.
"""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import cv2
import numpy as np
import yaml

# ============================================================
# Hardcoded runtime settings
# Edit only this block for your demo.
# ============================================================

HARDCODED_ROOT = Path("/data/shared/scuppy/Full_System")
HARDCODED_CONFIG = HARDCODED_ROOT / "configs/full_dms_config_template.yaml"

# TODO: set these two paths to the face/body videos you want to demo.
# Example format:
# HARDCODED_FACE_VIDEO = Path("/data/shared/.../..._ir_face.mp4")
# HARDCODED_BODY_VIDEO = Path("/data/shared/.../..._ir_body.mp4")
HARDCODED_FACE_VIDEO = Path("/data/shared/scuppy/TEST/record_camera_raw_20260602_232944.mp4")
HARDCODED_BODY_VIDEO = Path("/data/shared/scuppy/TEST/pose_camera_raw_20260602_232944.mp4")

HARDCODED_OUT_VIDEO = HARDCODED_ROOT / "outputs/full_warning_overlay.mp4"
HARDCODED_OUT_JSONL = HARDCODED_ROOT / "outputs/full_warning_predictions.jsonl"
HARDCODED_OUT_EVENTS = HARDCODED_ROOT / "outputs/full_warning_events.jsonl"
HARDCODED_OUT_VIDEO_AUDIO = HARDCODED_ROOT / "outputs/full_warning_overlay_with_audio.mp4"
HARDCODED_TTS_WORK_DIR = HARDCODED_ROOT / "outputs/tts_work"

# Edge TTS voice. Alternatives: ko-KR-InJoonNeural, ko-KR-SunHiNeural
HARDCODED_EDGE_VOICE = "ko-KR-SunHiNeural"
HARDCODED_WARNING_LANG = "ko"

# Presentation-friendly thresholds. Increase for more conservative warnings.
HARDCODED_ACTION_SECONDS = 4.0
HARDCODED_GAZE_SECONDS = 4.0
HARDCODED_HANDS_SECONDS = 4.0
HARDCODED_TALK_SECONDS = 4.0

# Set None for full video. Use 1000 for a quick test.
HARDCODED_MAX_FRAMES = None

# Always embed edge-tts audio into final mp4 in this hardcoded version.
HARDCODED_EMBED_TTS_AUDIO = True
HARDCODED_DISPLAY = False
HARDCODED_NO_POSE_OVERLAY = False
HARDCODED_MAX_TTS_EVENTS = 40


# ============================================================
# Label maps. Change only here if your DMS label order differs.
# ============================================================

ACTION_LABELS = {
    0: "safe_drive",
    1: "texting_right",
    2: "texting_left",
    3: "phonecall_right",
    4: "phonecall_left",
    5: "radio",
    6: "drinking",
    7: "reach_side",
    8: "reach_backseat",
    9: "hair_and_makeup",
    10: "talking_to_passenger",
}

GAZE_LABELS = {
    0: "left_mirror",
    1: "left",
    2: "front",
    3: "center_mirror",
    4: "front_right",
    5: "right_mirror",
    6: "right",
    7: "infotainment",
    8: "steering_wheel",
}

HANDS_LABELS = {
    0: "both",
    1: "only_left",
    2: "only_right",
    3: "none",
}

TALK_LABELS = {
    0: "no_talk",
    1: "talk",
}

TASKS = ["action", "gaze", "hands", "talk"]
TASK_LABELS = {
    "action": ACTION_LABELS,
    "gaze": GAZE_LABELS,
    "hands": HANDS_LABELS,
    "talk": TALK_LABELS,
}

# Behavior groups. Tune these for your presentation/demo policy.
DANGEROUS_ACTIONS = {
    "texting_right",
    "texting_left",
    "phonecall_right",
    "phonecall_left",
    "drinking",
    "reach_side",
    "reach_backseat",
    "hair_and_makeup",
}

# Gaze labels that indicate attention is not on the road.
OFF_ROAD_GAZE = {
    "left_mirror",
    "left",
    "center_mirror",
    "front_right",
    "right_mirror",
    "right",
    "infotainment",
    "steering_wheel",
}

UNSAFE_HANDS = {"only_left", "only_right", "none"}


# ============================================================
# Utility
# ============================================================

def id_to_label(task: str, class_id: Any) -> str:
    try:
        class_id = int(class_id)
    except Exception:
        return str(class_id)
    return TASK_LABELS.get(task, {}).get(class_id, f"class_{class_id}")


def softmax_margin(prob: Optional[List[float]]) -> Optional[float]:
    if prob is None:
        return None
    arr = np.asarray(prob, dtype=np.float32).reshape(-1)
    if arr.size < 2:
        return None
    top2 = np.sort(arr)[-2:]
    return float(top2[-1] - top2[-2])


def parse_preds(raw_pred: Any) -> Dict[str, dict]:
    """Normalize FullDMSSystem prediction into task -> {class_id,label,confidence,prob,margin}."""
    if raw_pred is None or not isinstance(raw_pred, dict):
        return {}

    pred = raw_pred
    for wrapper in ["predictions", "tasks", "outputs"]:
        if wrapper in pred and isinstance(pred[wrapper], dict):
            pred = pred[wrapper]
            break

    out: Dict[str, dict] = {}

    for task in TASKS:
        if task not in pred:
            continue
        v = pred[task]

        class_id = None
        conf = None
        prob = None

        if isinstance(v, dict):
            for k in ["pred", "class_id", "pred_id", "argmax", "index", "cls"]:
                if k in v and v[k] is not None:
                    try:
                        class_id = int(v[k])
                        break
                    except Exception:
                        pass

            for k in ["confidence", "score", "max_prob"]:
                if k in v and v[k] is not None:
                    try:
                        conf = float(v[k])
                        break
                    except Exception:
                        pass

            if "prob" in v and v["prob"] is not None:
                try:
                    arr = np.asarray(v["prob"], dtype=np.float32).reshape(-1)
                    prob = [float(x) for x in arr]
                    if class_id is None and arr.size > 0:
                        class_id = int(np.argmax(arr))
                    if conf is None and class_id is not None and 0 <= class_id < arr.size:
                        conf = float(arr[class_id])
                except Exception:
                    prob = None

        elif isinstance(v, (list, tuple, np.ndarray)):
            arr = np.asarray(v, dtype=np.float32).reshape(-1)
            if arr.size > 0:
                class_id = int(np.argmax(arr))
                conf = float(arr[class_id])
                prob = [float(x) for x in arr]

        else:
            # Already a plain label-like object.
            out[task] = {
                "class_id": None,
                "label": str(v),
                "confidence": None,
                "prob": None,
                "margin": None,
            }
            continue

        if class_id is None:
            continue

        out[task] = {
            "class_id": int(class_id),
            "label": id_to_label(task, class_id),
            "confidence": conf,
            "prob": prob,
            "margin": softmax_margin(prob),
        }

    return out


def resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / max(h, 1)
    return cv2.resize(img, (int(round(w * scale)), target_h), interpolation=cv2.INTER_AREA)


def put_text_box(
    img: np.ndarray,
    lines: List[str],
    x: int,
    y: int,
    font_scale: float = 0.58,
    thickness: int = 1,
    fg: Tuple[int, int, int] = (255, 255, 255),
    bg: Tuple[int, int, int] = (0, 0, 0),
    alpha: float = 0.62,
    line_h: int = 24,
    pad: int = 10,
) -> None:
    if not lines:
        return
    max_w = 0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        max_w = max(max_w, tw)
    box_w = max_w + pad * 2
    box_h = line_h * len(lines) + pad * 2

    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), bg, -1)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, dst=img)

    for i, line in enumerate(lines):
        cv2.putText(
            img,
            line,
            (x + pad, y + pad + 17 + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            fg,
            thickness,
            cv2.LINE_AA,
        )


# ============================================================
# Alert rules
# ============================================================

@dataclass
class RuleConfig:
    min_seconds: float
    cooldown_seconds: float
    min_confidence: float
    severity: str
    message_ko: str
    message_en: str


DEFAULT_RULES: Dict[str, RuleConfig] = {
    "dangerous_action": RuleConfig(
        min_seconds=2.0,
        cooldown_seconds=5.0,
        min_confidence=0.45,
        severity="HIGH",
        message_ko="위험 행동이 감지되었습니다. 전방 주시에 집중하세요.",
        message_en="Dangerous driving behavior detected. Please focus on the road.",
    ),
    "off_road_gaze": RuleConfig(
        min_seconds=1.5,
        cooldown_seconds=4.0,
        min_confidence=0.40,
        severity="MEDIUM",
        message_ko="시선이 전방에서 벗어났습니다. 전방을 주시하세요.",
        message_en="Your gaze is away from the road. Please look ahead.",
    ),
    "unsafe_hands": RuleConfig(
        min_seconds=2.5,
        cooldown_seconds=5.0,
        min_confidence=0.40,
        severity="MEDIUM",
        message_ko="핸들 파지가 불안정합니다. 양손으로 운전하세요.",
        message_en="Unstable hand position detected. Please hold the wheel.",
    ),
    "talking": RuleConfig(
        min_seconds=5.0,
        cooldown_seconds=8.0,
        min_confidence=0.50,
        severity="LOW",
        message_ko="대화가 지속되고 있습니다. 주의가 분산되지 않도록 하세요.",
        message_en="Conversation is continuing. Please avoid distraction.",
    ),
}


@dataclass
class ActiveState:
    """State for one warning rule.

    accumulated_seconds is the actual warning stack. It increases only while the
    rule is active. If the rule becomes inactive or changes label briefly, the
    stack is preserved for `stack_hold_seconds` (= rule threshold by default).
    After one warning event is emitted, the stack is reset to 0 so the same
    behavior must persist for another full threshold before another TTS event.
    """
    accumulated_seconds: float = 0.0
    active_since: Optional[float] = None
    last_update_time: Optional[float] = None
    last_active_time: Optional[float] = None
    last_alert_time: float = -1e9
    current_label: Optional[str] = None
    current_confidence: Optional[float] = None

    def reset_stack(self) -> None:
        self.accumulated_seconds = 0.0
        self.active_since = None
        self.current_label = None
        self.current_confidence = None


@dataclass
class AlertEvent:
    frame_idx: int
    video_time: float
    wall_time: float
    rule: str
    severity: str
    duration: float
    label: str
    confidence: Optional[float]
    message_ko: str
    message_en: str

    def as_dict(self) -> dict:
        return {
            "frame_idx": int(self.frame_idx),
            "video_time": float(self.video_time),
            "wall_time": float(self.wall_time),
            "rule": self.rule,
            "severity": self.severity,
            "duration": float(self.duration),
            "label": self.label,
            "confidence": self.confidence,
            "message_ko": self.message_ko,
            "message_en": self.message_en,
        }


RULE_PRIORITY = {
    "dangerous_action": 4,
    "unsafe_hands": 3,
    "off_road_gaze": 2,
    "talking": 1,
}


class WarningStateMachine:
    """Sustained-warning state machine.

    Policy:
      1. Accumulate time only while an abnormal rule is active.
      2. If the rule becomes inactive or moves to another label briefly, keep the
         previous stack for `min_seconds` seconds. This prevents flicker from
         immediately clearing the warning stack.
      3. When threshold is reached, emit exactly one event and reset that rule's
         stack to 0. If the behavior continues, it must persist for another full
         threshold before another event can be emitted.
      4. If several rules trigger at the same frame, return only the highest
         priority event.
    """
    def __init__(self, rules: Dict[str, RuleConfig], fps: float):
        self.rules = rules
        self.fps = float(fps)
        self.states: Dict[str, ActiveState] = {k: ActiveState() for k in rules}
        self.active_flags: Dict[str, dict] = {}
        self.last_event: Optional[AlertEvent] = None

    def _conditions(self, preds: Dict[str, dict]) -> Dict[str, Tuple[bool, str, Optional[float]]]:
        action = preds.get("action", {})
        gaze = preds.get("gaze", {})
        hands = preds.get("hands", {})
        talk = preds.get("talk", {})

        action_label = action.get("label")
        gaze_label = gaze.get("label")
        hands_label = hands.get("label")
        talk_label = talk.get("label")

        return {
            "dangerous_action": (
                action_label in DANGEROUS_ACTIONS,
                str(action_label),
                action.get("confidence"),
            ),
            "off_road_gaze": (
                gaze_label in OFF_ROAD_GAZE,
                str(gaze_label),
                gaze.get("confidence"),
            ),
            "unsafe_hands": (
                hands_label in UNSAFE_HANDS,
                str(hands_label),
                hands.get("confidence"),
            ),
            "talking": (
                talk_label == "talk",
                str(talk_label),
                talk.get("confidence"),
            ),
        }

    def update(self, frame_idx: int, video_time: float, preds: Dict[str, dict]) -> Optional[AlertEvent]:
        now = float(video_time)
        conds = self._conditions(preds)
        self.active_flags = {}

        triggered: List[AlertEvent] = []

        for rule_name, (is_active, label, conf) in conds.items():
            cfg = self.rules[rule_name]
            st = self.states[rule_name]

            # Time delta since last update for this rule.
            if st.last_update_time is None:
                dt = 0.0
            else:
                dt = max(0.0, now - st.last_update_time)
            st.last_update_time = now

            conf_ok = conf is None or float(conf) >= cfg.min_confidence
            active = bool(is_active and conf_ok)

            # Hold previous stack for the rule threshold duration.
            # If detection flickers or switches briefly, the stack is preserved
            # but it does not increase while inactive.
            stack_hold_seconds = max(0.0, float(cfg.min_seconds))
            holding_stack = (
                st.last_active_time is not None
                and (now - st.last_active_time) <= stack_hold_seconds
                and st.accumulated_seconds > 0.0
            )

            if active:
                if st.last_active_time is not None and (now - st.last_active_time) > stack_hold_seconds:
                    # Previous stack expired before this detection came back.
                    st.reset_stack()

                # If label changes inside the same rule, keep the stack.
                # Example: texting_right -> phonecall_right still counts as
                # sustained dangerous_action.
                if st.active_since is None:
                    st.active_since = now - st.accumulated_seconds

                st.accumulated_seconds += dt
                st.last_active_time = now
                st.current_label = label
                st.current_confidence = conf

                self.active_flags[rule_name] = {
                    "duration": st.accumulated_seconds,
                    "label": label,
                    "confidence": conf,
                    "severity": cfg.severity,
                    "threshold": cfg.min_seconds,
                    "holding": False,
                }

                if st.accumulated_seconds >= cfg.min_seconds:
                    # Emit once, then reset the threshold stack. This prevents
                    # continuous TTS while the same state remains active.
                    ev = AlertEvent(
                        frame_idx=frame_idx,
                        video_time=video_time,
                        wall_time=time.time(),
                        rule=rule_name,
                        severity=cfg.severity,
                        duration=st.accumulated_seconds,
                        label=label,
                        confidence=conf,
                        message_ko=cfg.message_ko,
                        message_en=cfg.message_en,
                    )
                    st.last_alert_time = now
                    triggered.append(ev)

                    # Reset only this rule's stack after speaking once.
                    st.accumulated_seconds = 0.0
                    st.active_since = now
                    st.last_active_time = now
                    st.current_label = label
                    st.current_confidence = conf

            else:
                if holding_stack:
                    # Keep the stack available for a short time. Do not increase it.
                    self.active_flags[rule_name] = {
                        "duration": st.accumulated_seconds,
                        "label": st.current_label,
                        "confidence": st.current_confidence,
                        "severity": cfg.severity,
                        "threshold": cfg.min_seconds,
                        "holding": True,
                        "hold_remaining": max(0.0, stack_hold_seconds - (now - st.last_active_time)),
                    }
                else:
                    st.reset_stack()
                    st.last_active_time = None

        if not triggered:
            return None

        # Priority: explicit rule priority first, then severity, then duration.
        severity_priority = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        triggered.sort(
            key=lambda e: (
                RULE_PRIORITY.get(e.rule, 0),
                severity_priority.get(e.severity, 0),
                e.duration,
            ),
            reverse=True,
        )
        self.last_event = triggered[0]
        return triggered[0]


# ============================================================
# TTS/audio backend
# ============================================================

class NoLiveTTS:
    """No live TTS playback. This script embeds edge-tts audio after inference."""
    def __init__(self, enabled: bool = False, *args, **kwargs):
        self.enabled = False

    def say(self, text: str):
        return

    def close(self):
        return


# ============================================================
# Drawing
# ============================================================

COCO_EDGES = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]


def draw_pose(img: np.ndarray, keypoints: Any, conf: Any = None, conf_thres: float = 0.15) -> None:
    if keypoints is None:
        return
    keypoints = np.asarray(keypoints)
    if keypoints.shape != (17, 2):
        return
    if conf is None:
        conf = np.ones((17,), dtype=np.float32)
    else:
        conf = np.asarray(conf).reshape(-1)
        if conf.shape[0] != 17:
            conf = np.ones((17,), dtype=np.float32)

    for a, b in COCO_EDGES:
        if conf[a] < conf_thres or conf[b] < conf_thres:
            continue
        xa, ya = keypoints[a]
        xb, yb = keypoints[b]
        if xa <= 0 or ya <= 0 or xb <= 0 or yb <= 0:
            continue
        cv2.line(img, (int(round(xa)), int(round(ya))), (int(round(xb)), int(round(yb))), (0, 255, 255), 2, cv2.LINE_AA)

    for i, (x, y) in enumerate(keypoints):
        if conf[i] < conf_thres or x <= 0 or y <= 0:
            continue
        cv2.circle(img, (int(round(x)), int(round(y))), 4, (0, 0, 255), -1, cv2.LINE_AA)


EYE_LANDMARKS = {
    # left eye + iris + nearby
    33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173,
    468, 469, 470, 471, 472,
    22, 23, 24, 25, 26, 27, 28, 29, 30, 110, 112, 113, 124, 130, 143, 156, 189, 190, 221, 222, 223, 224, 225,
    # right eye + iris + nearby
    263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398,
    473, 474, 475, 476, 477,
    252, 253, 254, 255, 256, 257, 258, 259, 260, 339, 341, 342, 353, 359, 372, 383, 413, 414, 441, 442, 443, 444, 445,
}


def draw_face_debug(face_img: np.ndarray, debug: dict, scale_x: float = 1.0, scale_y: float = 1.0) -> None:
    if not isinstance(debug, dict):
        return

    bbox = debug.get("face_bbox", None)
    bbox_detected = bool(debug.get("bbox_detected", False))
    if bbox_detected and bbox is not None and len(bbox) == 4:
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
            cv2.rectangle(
                face_img,
                (int(round(x1 * scale_x)), int(round(y1 * scale_y))),
                (int(round(x2 * scale_x)), int(round(y2 * scale_y))),
                (0, 255, 255),
                2,
            )
        except Exception:
            pass

    lm = debug.get("facemesh", None)
    if lm is not None:
        try:
            lm = np.asarray(lm, dtype=np.float32)
            if lm.ndim == 2 and lm.shape[0] >= 100 and lm.shape[1] >= 2:
                for idx, p in enumerate(lm):
                    is_eye = idx in EYE_LANDMARKS
                    if idx % 4 != 0 and not is_eye:
                        continue
                    x, y = float(p[0]), float(p[1])
                    if x <= 0 or y <= 0:
                        continue
                    px = int(round(x * scale_x))
                    py = int(round(y * scale_y))
                    if not (0 <= px < face_img.shape[1] and 0 <= py < face_img.shape[0]):
                        continue
                    color = (255, 255, 0) if is_eye else (0, 255, 0)
                    radius = 2 if is_eye else 1
                    cv2.circle(face_img, (px, py), radius, color, -1, cv2.LINE_AA)
        except Exception:
            pass

    labels = debug.get("face_occ_labels", {})
    occ = debug.get("face_occ_feature", None)
    lines = []
    if isinstance(labels, dict):
        try:
            active = [k for k, v in labels.items() if int(v) == 1]
            lines.append("occ: " + (",".join(active) if active else "none"))
        except Exception:
            pass
    if occ is not None and len(occ) >= 5:
        try:
            lines += [
                f"L-eye {float(occ[0]):.2f}",
                f"R-eye {float(occ[1]):.2f}",
                f"Nose  {float(occ[2]):.2f}",
                f"Mouth {float(occ[3]):.2f}",
                f"valid {float(occ[4]):.0f}",
            ]
        except Exception:
            pass
    if lines:
        put_text_box(face_img, lines, 8, 8, font_scale=0.48, thickness=1, line_h=20, alpha=0.55)


def draw_status_overlay(
    img: np.ndarray,
    frame_idx: int,
    video_time: float,
    preds: Dict[str, dict],
    ready: bool,
    warning_sm: WarningStateMachine,
    last_event: Optional[AlertEvent],
) -> None:
    lines = [
        f"Frame: {frame_idx} | t={video_time:.1f}s",
        f"DMS: {'READY' if ready else 'BUFFERING'}",
    ]
    for task in TASKS:
        p = preds.get(task)
        if not p:
            lines.append(f"{task}: -")
            continue
        conf = p.get("confidence")
        margin = p.get("margin")
        tail = ""
        if conf is not None:
            tail += f" conf={float(conf):.2f}"
        if margin is not None:
            tail += f" margin={float(margin):.2f}"
        lines.append(f"{task}: {p.get('label')} [{p.get('class_id')}]{tail}")

    put_text_box(img, lines, 10, 8, font_scale=0.43, thickness=1, line_h=18, pad=7, alpha=0.55)

    # Active warning progress box.
    active_lines = []
    for rule, info in warning_sm.active_flags.items():
        active_lines.append(
            f"{rule}: {info['duration']:.1f}/{info['threshold']:.1f}s | {info['label']}"
        )
    if active_lines:
        put_text_box(
            img,
            ["WARNING WATCH"] + active_lines,
            10,
            134,
            font_scale=0.42,
            thickness=1,
            fg=(0, 255, 255),
            bg=(0, 0, 80),
            line_h=18,
            pad=7,
            alpha=0.62,
        )

    if last_event is not None and video_time - last_event.video_time < 3.0:
        color = (0, 0, 255) if last_event.severity == "HIGH" else (0, 165, 255)
        overlay = img.copy()
        banner_h = 66
        y0 = img.shape[0] - banner_h
        cv2.rectangle(overlay, (0, y0), (img.shape[1], img.shape[0]), color, -1)
        cv2.addWeighted(overlay, 0.34, img, 0.66, 0, dst=img)
        cv2.putText(
            img,
            f"WARNING [{last_event.severity}] {last_event.rule}",
            (18, y0 + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            last_event.message_en,
            (18, y0 + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


# ============================================================
# Build FullDMSSystem from existing project config
# ============================================================

def import_project(root: Path):
    sys.path.insert(0, str(root))
    from full_dms_system.full_system import FullDMSSystem, FullDMSConfig  # type: ignore
    from full_dms_system.yolo_pose_skeleton_extractor import YoloPoseSkeletonExtractor  # type: ignore
    return FullDMSSystem, FullDMSConfig, YoloPoseSkeletonExtractor


def build_full_dms_system(config_path: Path, root: Path):
    FullDMSSystem, FullDMSConfig, _ = import_project(root)
    with open(config_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)

    m = y["models"]
    r = y.get("runtime", {})
    p = y.get("paths", {})
    t = y.get("thresholds", {})
    face = y.get("face", {})

    cfg = FullDMSConfig(
        yolo_pose_path=m["yolo_pose_path"],
        yolo_face_path=m["yolo_face_path"],
        occ_cnn_path=m.get("occ_cnn_path"),
        orformer_ckpt=m.get("orformer_ckpt"),
        hgnet_ckpt=m.get("hgnet_ckpt"),
        dms_config_path=m["dms_config_path"],
        dms_checkpoint_path=m["dms_checkpoint_path"],
        classifier_root=p.get("classifier_root"),
        vendor_landmark_root=p.get("vendor_landmark_root"),
        vendor_orformer_path=p.get("vendor_orformer_path"),
        device=r.get("device", None),
        window_size=int(r.get("window_size", 48)),
        predict_stride=int(r.get("predict_stride", 1)),
        yolo_img_size=int(t.get("yolo_img_size", 640)),
        yolo_pose_conf=float(t.get("yolo_pose_conf", 0.25)),
        yolo_face_conf=float(t.get("yolo_face_conf", 0.25)),
        yolo_iou=float(t.get("yolo_iou", 0.6)),
        occ_visible_threshold=float(t.get("occ_visible_threshold", 0.5)),
        facemesh_pad_ratio=float(face.get("facemesh_pad_ratio", 0.2)),
        hgnet_enabled=bool(face.get("hgnet_enabled", True)),
        hgnet_crop_pad_ratio=float(face.get("hgnet_crop_pad_ratio", 0.10)),
        default_visible_prob=float(face.get("default_visible_prob", 0.5)),
        default_crop_valid=float(face.get("default_crop_valid", 0.0)),
    )
    return FullDMSSystem(cfg), y




# ============================================================
# Post-process TTS audio track + mux with video
# ============================================================

def _ffmpeg_exists() -> bool:
    return shutil.which("ffmpeg") is not None


def _read_warning_events(events_path: Path) -> List[dict]:
    events: List[dict] = []
    if not events_path.exists():
        return events
    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if "video_time" in ev:
                events.append(ev)
    return events


def _save_tts_mp3_edge(text: str, out_mp3: Path, voice: str) -> bool:
    """Save Korean/English TTS with edge-tts only. Requires: pip install edge-tts."""
    out_mp3.parent.mkdir(parents=True, exist_ok=True)

    if shutil.which("edge-tts") is None:
        raise RuntimeError("edge-tts command not found. Install it with: pip install edge-tts")

    cmd = [
        "edge-tts",
        "--voice", voice,
        "--text", text,
        "--write-media", str(out_mp3),
    ]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise RuntimeError("edge-tts failed: " + (r.stderr[-2000:] if r.stderr else ""))

    if not out_mp3.exists() or out_mp3.stat().st_size <= 1024:
        raise RuntimeError(f"edge-tts output is empty or missing: {out_mp3}")

    return True


def _normalize_audio_to_wav(src: Path, dst: Path) -> bool:
    if not _ffmpeg_exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le",
        str(dst),
    ]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return r.returncode == 0 and dst.exists() and dst.stat().st_size > 1024



TTS_GAP_SECONDS = 0.35


def _get_audio_duration_sec(path: Path) -> float:
    """Return audio duration in seconds using ffprobe."""
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe command not found. ffmpeg installation should include ffprobe.")
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {r.stderr[-1000:]}")
    return float(r.stdout.strip())


def _select_non_overlapping_tts_segments(candidates: List[dict]) -> List[dict]:
    """Select a non-overlapping set of TTS segments.

    If segments overlap, keep the higher-priority rule. If priority is equal,
    keep the earlier segment. This makes the final audio behave like a single
    driver-warning channel instead of mixing several messages at once.
    """
    selected: List[dict] = []

    for cand in sorted(candidates, key=lambda x: (x["start"], -x["priority"])):
        keep = True
        remove_indices: List[int] = []

        for idx, old in enumerate(selected):
            overlap = not (cand["end"] <= old["start"] or old["end"] <= cand["start"])
            if not overlap:
                continue

            if cand["priority"] > old["priority"]:
                remove_indices.append(idx)
            else:
                keep = False
                break

        if keep:
            for idx in reversed(remove_indices):
                selected.pop(idx)
            selected.append(cand)
            selected.sort(key=lambda x: x["start"])

    return selected


def _build_mixed_tts_track(
    events_path: Path,
    out_wav: Path,
    duration_sec: float,
    lang: str,
    tmp_dir: Path,
    max_events: int = 40,
    edge_voice: str = "ko-KR-SunHiNeural",
) -> bool:
    """Create one WAV track where each warning TTS starts at event['video_time']."""
    if not _ffmpeg_exists():
        print("[WARN] ffmpeg not found. Cannot build embedded audio track.")
        return False

    events = _read_warning_events(events_path)
    if not events:
        print("[INFO] no warning events. Creating silent audio track only.")

    # Avoid pathological huge ffmpeg graphs if a video fires too many alerts.
    events = events[:max_events]
    tmp_dir.mkdir(parents=True, exist_ok=True)

    candidates: List[dict] = []
    for i, ev in enumerate(events):
        msg = ev.get("message_ko") if lang == "ko" else ev.get("message_en")
        if not msg:
            msg = ev.get("message_en") or ev.get("message_ko") or "Warning"

        rule = str(ev.get("rule", "unknown"))
        priority = RULE_PRIORITY.get(rule, 0)

        raw_mp3 = tmp_dir / f"tts_raw_{i:04d}_{rule}.mp3"
        norm_wav = tmp_dir / f"tts_{i:04d}_{rule}.wav"

        _save_tts_mp3_edge(str(msg), raw_mp3, voice=edge_voice)
        if not _normalize_audio_to_wav(raw_mp3, norm_wav):
            raise RuntimeError(f"failed to convert edge-tts mp3 to wav: {raw_mp3}")

        start_sec = max(0.0, float(ev.get("video_time", 0.0)))
        dur_sec = _get_audio_duration_sec(norm_wav)
        end_sec = start_sec + dur_sec + TTS_GAP_SECONDS

        candidates.append({
            "event": ev,
            "path": norm_wav,
            "start": start_sec,
            "end": end_sec,
            "duration": dur_sec,
            "priority": priority,
            "rule": rule,
        })

    selected = _select_non_overlapping_tts_segments(candidates)
    skipped = len(candidates) - len(selected)
    print(f"[INFO] TTS schedule: raw_events={len(candidates)}, selected={len(selected)}, skipped_overlap={skipped}")

    segment_paths: List[Tuple[Path, int]] = []
    for seg in selected:
        delay_ms = max(0, int(round(float(seg["start"]) * 1000.0)))
        segment_paths.append((seg["path"], delay_ms))

    out_wav.parent.mkdir(parents=True, exist_ok=True)

    if not segment_paths:
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-t", f"{max(duration_sec, 0.1):.3f}",
            "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-c:a", "pcm_s16le",
            str(out_wav),
        ]
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return r.returncode == 0 and out_wav.exists()

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-t", f"{max(duration_sec, 0.1):.3f}",
        "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
    ]
    for seg, _ in segment_paths:
        cmd += ["-i", str(seg)]

    filters = []
    mix_inputs = ["[0:a]"]
    for idx, (_, delay_ms) in enumerate(segment_paths, start=1):
        tag = f"a{idx}"
        filters.append(f"[{idx}:a]adelay={delay_ms}|{delay_ms}[{tag}]")
        mix_inputs.append(f"[{tag}]")

    n_inputs = 1 + len(segment_paths)
    filters.append("".join(mix_inputs) + f"amix=inputs={n_inputs}:duration=first:dropout_transition=0[aout]")
    filter_complex = ";".join(filters)

    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[aout]",
        "-c:a", "pcm_s16le",
        str(out_wav),
    ]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        print("[WARN] ffmpeg audio mix failed")
        print(r.stderr[-2000:])
        return False
    return out_wav.exists() and out_wav.stat().st_size > 1024


def _mux_video_audio(video_path: Path, audio_wav: Path, out_video: Path) -> bool:
    if not _ffmpeg_exists():
        return False
    out_video.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_wav),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(out_video),
    ]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        print("[WARN] ffmpeg mux failed")
        print(r.stderr[-2000:])
        return False
    return out_video.exists() and out_video.stat().st_size > 1024


def embed_warning_audio_from_events(
    video_path: Path,
    events_path: Path,
    out_video_with_audio: Path,
    duration_sec: float,
    lang: str,
    work_dir: Path,
    max_events: int,
    edge_voice: str,
) -> bool:
    audio_track = work_dir / "warning_tts_track.wav"
    ok_audio = _build_mixed_tts_track(
        events_path=events_path,
        out_wav=audio_track,
        duration_sec=duration_sec,
        lang=lang,
        tmp_dir=work_dir / "tts_segments",
        max_events=max_events,
        edge_voice=edge_voice,
    )
    if not ok_audio:
        return False
    return _mux_video_audio(video_path, audio_track, out_video_with_audio)


# ============================================================
# Main runner
# ============================================================

def parse_args():
    """
    Hardcoded by default.
    You can still override values from CLI, but running without arguments is enough.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=HARDCODED_ROOT)
    ap.add_argument("--config", type=Path, default=HARDCODED_CONFIG)
    ap.add_argument("--face-video", type=Path, default=HARDCODED_FACE_VIDEO)
    ap.add_argument("--body-video", type=Path, default=HARDCODED_BODY_VIDEO)
    ap.add_argument("--out-video", type=Path, default=HARDCODED_OUT_VIDEO)
    ap.add_argument("--out-jsonl", type=Path, default=HARDCODED_OUT_JSONL)
    ap.add_argument("--out-events", type=Path, default=HARDCODED_OUT_EVENTS)
    ap.add_argument("--out-video-audio", type=Path, default=HARDCODED_OUT_VIDEO_AUDIO)
    ap.add_argument("--embed-tts-audio", action="store_true", default=HARDCODED_EMBED_TTS_AUDIO)
    ap.add_argument("--tts-work-dir", type=Path, default=HARDCODED_TTS_WORK_DIR)
    ap.add_argument("--max-tts-events", type=int, default=HARDCODED_MAX_TTS_EVENTS)
    ap.add_argument("--edge-voice", type=str, default=HARDCODED_EDGE_VOICE)
    ap.add_argument("--max-frames", type=int, default=HARDCODED_MAX_FRAMES)
    ap.add_argument("--display", action="store_true", default=HARDCODED_DISPLAY)
    ap.add_argument("--warning-lang", choices=["ko", "en"], default=HARDCODED_WARNING_LANG)
    ap.add_argument("--no-pose-overlay", action="store_true", default=HARDCODED_NO_POSE_OVERLAY)

    # Threshold overrides.
    ap.add_argument("--action-seconds", type=float, default=HARDCODED_ACTION_SECONDS)
    ap.add_argument("--gaze-seconds", type=float, default=HARDCODED_GAZE_SECONDS)
    ap.add_argument("--hands-seconds", type=float, default=HARDCODED_HANDS_SECONDS)
    ap.add_argument("--talk-seconds", type=float, default=HARDCODED_TALK_SECONDS)
    return ap.parse_args()


def main():
    args = parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config

    args.out_video = args.out_video if args.out_video.is_absolute() else root / args.out_video
    args.out_jsonl = args.out_jsonl if args.out_jsonl.is_absolute() else root / args.out_jsonl
    args.out_events = args.out_events if args.out_events.is_absolute() else root / args.out_events
    args.out_video_audio = args.out_video_audio if args.out_video_audio.is_absolute() else root / args.out_video_audio
    args.tts_work_dir = args.tts_work_dir if args.tts_work_dir.is_absolute() else root / args.tts_work_dir
    args.out_video.parent.mkdir(parents=True, exist_ok=True)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.out_events.parent.mkdir(parents=True, exist_ok=True)
    args.out_video_audio.parent.mkdir(parents=True, exist_ok=True)
    args.tts_work_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] loading FullDMSSystem")
    system, loaded_cfg = build_full_dms_system(config_path, root)
    _, _, YoloPoseSkeletonExtractor = import_project(root)

    pose_extractor = None
    if not args.no_pose_overlay:
        print("[INFO] loading pose overlay extractor")
        pose_extractor = YoloPoseSkeletonExtractor(
            model_path=loaded_cfg["models"]["yolo_pose_path"],
            img_size=int(loaded_cfg.get("thresholds", {}).get("yolo_img_size", 640)),
            conf=float(loaded_cfg.get("thresholds", {}).get("yolo_pose_conf", 0.25)),
            iou=float(loaded_cfg.get("thresholds", {}).get("yolo_iou", 0.6)),
        )

    face_cap = cv2.VideoCapture(str(args.face_video))
    body_cap = cv2.VideoCapture(str(args.body_video))
    if not face_cap.isOpened():
        raise RuntimeError(f"face video open failed: {args.face_video}")
    if not body_cap.isOpened():
        raise RuntimeError(f"body video open failed: {args.body_video}")

    fps = float(body_cap.get(cv2.CAP_PROP_FPS))
    if fps <= 1e-6:
        fps = 30.0

    face_total = int(face_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    body_total = int(body_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = min(face_total, body_total)
    if args.max_frames is not None:
        total = min(total, int(args.max_frames))

    ok_f, face0 = face_cap.read()
    ok_b, body0 = body_cap.read()
    if not ok_f or not ok_b:
        raise RuntimeError("first frame read failed")

    body_h, body_w = body0.shape[:2]
    face0_resized = resize_to_height(face0, body_h)
    out_w = body_w + face0_resized.shape[1]
    out_h = body_h

    writer = cv2.VideoWriter(
        str(args.out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_w, out_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"video writer open failed: {args.out_video}")

    face_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    body_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    rules = dict(DEFAULT_RULES)
    rules["dangerous_action"].min_seconds = args.action_seconds
    rules["off_road_gaze"].min_seconds = args.gaze_seconds
    rules["unsafe_hands"].min_seconds = args.hands_seconds
    rules["talking"].min_seconds = args.talk_seconds

    warning_sm = WarningStateMachine(rules, fps=fps)
    tts = NoLiveTTS(enabled=False)

    last_preds: Dict[str, dict] = {}
    last_event: Optional[AlertEvent] = None
    processed = 0
    start_wall = time.time()

    print(f"[INFO] face frames={face_total}, body frames={body_total}, use={total}, fps={fps:.2f}")
    print(f"[INFO] out video : {args.out_video}")
    print(f"[INFO] out jsonl : {args.out_jsonl}")
    print(f"[INFO] out events: {args.out_events}")

    with args.out_jsonl.open("w", encoding="utf-8") as jf, args.out_events.open("w", encoding="utf-8") as ef:
        for frame_idx in range(total):
            ok_f, face_frame = face_cap.read()
            ok_b, body_frame = body_cap.read()
            if not ok_f or not ok_b:
                break

            video_time = frame_idx / fps
            ready = False
            raw_pred = None
            event = None

            try:
                raw_pred = system.step(face_frame=face_frame, body_frame=body_frame)
                if raw_pred is not None:
                    parsed = parse_preds(raw_pred)
                    if parsed:
                        last_preds = parsed
                        ready = True
                        event = warning_sm.update(frame_idx, video_time, last_preds)
                        if event is not None:
                            last_event = event
                            ef.write(json.dumps(event.as_dict(), ensure_ascii=False) + "\n")
                            ef.flush()
                            msg = event.message_ko if args.warning_lang == "ko" else event.message_en
                            print(f"[ALERT] t={video_time:.1f}s frame={frame_idx} {event.rule} {event.label} dur={event.duration:.1f}s")
                            tts.say(msg)
            except Exception as e:
                print(f"[WARN] DMS failed at frame {frame_idx}: {type(e).__name__}: {e}")

            body_vis = body_frame.copy()
            if pose_extractor is not None:
                try:
                    pose = pose_extractor(body_frame)
                    if pose.get("detected", False):
                        draw_pose(body_vis, pose["keypoints"], pose.get("conf", None))
                except Exception as e:
                    print(f"[WARN] pose overlay failed at frame {frame_idx}: {type(e).__name__}: {e}")

            face_vis = resize_to_height(face_frame, body_h)
            debug = raw_pred.get("debug", {}) if isinstance(raw_pred, dict) else {}
            scale_x = face_vis.shape[1] / max(face_frame.shape[1], 1)
            scale_y = face_vis.shape[0] / max(face_frame.shape[0], 1)
            draw_face_debug(face_vis, debug, scale_x=scale_x, scale_y=scale_y)

            combined = np.concatenate([body_vis, face_vis], axis=1)
            warning_banner_visible = last_event is not None and video_time - last_event.video_time < 3.0
            draw_status_overlay(combined, frame_idx, video_time, last_preds, ready, warning_sm, last_event)

            if not warning_banner_visible:
                cv2.putText(combined, "BODY / POSE", (16, out_h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(combined, "FACE / MESH / OCC / WARNING", (body_w + 16, out_h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)

            writer.write(combined)

            raw_for_save = raw_pred
            if isinstance(raw_for_save, dict):
                raw_for_save = dict(raw_for_save)
                if "debug" in raw_for_save and isinstance(raw_for_save["debug"], dict):
                    raw_for_save["debug"] = dict(raw_for_save["debug"])
                    raw_for_save["debug"].pop("facemesh", None)

            jf.write(json.dumps({
                "frame_idx": frame_idx,
                "video_time": video_time,
                "ready": ready,
                "predictions": last_preds,
                "active_warnings": warning_sm.active_flags,
                "event": event.as_dict() if event is not None else None,
                "raw": raw_for_save,
            }, ensure_ascii=False, default=str) + "\n")

            if args.display:
                cv2.imshow("Full DMS Warning System", combined)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            processed += 1
            if frame_idx > 0 and frame_idx % 100 == 0:
                elapsed = time.time() - start_wall
                print(f"[INFO] {frame_idx}/{total} | avg_fps={processed / max(elapsed, 1e-9):.2f}")

    face_cap.release()
    body_cap.release()
    writer.release()
    if args.display:
        cv2.destroyAllWindows()
    try:
        system.close()
    except Exception:
        pass
    tts.close()

    video_duration = processed / max(fps, 1e-9)
    embedded_audio_ok = False
    if args.embed_tts_audio:
        print("[INFO] building embedded edge-tts audio track from warning events")
        embedded_audio_ok = embed_warning_audio_from_events(
            video_path=args.out_video,
            events_path=args.out_events,
            out_video_with_audio=args.out_video_audio,
            duration_sec=video_duration,
            lang=args.warning_lang,
            work_dir=args.tts_work_dir,
            max_events=args.max_tts_events,
            edge_voice=args.edge_voice,
        )
        if embedded_audio_ok:
            print(f"[SAVE] video with audio: {args.out_video_audio}")
        else:
            print("[WARN] failed to create video with embedded TTS audio")

    elapsed = time.time() - start_wall
    print("\n[DONE]")
    print("video :", args.out_video)
    print("jsonl :", args.out_jsonl)
    print("events:", args.out_events)
    if args.embed_tts_audio:
        print("video+audio:", args.out_video_audio)
    print(f"frames: {processed}/{total}")
    print(f"elapsed: {elapsed:.1f}s | avg_fps={processed / max(elapsed, 1e-9):.2f}")


if __name__ == "__main__":
    main()
