from pathlib import Path
import sys
import json
import time

import cv2
import numpy as np

ROOT = Path("/data/shared/scuppy/Full_System")
sys.path.insert(0, str(ROOT))

import yaml
from full_dms_system.full_system import FullDMSSystem, FullDMSConfig
from full_dms_system.yolo_pose_skeleton_extractor import YoloPoseSkeletonExtractor


# ============================================================
# 1. 하드코딩 경로
# ============================================================

CONFIG_PATH = Path("/data/shared/scuppy/Full_System/configs/full_dms_config_template.yaml")

FACE_VIDEO = Path(
    "/data/shared/Occlusion_subset_dataset/region_occlusion_video_dataset_v3_gaze_fixedmask/"
    "videos/left_eye/checker/gaze_dmd_gA_5_s6_gA_5_s6_2019-03-08T10;40;35+01;00_ir_face_facemesh_left_eye_checker_origsize_fixedmask.mp4"
)

BODY_VIDEO = Path(
    "/data/shared/DMD/distraction/dmd/gA/1/s1/"
    "gA_1_s1_2019-03-08T09;31;15+01;00_ir_body.mp4"
)

OUT_VIDEO = ROOT / "outputs/sample_dms_overlay.mp4"
OUT_JSONL = ROOT / "outputs/sample_dms_predictions.jsonl"

# 테스트만 할 때는 300, 1000 같은 값으로 제한
MAX_FRAMES = None


# ============================================================
# 2. Pose skeleton drawing
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

TASKS = ["action", "gaze", "hands", "talk"]


# ============================================================
# 3. Label map
# ============================================================
# 필요하면 실제 DMD label 순서에 맞춰 여기만 수정하면 됨.

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

TASK_LABELS = {
    "action": ACTION_LABELS,
    "gaze": GAZE_LABELS,
    "hands": HANDS_LABELS,
    "talk": TALK_LABELS,
}


def id_to_label(task, class_id):
    try:
        class_id = int(class_id)
    except Exception:
        return str(class_id)

    table = TASK_LABELS.get(task, {})
    return table.get(class_id, f"class_{class_id}")


# ============================================================
# 4. Full system builder
# ============================================================

def build_full_dms_system(config_path):
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

    return FullDMSSystem(cfg)


# ============================================================
# 5. Drawing utils
# ============================================================

def draw_pose(img, keypoints, conf=None, conf_thres=0.15):
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

        cv2.line(
            img,
            (int(round(xa)), int(round(ya))),
            (int(round(xb)), int(round(yb))),
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    for i, (x, y) in enumerate(keypoints):
        if conf[i] < conf_thres:
            continue
        if x <= 0 or y <= 0:
            continue

        cv2.circle(
            img,
            (int(round(x)), int(round(y))),
            4,
            (0, 0, 255),
            -1,
            cv2.LINE_AA,
        )


def resize_to_height(img, target_h):
    h, w = img.shape[:2]
    if h == target_h:
        return img

    scale = target_h / max(h, 1)
    new_w = int(round(w * scale))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def get_restored_facemesh_indices(restored_regions):
    """
    restored_regions에 포함된 region의 landmark index를 반환.
    현재 mapping은 MediaPipe 478 기준 approximate mapping.

    주의:
      OCC label의 left/right와 화면 overlay에서 보이는 MediaPipe landmark left/right가
      반대로 보이는 경우가 있어 eye region은 swap해서 표시한다.
    """
    restored_regions = set(str(x) for x in (restored_regions or []))

    LEFT_EYE = {
        33, 7, 163, 144, 145, 153, 154, 155, 133,
        246, 161, 160, 159, 158, 157, 173,
    }

    RIGHT_EYE = {
        263, 249, 390, 373, 374, 380, 381, 382, 362,
        466, 388, 387, 386, 385, 384, 398,
    }

    NOSE = {
        1, 2, 4, 5, 6, 19, 45, 51, 94, 97, 98, 115,
        168, 195, 197, 220, 275, 281, 326, 327, 344,
    }

    MOUTH = {
        0, 11, 12, 13, 14, 15, 16, 17,
        37, 39, 40, 61, 78, 80, 81, 82, 84, 87,
        88, 91, 95, 146, 178, 181, 185, 191,
        267, 269, 270, 291, 308, 310, 311, 312, 314,
        317, 318, 321, 324, 375, 402, 405, 409, 415,
    }

    out = set()

    # eye만 좌우 swap
    if "left_eye" in restored_regions or "left_eye_visible" in restored_regions:
        out |= RIGHT_EYE

    if "right_eye" in restored_regions or "right_eye_visible" in restored_regions:
        out |= LEFT_EYE

    if "nose" in restored_regions or "nose_visible" in restored_regions:
        out |= NOSE

    if "mouth" in restored_regions or "mouth_visible" in restored_regions:
        out |= MOUTH

    return out


def draw_face_debug(face_img, debug, scale_x=1.0, scale_y=1.0):
    """
    face_img: resize된 face image
    debug: FullDMSSystem.step()의 raw_pred["debug"]
    scale_x / scale_y: 원본 face frame 좌표 → resize된 face_img 좌표 보정

    색상:
      - YOLO bbox: yellow
      - MediaPipe 유지 landmark: green
      - HGNet 복원 landmark: red
    """
    if not isinstance(debug, dict):
        return

    # -------------------------
    # 1) YOLO face bbox
    # -------------------------
    bbox = debug.get("face_bbox", None)
    bbox_detected = bool(debug.get("bbox_detected", False))

    if bbox_detected and bbox is not None and len(bbox) == 4:
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
            p1 = (int(round(x1 * scale_x)), int(round(y1 * scale_y)))
            p2 = (int(round(x2 * scale_x)), int(round(y2 * scale_y)))
            cv2.rectangle(face_img, p1, p2, (0, 255, 255), 2)
        except Exception:
            pass

    # -------------------------
    # 2) FaceMesh points
    # -------------------------
    lm = debug.get("facemesh", None)

    hgnet_used = bool(debug.get("hgnet_used", False))
    restored_regions = debug.get("restored_regions", [])
    restored_indices = get_restored_facemesh_indices(restored_regions) if hgnet_used else set()

    if lm is not None:
        try:
            lm = np.asarray(lm, dtype=np.float32)

            if lm.ndim == 2 and lm.shape[0] >= 100 and lm.shape[1] >= 2:
                for idx, p in enumerate(lm):
                    # 일반 MediaPipe 점은 4개마다 하나씩.
                    # HGNet 복원점은 region 강조를 위해 전부 표시.
                    if idx % 4 != 0 and idx not in restored_indices:
                        continue

                    x, y = float(p[0]), float(p[1])
                    if x <= 0 or y <= 0:
                        continue

                    px = int(round(x * scale_x))
                    py = int(round(y * scale_y))

                    if not (0 <= px < face_img.shape[1] and 0 <= py < face_img.shape[0]):
                        continue

                    if idx in restored_indices:
                        color = (0, 0, 255)     # HGNet restored = red
                        radius = 2
                    else:
                        color = (0, 255, 0)     # MediaPipe kept = green
                        radius = 1

                    cv2.circle(face_img, (px, py), radius, color, -1, cv2.LINE_AA)

        except Exception:
            pass

    # -------------------------
    # 3) Occ / face status text
    # -------------------------
    status = debug.get("face_status", "-")
    occ = debug.get("face_occ_feature", None)
    labels = debug.get("face_occ_labels", {})

    has_occ = bool(debug.get("has_occ", False))
    hgnet_detected = bool(debug.get("hgnet_detected", False))
    hgnet_used = bool(debug.get("hgnet_used", False))
    restored_regions = debug.get("restored_regions", [])

    lines = [
        f"face: {status}",
        f"occ: {'YES' if has_occ else 'NO'}",
        f"HGNet: {'ON' if hgnet_used else 'OFF'}",
    ]

    if has_occ and not hgnet_detected:
        lines.append("restore: fallback MP")

    if restored_regions:
        lines.append("restored: " + ",".join([str(x) for x in restored_regions]))
    else:
        lines.append("restored: none")

    if occ is not None and len(occ) >= 5:
        try:
            lines.append(f"L-eye: {float(occ[0]):.2f}")
            lines.append(f"R-eye: {float(occ[1]):.2f}")
            lines.append(f"Nose : {float(occ[2]):.2f}")
            lines.append(f"Mouth: {float(occ[3]):.2f}")
            lines.append(f"valid: {float(occ[4]):.0f}")
        except Exception:
            pass

    if isinstance(labels, dict):
        try:
            occ_regions = [k for k, v in labels.items() if int(v) == 1]
            if occ_regions:
                lines.append("occ regions: " + ",".join(occ_regions))
            else:
                lines.append("occ regions: none")
        except Exception:
            pass

    x, y = 15, 30
    line_h = 22
    box_w = 340
    box_h = 18 + line_h * len(lines)

    cv2.rectangle(
        face_img,
        (8, 8),
        (box_w, box_h),
        (0, 0, 0),
        -1,
    )

    for i, line in enumerate(lines):
        cv2.putText(
            face_img,
            line,
            (x, y + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def parse_preds(pred):
    """
    FullDMSSystem 출력에서 task별 class id + class name + confidence를 읽음.
    예상:
      pred["action"] = {"pred": int, "prob": list/array, "confidence": float}
    """
    if pred is None:
        return {}

    if not isinstance(pred, dict):
        return {}

    # 혹시 predictions/tasks/outputs로 감싸져 있으면 벗김
    if "predictions" in pred and isinstance(pred["predictions"], dict):
        pred = pred["predictions"]
    elif "tasks" in pred and isinstance(pred["tasks"], dict):
        pred = pred["tasks"]
    elif "outputs" in pred and isinstance(pred["outputs"], dict):
        pred = pred["outputs"]

    out = {}

    for task in TASKS:
        if task not in pred:
            continue

        v = pred[task]

        if isinstance(v, dict):
            class_id = None

            for k in ["pred", "class_id", "pred_id", "argmax", "index", "cls"]:
                if k in v and v[k] is not None:
                    try:
                        class_id = int(v[k])
                        break
                    except Exception:
                        pass

            conf = None

            # confidence 우선
            for k in ["confidence", "score", "max_prob"]:
                if k in v and v[k] is not None:
                    try:
                        conf = float(v[k])
                        break
                    except Exception:
                        pass

            # prob 배열이면 class_id 위치 값 사용
            if conf is None and "prob" in v and v["prob"] is not None:
                try:
                    arr = np.asarray(v["prob"], dtype=np.float32).reshape(-1)
                    if class_id is None:
                        class_id = int(np.argmax(arr))
                    conf = float(arr[class_id])
                except Exception:
                    pass

            if class_id is not None:
                label = id_to_label(task, class_id)
                out[task] = (f"{label} [{class_id}]", conf)
            else:
                out[task] = (str(v), conf)

        elif isinstance(v, (list, tuple, np.ndarray)):
            arr = np.asarray(v, dtype=np.float32).reshape(-1)
            if arr.size > 0:
                class_id = int(np.argmax(arr))
                conf = float(arr[class_id])
                label = id_to_label(task, class_id)
                out[task] = (f"{label} [{class_id}]", conf)

        else:
            out[task] = (str(v), None)

    return out


def draw_result_text(img, frame_idx, preds, ready):
    lines = [f"Frame: {frame_idx}", f"DMS: {'ready' if ready else 'buffering'}"]

    for task in TASKS:
        if task not in preds:
            lines.append(f"{task}: -")
        else:
            label, conf = preds[task]
            if conf is None:
                lines.append(f"{task}: {label}")
            else:
                lines.append(f"{task}: {label} ({float(conf):.3f})")

    x, y = 20, 35
    line_h = 27

    # 텍스트 길이에 맞춰 박스 폭 대략 크게
    box_w = 560
    box_h = 20 + line_h * len(lines)

    cv2.rectangle(
        img,
        (10, 8),
        (box_w, box_h),
        (0, 0, 0),
        -1,
    )

    for i, line in enumerate(lines):
        cv2.putText(
            img,
            line,
            (x, y + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def format_time(sec):
    sec = max(0.0, float(sec))

    if sec < 60:
        return f"{sec:.1f}s"

    minutes = sec / 60.0

    if minutes < 60:
        return f"{minutes:.1f}m"

    hours = minutes / 60.0
    return f"{hours:.2f}h"


# ============================================================
# 6. Main
# ============================================================

def main():
    OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] loading FullDMSSystem")
    system = build_full_dms_system(CONFIG_PATH)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        loaded_cfg = yaml.safe_load(f)

    print("[INFO] loading pose overlay extractor")
    pose_extractor = YoloPoseSkeletonExtractor(
        model_path=loaded_cfg["models"]["yolo_pose_path"],
        img_size=640,
        conf=0.25,
        iou=0.6,
    )

    face_cap = cv2.VideoCapture(str(FACE_VIDEO))
    body_cap = cv2.VideoCapture(str(BODY_VIDEO))

    if not face_cap.isOpened():
        raise RuntimeError(f"face video open failed: {FACE_VIDEO}")

    if not body_cap.isOpened():
        raise RuntimeError(f"body video open failed: {BODY_VIDEO}")

    fps = body_cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = 30.0

    face_total = int(face_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    body_total = int(body_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = min(face_total, body_total)

    if MAX_FRAMES is not None:
        total = min(total, MAX_FRAMES)

    print(f"[INFO] frames: face={face_total}, body={body_total}, use={total}")
    print(f"[INFO] video fps: {fps:.2f}")

    ok_f, face0 = face_cap.read()
    ok_b, body0 = body_cap.read()

    if not ok_f or not ok_b:
        raise RuntimeError("first frame read failed")

    body_h, body_w = body0.shape[:2]
    face0_resized = resize_to_height(face0, body_h)

    out_w = body_w + face0_resized.shape[1]
    out_h = body_h

    writer = cv2.VideoWriter(
        str(OUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_w, out_h),
    )

    if not writer.isOpened():
        raise RuntimeError(f"video writer open failed: {OUT_VIDEO}")

    # rewind
    face_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    body_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    last_preds = {}

    global_start_time = time.time()
    block_start_time = time.time()
    block_start_frame = 0

    total_dms_time = 0.0
    total_pose_overlay_time = 0.0
    total_draw_write_time = 0.0

    processed_frames = 0

    with open(OUT_JSONL, "w", encoding="utf-8") as jf:
        for frame_idx in range(total):
            ok_f, face_frame = face_cap.read()
            ok_b, body_frame = body_cap.read()

            if not ok_f or not ok_b:
                print(f"[WARN] video read stopped at frame {frame_idx}")
                break

            ready = False
            raw_pred = None

            # ---------------------------------------------
            # 1) Full DMS step
            # ---------------------------------------------
            dms_t0 = time.time()

            try:
                raw_pred = system.step(face_frame=face_frame, body_frame=body_frame)

                if raw_pred is not None:
                    ready = True
                    parsed = parse_preds(raw_pred)

                    if parsed:
                        last_preds = parsed

            except Exception as e:
                print(f"[WARN] DMS failed at frame {frame_idx}: {type(e).__name__}: {e}")

            total_dms_time += time.time() - dms_t0

            # ---------------------------------------------
            # 2) Body pose overlay
            # ---------------------------------------------
            pose_t0 = time.time()
            body_vis = body_frame.copy()

            try:
                pose = pose_extractor(body_frame)

                if pose.get("detected", False):
                    draw_pose(body_vis, pose["keypoints"], pose.get("conf", None))

            except Exception as e:
                print(f"[WARN] pose overlay failed at frame {frame_idx}: {type(e).__name__}: {e}")

            total_pose_overlay_time += time.time() - pose_t0

            # ---------------------------------------------
            # 3) Face overlay + combine
            # ---------------------------------------------
            draw_t0 = time.time()

            face_vis = resize_to_height(face_frame, body_h)

            if raw_pred is not None and isinstance(raw_pred, dict):
                debug = raw_pred.get("debug", {})
            else:
                debug = {}

            scale_x = face_vis.shape[1] / max(face_frame.shape[1], 1)
            scale_y = face_vis.shape[0] / max(face_frame.shape[0], 1)

            draw_face_debug(face_vis, debug, scale_x=scale_x, scale_y=scale_y)

            combined = np.concatenate([body_vis, face_vis], axis=1)

            draw_result_text(combined, frame_idx, last_preds, ready)

            cv2.putText(
                combined,
                "BODY / POSE",
                (20, out_h - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                combined,
                "FACE / BBOX / MESH / OCC",
                (body_w + 20, out_h - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(combined)

            # JSONL 저장 시 facemesh 제거. 너무 커짐.
            raw_for_save = raw_pred
            if isinstance(raw_for_save, dict):
                raw_for_save = dict(raw_for_save)

                if "debug" in raw_for_save and isinstance(raw_for_save["debug"], dict):
                    raw_for_save["debug"] = dict(raw_for_save["debug"])
                    raw_for_save["debug"].pop("facemesh", None)

            row = {
                "frame_idx": frame_idx,
                "ready": ready,
                "predictions": last_preds,
                "raw": raw_for_save,
            }

            jf.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

            total_draw_write_time += time.time() - draw_t0
            processed_frames += 1

            # ---------------------------------------------
            # 4) 100-frame timing report
            # ---------------------------------------------
            if frame_idx > 0 and frame_idx % 100 == 0:
                now = time.time()

                block_frames = frame_idx - block_start_frame
                block_elapsed = now - block_start_time

                block_sec_per_100 = block_elapsed / max(block_frames, 1) * 100.0
                block_fps = block_frames / max(block_elapsed, 1e-9)

                total_elapsed = now - global_start_time
                avg_fps = frame_idx / max(total_elapsed, 1e-9)

                remain_frames = total - frame_idx
                eta_sec = remain_frames / max(avg_fps, 1e-9)

                avg_dms_per_frame = total_dms_time / max(frame_idx + 1, 1)
                avg_pose_overlay_per_frame = total_pose_overlay_time / max(frame_idx + 1, 1)
                avg_draw_write_per_frame = total_draw_write_time / max(frame_idx + 1, 1)

                print(
                    f"[INFO] {frame_idx}/{total} | "
                    f"last {block_frames}f={block_elapsed:.1f}s "
                    f"({block_sec_per_100:.1f}s/100f, {block_fps:.2f} fps) | "
                    f"avg={avg_fps:.2f} fps | "
                    f"elapsed={format_time(total_elapsed)} | "
                    f"ETA={format_time(eta_sec)} | "
                    f"avg/frame: dms={avg_dms_per_frame:.3f}s, "
                    f"pose_overlay={avg_pose_overlay_per_frame:.3f}s, "
                    f"draw_write={avg_draw_write_per_frame:.3f}s"
                )

                block_start_time = now
                block_start_frame = frame_idx

    face_cap.release()
    body_cap.release()
    writer.release()

    try:
        system.close()
    except Exception:
        pass

    total_elapsed = time.time() - global_start_time

    print("\n[DONE]")
    print("[DONE] video:", OUT_VIDEO)
    print("[DONE] jsonl:", OUT_JSONL)
    print(f"[DONE] processed frames: {processed_frames}/{total}")
    print(f"[DONE] total elapsed: {format_time(total_elapsed)}")
    print(f"[DONE] avg fps: {processed_frames / max(total_elapsed, 1e-9):.2f}")
    print(f"[DONE] avg sec / 100 frames: {total_elapsed / max(processed_frames, 1) * 100.0:.1f}s")


if __name__ == "__main__":
    main()