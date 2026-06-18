from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml

# Allow running from repository root without installing.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from full_dms_system import FullDMSConfig, FullDMSSystem


def _resolve_path(p, root: Path):
    if p is None or str(p).lower() == "null" or str(p).strip() == "":
        return None
    pp = Path(str(p))
    if not pp.is_absolute():
        pp = root / pp
    return str(pp)


def load_full_config(path: str | Path) -> FullDMSConfig:
    path = Path(path)
    root = path.resolve().parents[1] if path.name.endswith(".yaml") else Path.cwd()
    with path.open("r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    m = y.get("models", {})
    r = y.get("runtime", {})
    p = y.get("paths", {})
    t = y.get("thresholds", {})
    face = y.get("face", {})
    return FullDMSConfig(
        yolo_pose_path=_resolve_path(m["yolo_pose_path"], root),
        yolo_face_path=_resolve_path(m["yolo_face_path"], root),
        occ_cnn_path=_resolve_path(m.get("occ_cnn_path"), root),
        orformer_ckpt=_resolve_path(m.get("orformer_ckpt"), root),
        hgnet_ckpt=_resolve_path(m.get("hgnet_ckpt"), root),
        dms_config_path=_resolve_path(m["dms_config_path"], root),
        dms_checkpoint_path=_resolve_path(m["dms_checkpoint_path"], root),
        classifier_root=_resolve_path(p.get("classifier_root"), root),
        vendor_landmark_root=_resolve_path(p.get("vendor_landmark_root"), root),
        vendor_orformer_path=_resolve_path(p.get("vendor_orformer_path"), root),
        device=r.get("device"),
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


def to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    return obj


def main():
    ap = argparse.ArgumentParser(description="Run integrated face/body DMS on a pair of videos.")
    ap.add_argument("--config", required=True, help="configs/full_dms_config_template.yaml after editing paths")
    ap.add_argument("--face-video", required=True)
    ap.add_argument("--body-video", required=True)
    ap.add_argument("--out-jsonl", default="outputs/dms_predictions.jsonl")
    ap.add_argument("--max-frames", type=int, default=-1)
    args = ap.parse_args()

    cfg = load_full_config(args.config)
    system = FullDMSSystem(cfg)

    cap_f = cv2.VideoCapture(args.face_video)
    cap_b = cv2.VideoCapture(args.body_video)
    if not cap_f.isOpened():
        raise RuntimeError(f"cannot open face video: {args.face_video}")
    if not cap_b.isOpened():
        raise RuntimeError(f"cannot open body video: {args.body_video}")

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    n_pred = 0
    try:
        with out_path.open("w", encoding="utf-8") as f:
            while True:
                ok_f, face_frame = cap_f.read()
                ok_b, body_frame = cap_b.read()
                if not ok_f or not ok_b:
                    break
                pred = system.step(face_frame, body_frame)
                if pred is not None:
                    f.write(json.dumps(to_jsonable(pred), ensure_ascii=False) + "\n")
                    n_pred += 1
                n += 1
                if args.max_frames > 0 and n >= args.max_frames:
                    break
                if n % 100 == 0:
                    print(f"processed={n} predictions={n_pred}")
    finally:
        cap_f.release()
        cap_b.release()
        system.close()

    print(f"done frames={n} predictions={n_pred} out={out_path}")


if __name__ == "__main__":
    main()
