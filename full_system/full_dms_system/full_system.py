from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from .utils import make_zero_facemesh, make_zero_pose
from .yolo_pose_skeleton_extractor import YoloPoseSkeletonExtractor
from .yolo_face_bbox_extractor import YoloFaceBBoxExtractor
from .mediapipe_facemesh_yolo import MediaPipeFaceMeshOnYoloCrop
from .occ_cnn_realtime import OccCNNRealtimeWrapper
from .hgnet_restorer import HGNetRestorer
from .occ_gate_merger import OccGatedFaceMeshMerger
from .temporal_buffer import TemporalDMSBuffer
from .dms_classifier_wrapper import DMSClassifierWrapper


@dataclass
class FullDMSConfig:
    # Required model paths
    yolo_pose_path: str
    yolo_face_path: str
    dms_config_path: str
    dms_checkpoint_path: str

    # Optional face reliability / restoration paths
    occ_cnn_path: Optional[str] = None
    orformer_ckpt: Optional[str] = None
    hgnet_ckpt: Optional[str] = None

    # Optional vendor roots
    classifier_root: Optional[str] = None
    vendor_landmark_root: Optional[str] = None
    vendor_orformer_path: Optional[str] = None

    # Runtime settings
    device: Optional[Union[str, int]] = None
    window_size: int = 48
    predict_stride: int = 1

    # Detector settings
    yolo_img_size: int = 640
    yolo_pose_conf: float = 0.25
    yolo_face_conf: float = 0.25
    yolo_iou: float = 0.6

    # Face settings
    facemesh_pad_ratio: float = 0.2
    occ_visible_threshold: float = 0.5
    hgnet_enabled: bool = True
    hgnet_crop_pad_ratio: float = 0.10

    # Fallback policy
    default_visible_prob: float = 0.5
    default_crop_valid: float = 0.0


class FullDMSSystem:
    """
    End-to-end integrated DMS system.

    Input per step:
        face_frame: OpenCV BGR image from face camera/video
        body_frame: OpenCV BGR image from body camera/video

    Output per step:
        None until temporal buffer is ready, then a dict with action/gaze/hands/talk predictions.

    Failure policy:
        - If body detection fails: zero skeleton + zero confidence.
        - If face bbox/FaceMesh fails: zero FaceMesh, neutral occ vector, crop_valid=0.
        - If HGNet is unavailable or fails: keep MediaPipe landmarks and still produce DMS output.

    Debug policy:
        - final_landmarks are always the landmarks actually fed into DMS.
        - If HGNet is used, final_landmarks contain HGNet-restored values for occluded regions.
        - debug["hgnet_used"] indicates whether HGNet-restored landmarks were actually merged.
        - debug["restored_regions"] indicates which face regions were replaced by HGNet output.
    """

    def __init__(self, cfg: FullDMSConfig):
        self.cfg = cfg
        self.device = cfg.device if cfg.device is not None else (0 if torch.cuda.is_available() else "cpu")
        self.window_size = int(cfg.window_size)
        self.predict_stride = max(1, int(cfg.predict_stride))
        self.frame_count = 0

        self.pose = YoloPoseSkeletonExtractor(
            model_path=cfg.yolo_pose_path,
            img_size=cfg.yolo_img_size,
            conf=cfg.yolo_pose_conf,
            iou=cfg.yolo_iou,
            device=self.device,
        )

        self.face_bbox = YoloFaceBBoxExtractor(
            model_path=cfg.yolo_face_path,
            img_size=cfg.yolo_img_size,
            conf=cfg.yolo_face_conf,
            iou=cfg.yolo_iou,
            device=self.device,
        )

        self.facemesh = MediaPipeFaceMeshOnYoloCrop(
            refine_landmarks=True,
            min_detection_confidence=0.3,
            pad_ratio=cfg.facemesh_pad_ratio,
            static_image_mode=True,
        )

        self.occ = OccCNNRealtimeWrapper(
            ckpt_path=cfg.occ_cnn_path,
            device=self.device,
            default_visible_prob=cfg.default_visible_prob,
            default_crop_valid=cfg.default_crop_valid,
        )

        self.hgnet = HGNetRestorer(
            orformer_ckpt=cfg.orformer_ckpt,
            hgnet_ckpt=cfg.hgnet_ckpt,
            vendor_landmark_root=cfg.vendor_landmark_root,
            vendor_orformer_path=cfg.vendor_orformer_path,
            device=self.device,
            enabled=cfg.hgnet_enabled,
            crop_pad_ratio=cfg.hgnet_crop_pad_ratio,
        )

        self.merger = OccGatedFaceMeshMerger(visible_threshold=cfg.occ_visible_threshold)
        self.buffer = TemporalDMSBuffer(window_size=self.window_size)

        self.dms = DMSClassifierWrapper(
            config_path=cfg.dms_config_path,
            checkpoint_path=cfg.dms_checkpoint_path,
            classifier_root=cfg.classifier_root,
            device=self.device,
        )

    def close(self) -> None:
        if hasattr(self, "facemesh") and self.facemesh is not None:
            self.facemesh.close()

    def step(self, face_frame: np.ndarray, body_frame: np.ndarray) -> Optional[Dict[str, Any]]:
        self.frame_count += 1

        # ------------------------------------------------------------
        # Body branch
        # ------------------------------------------------------------
        body = self.pose(body_frame)

        if not body.get("detected", False):
            body_kp, body_cf = make_zero_pose()
        else:
            body_kp = body["keypoints"]
            body_cf = body["conf"]

        # ------------------------------------------------------------
        # Face branch
        # ------------------------------------------------------------
        face = self._extract_face_branch(face_frame)

        # ------------------------------------------------------------
        # Temporal buffer
        # ------------------------------------------------------------
        self.buffer.append(
            body_keypoints=body_kp,
            body_conf=body_cf,
            face_lm=face["final_landmarks"],
            face_detected=face["face_detected"],
            face_bbox=face["bbox"],
            face_det_score=face["det_score"],
            face_bbox_detected=face["bbox_detected"],
            occ_feature=face["occ_feature"],
        )

        if not self.buffer.ready:
            return None

        if (self.frame_count - self.window_size) % self.predict_stride != 0:
            return None

        # ------------------------------------------------------------
        # DMS prediction
        # ------------------------------------------------------------
        arrays = self.buffer.as_arrays()
        pred = self.dms.predict_window(**arrays)

        # ------------------------------------------------------------
        # Lightweight metadata
        # ------------------------------------------------------------
        pred["frame_index"] = self.frame_count - 1
        pred["face_status"] = face["status"]
        pred["face_occ_labels"] = face["occ_labels"]
        pred["face_occ_feature"] = face["occ_feature"]
        pred["body_detected"] = bool(body.get("detected", False))
        pred["hgnet_used"] = bool(face.get("hgnet_used", False))
        pred["restored_regions"] = list(face.get("restored_regions", []))

        # ------------------------------------------------------------
        # Overlay/debug metadata
        # facemesh는 JSONL 저장 시 너무 커지므로 run script에서 제거 권장.
        # ------------------------------------------------------------
        pred["debug"] = {
            "face_bbox": self._to_list(face["bbox"]),
            "face_detected": bool(face["face_detected"]),
            "bbox_detected": bool(face["bbox_detected"]),
            "face_det_score": float(face["det_score"]),
            "face_status": face["status"],

            "face_occ_labels": face["occ_labels"],
            "face_occ_feature": self._to_list(face["occ_feature"]),

            # 최종적으로 DMS에 들어간 landmark.
            # HGNet이 사용된 경우, 가려진 region은 이미 HGNet 값으로 교체되어 있음.
            "facemesh": self._to_list(face["final_landmarks"]),

            # HGNet / restoration info
            "has_occ": bool(face.get("has_occ", False)),
            "hgnet_used": bool(face.get("hgnet_used", False)),
            "hgnet_detected": bool(face.get("hgnet_detected", False)),
            "restored_regions": list(face.get("restored_regions", [])),
        }

        return pred

    def _extract_face_branch(self, face_frame: np.ndarray) -> Dict[str, Any]:
        # ------------------------------------------------------------
        # 1) YOLO-face bbox
        # ------------------------------------------------------------
        fb = self.face_bbox(face_frame)

        bbox_detected = bool(fb.get("detected", False))
        bbox = fb.get("bbox", np.zeros((4,), dtype=np.float32)).astype(np.float32)
        det_score = float(fb.get("det_score", 0.0))

        if not bbox_detected:
            return self._face_fallback("no_yolo_face")

        # ------------------------------------------------------------
        # 2) Occ CNN on YOLO bbox crop
        # ------------------------------------------------------------
        occ = self.occ(face_frame, bbox, bbox_detected)

        occ_probs = occ["probs"].astype(np.float32)  # (4,)
        crop_valid = bool(occ["crop_valid"])

        occ_feature = np.concatenate(
            [
                occ_probs,
                np.array([1.0 if crop_valid else 0.0], dtype=np.float32),
            ],
            axis=0,
        )  # (5,)

        # ------------------------------------------------------------
        # 3) MediaPipe FaceMesh on YOLO bbox crop
        # ------------------------------------------------------------
        mp = self.facemesh(face_frame, bbox, bbox_detected)

        mp_lm = mp["landmarks"]
        mp_detected = bool(mp["detected"])

        if not mp_detected:
            return {
                "final_landmarks": make_zero_facemesh(),
                "face_detected": False,
                "bbox": bbox,
                "det_score": det_score,
                "bbox_detected": bbox_detected,
                "occ_feature": occ_feature,
                "occ_labels": {"left_eye": 0, "right_eye": 0, "nose": 0, "mouth": 0},
                "status": "no_mediapipe_facemesh",

                "has_occ": False,
                "hgnet_detected": False,
                "hgnet_used": False,
                "restored_regions": [],
            }

        # ------------------------------------------------------------
        # 4) Occ labels
        # visible_probs_to_labels:
        #   label 1 = occluded / restore target
        #   label 0 = visible / keep MediaPipe
        # ------------------------------------------------------------
        labels = self.merger.visible_probs_to_labels(occ_probs, crop_valid)
        has_occ = any(int(v) == 1 for v in labels.values())

        # ------------------------------------------------------------
        # 5) Optional HGNet restoration
        # ------------------------------------------------------------
        hg_lm = None
        hg_detected = False

        if has_occ:
            hg = self.hgnet(face_frame, bbox, bbox_detected)
            hg_lm = hg["landmarks"]
            hg_detected = bool(hg["detected"])

        # ------------------------------------------------------------
        # 6) Occ-gated merge
        # final_lm:
        #   visible region -> MediaPipe
        #   occluded region -> HGNet if hg_detected else MediaPipe fallback
        # ------------------------------------------------------------
        final_lm, labels = self.merger.merge(
            mediapipe_lm=mp_lm,
            hgnet_lm_frame=hg_lm,
            occ_probs=occ_probs,
            crop_valid=crop_valid,
            mp_detected=mp_detected,
            hg_detected=hg_detected,
        )

        hgnet_used = bool(has_occ and hg_detected)
        restored_regions = [
            str(k) for k, v in labels.items()
            if int(v) == 1 and hgnet_used
        ]

        if not has_occ:
            status = "ok_no_occ"
        elif hg_detected:
            status = "ok_occ_hgnet"
        else:
            status = "ok_occ_no_hgnet_fallback_mp"

        return {
            "final_landmarks": final_lm,
            "face_detected": bool(mp_detected),
            "bbox": bbox,
            "det_score": det_score,
            "bbox_detected": bbox_detected,
            "occ_feature": occ_feature,
            "occ_labels": labels,
            "status": status,

            "has_occ": bool(has_occ),
            "hgnet_detected": bool(hg_detected),
            "hgnet_used": bool(hgnet_used),
            "restored_regions": restored_regions,
        }

    def _face_fallback(self, status: str) -> Dict[str, Any]:
        return {
            "final_landmarks": make_zero_facemesh(),
            "face_detected": False,
            "bbox": np.zeros((4,), dtype=np.float32),
            "det_score": 0.0,
            "bbox_detected": False,
            "occ_feature": np.array(
                [
                    self.cfg.default_visible_prob,
                    self.cfg.default_visible_prob,
                    self.cfg.default_visible_prob,
                    self.cfg.default_visible_prob,
                    self.cfg.default_crop_valid,
                ],
                dtype=np.float32,
            ),
            "occ_labels": {"left_eye": 0, "right_eye": 0, "nose": 0, "mouth": 0},
            "status": status,

            "has_occ": False,
            "hgnet_detected": False,
            "hgnet_used": False,
            "restored_regions": [],
        }

    @staticmethod
    def _to_list(x: Any) -> Any:
        if hasattr(x, "tolist"):
            return x.tolist()
        return x