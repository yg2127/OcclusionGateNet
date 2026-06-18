from __future__ import annotations

from typing import Any, Dict
import cv2
import numpy as np

from .utils import ensure_bgr_frame, expand_bbox_xyxy, make_zero_facemesh


class MediaPipeFaceMeshOnYoloCrop:
    """MediaPipe FaceMesh on a YOLO-face ROI, returning full-frame coordinates."""

    def __init__(
        self,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.3,
        pad_ratio: float = 0.2,
        static_image_mode: bool = True,
    ):
        import mediapipe as mp

        self.pad_ratio = float(pad_ratio)
        self.fm = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=1,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
        )

    def close(self) -> None:
        if self.fm is not None:
            self.fm.close()
            self.fm = None

    def __call__(self, frame_bgr: np.ndarray, face_bbox: np.ndarray, face_detected: bool) -> Dict[str, Any]:
        return self.extract(frame_bgr, face_bbox, face_detected)

    def extract(self, frame_bgr: np.ndarray, face_bbox: np.ndarray, face_detected: bool) -> Dict[str, Any]:
        frame_bgr = ensure_bgr_frame(frame_bgr)
        if not face_detected:
            return self._fallback()
        try:
            crop_bbox = expand_bbox_xyxy(face_bbox, frame_bgr.shape, pad_ratio=self.pad_ratio, square=False)
            x1, y1, x2, y2 = [int(round(x)) for x in crop_bbox]
            roi = frame_bgr[y1:y2, x1:x2]
            if roi.size == 0:
                return self._fallback()
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            res = self.fm.process(roi_rgb)
            if not res.multi_face_landmarks:
                return self._fallback(crop_bbox)
            lm = res.multi_face_landmarks[0].landmark
            rh, rw = roi.shape[:2]
            out = np.empty((478, 3), dtype=np.float32)
            for k, p in enumerate(lm[:478]):
                out[k, 0] = p.x * rw + x1
                out[k, 1] = p.y * rh + y1
                out[k, 2] = p.z * rw
            if out.shape[0] != 478:
                return self._fallback(crop_bbox)
            return {"landmarks": out, "detected": True, "crop_bbox": crop_bbox.astype(np.float32)}
        except Exception:
            return self._fallback()

    def _fallback(self, crop_bbox=None) -> Dict[str, Any]:
        return {
            "landmarks": make_zero_facemesh(),
            "detected": False,
            "crop_bbox": np.zeros((4,), dtype=np.float32) if crop_bbox is None else np.asarray(crop_bbox, dtype=np.float32),
        }
