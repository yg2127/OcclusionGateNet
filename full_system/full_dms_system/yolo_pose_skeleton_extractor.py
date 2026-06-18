from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from ultralytics import YOLO

from .utils import ensure_bgr_frame


class YoloPoseSkeletonExtractor:
    """Frame-level YOLO-Pose COCO17 skeleton extractor.

    Input: one OpenCV BGR frame.
    Output keys:
        keypoints: (17, 2), xy pixel coordinates
        conf:      (17,)
        bbox:      (4,), x1,y1,x2,y2 for selected person
        det_conf:  float
        detected:  bool
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        img_size: int = 640,
        conf: float = 0.25,
        iou: float = 0.6,
        device: Optional[Union[int, str]] = None,
        select_policy: str = "largest_bbox",
    ):
        self.model_path = str(model_path)
        self.img_size = int(img_size)
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
        self.select_policy = select_policy
        self.kpt_num = 17

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"YOLO-Pose model not found: {self.model_path}")
        self.model = YOLO(self.model_path)

    def __call__(self, frame: np.ndarray) -> Dict[str, Any]:
        return self.extract(frame)

    def extract(self, frame: np.ndarray) -> Dict[str, Any]:
        frame = ensure_bgr_frame(frame)
        results = self.model.predict(
            source=frame,
            imgsz=self.img_size,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        result = results[0]
        selected_idx = self._select_person_index(result)
        if selected_idx is None:
            return self._empty_result()

        keypoints = result.keypoints.xy[selected_idx].cpu().numpy().astype(np.float32)
        if result.keypoints.conf is not None:
            kpt_conf = result.keypoints.conf[selected_idx].cpu().numpy().astype(np.float32)
        else:
            kpt_conf = np.ones((self.kpt_num,), dtype=np.float32)

        if result.boxes is not None and result.boxes.xyxy is not None and len(result.boxes.xyxy) > selected_idx:
            bbox = result.boxes.xyxy[selected_idx].cpu().numpy().astype(np.float32)
            det_conf = float(result.boxes.conf[selected_idx].cpu().item()) if result.boxes.conf is not None else 1.0
        else:
            bbox = np.zeros((4,), dtype=np.float32)
            det_conf = 0.0

        return {
            "keypoints": keypoints,
            "conf": kpt_conf,
            "bbox": bbox,
            "det_conf": det_conf,
            "detected": True,
        }

    def _select_person_index(self, result) -> Optional[int]:
        if result.keypoints is None or result.keypoints.xy is None:
            return None
        num_person = len(result.keypoints.xy)
        if num_person == 0:
            return None
        if self.select_policy == "first":
            return 0
        if self.select_policy == "largest_bbox":
            if result.boxes is None or result.boxes.xyxy is None or len(result.boxes.xyxy) == 0:
                return 0
            boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            return int(np.argmax(areas))
        raise ValueError(f"Unknown select_policy: {self.select_policy}")

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "keypoints": np.zeros((self.kpt_num, 2), dtype=np.float32),
            "conf": np.zeros((self.kpt_num,), dtype=np.float32),
            "bbox": np.zeros((4,), dtype=np.float32),
            "det_conf": 0.0,
            "detected": False,
        }
