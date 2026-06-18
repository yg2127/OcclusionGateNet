from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from ultralytics import YOLO

from .utils import ensure_bgr_frame


class YoloFaceBBoxExtractor:
    """Frame-level YOLO-face bbox extractor.

    Input: one OpenCV BGR frame.
    Output keys:
        bbox:      (4,), x1,y1,x2,y2
        kps5:      (5,2), if model provides 5 face keypoints, else zeros
        det_score: float
        detected:  bool
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        img_size: int = 640,
        conf: float = 0.25,
        iou: float = 0.6,
        device: Optional[Union[int, str]] = None,
        select_policy: str = "highest_conf",
    ):
        self.model_path = str(model_path)
        self.img_size = int(img_size)
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
        self.select_policy = select_policy
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"YOLO-face model not found: {self.model_path}")
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
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return self._empty_result()

        boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        confs = r.boxes.conf.cpu().numpy().astype(np.float32) if r.boxes.conf is not None else np.ones(len(boxes), dtype=np.float32)
        if self.select_policy == "largest_bbox":
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            idx = int(np.argmax(areas))
        else:
            idx = int(np.argmax(confs))

        kps5 = np.zeros((5, 2), dtype=np.float32)
        if r.keypoints is not None and r.keypoints.xy is not None:
            k = r.keypoints.xy.cpu().numpy().astype(np.float32)
            if len(k) > idx:
                kps5[: min(5, k[idx].shape[0])] = k[idx][:5]

        return {
            "bbox": boxes[idx],
            "kps5": kps5,
            "det_score": float(confs[idx]),
            "detected": True,
        }

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "bbox": np.zeros((4,), dtype=np.float32),
            "kps5": np.zeros((5, 2), dtype=np.float32),
            "det_score": 0.0,
            "detected": False,
        }
