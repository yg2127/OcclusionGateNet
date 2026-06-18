from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms as T

from .utils import ensure_bgr_frame, expand_bbox_xyxy, crop_resize_gray, crop_landmarks_to_frame_xy, make_zero_facemesh


class HGNetRestorer:
    """ORFormer-assisted HGNet wrapper.

    This wrapper is optional. If checkpoints are missing or enabled=False, it returns a clean fallback
    instead of stopping the whole DMS system.
    """

    def __init__(
        self,
        orformer_ckpt: Optional[Union[str, Path]] = None,
        hgnet_ckpt: Optional[Union[str, Path]] = None,
        vendor_landmark_root: Optional[Union[str, Path]] = None,
        vendor_orformer_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, int]] = None,
        enabled: bool = True,
        crop_size: int = 112,
        crop_pad_ratio: float = 0.10,
    ):
        self.enabled = bool(enabled)
        self.orformer_ckpt = Path(orformer_ckpt) if orformer_ckpt else None
        self.hgnet_ckpt = Path(hgnet_ckpt) if hgnet_ckpt else None
        self.crop_size = int(crop_size)
        self.crop_pad_ratio = float(crop_pad_ratio)
        dev = f"cuda:{device}" if isinstance(device, int) else (device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        if isinstance(dev, str) and dev.startswith("cuda") and not torch.cuda.is_available():
            dev = "cpu"
        self.device = torch.device(dev)
        self.ready = False
        self.orformer = None
        self.hgnet = None
        self.NORM = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        if not self.enabled:
            return
        if self.orformer_ckpt is None or self.hgnet_ckpt is None:
            return
        if not self.orformer_ckpt.exists() or not self.hgnet_ckpt.exists():
            return

        # This bundle reuses the repository's top-level `landmark/` package (identical to the
        # original vendored copy) instead of duplicating it. External paths are still accepted.
        if vendor_landmark_root is None:
            vendor_landmark_root = Path(__file__).resolve().parents[2] / "landmark"
        vendor_landmark_root = Path(vendor_landmark_root)
        for p in [
            vendor_landmark_root / "src",
            vendor_landmark_root / "src" / "data",
            vendor_landmark_root / "configs",
        ]:
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        if vendor_orformer_path and str(vendor_orformer_path) not in sys.path:
            sys.path.insert(0, str(vendor_orformer_path))

        try:
            from models.VQVAE import VQVAE
            from models.simple_vit import ORFormer
            from models.StackedHGNet import IntergrationStackedHGNet
            from heatmap_gen import denorm_points
            from default import get_cfg

            cfg = get_cfg()
            ds_cfg = cfg.DMD
            vit = ORFormer(image_size=16, patch_size=1, num_classes=2048, dim=256, depth=3,
                           heads=8, mlp_dim=512, channels=256)
            self.orformer = VQVAE(h_dim=128, res_h_dim=32, output_dim=ds_cfg.NUM_EDGE, n_res_layers=2,
                                  n_embeddings=2048, embedding_dim=256, code_dim=256, beta=0.25, vit=vit).to(self.device).eval()
            o_ckpt = torch.load(str(self.orformer_ckpt), map_location=self.device, weights_only=False)
            self.orformer.load_state_dict(o_ckpt.get("model_state_dict", o_ckpt), strict=False)

            edge_info = [list(x) for x in ds_cfg.EDGE_INFO]
            self.hgnet = IntergrationStackedHGNet(
                classes_num=[ds_cfg.NUM_POINT, ds_cfg.NUM_EDGE, ds_cfg.NUM_POINT],
                edge_info=edge_info,
                nstack=4,
            ).to(self.device).eval()
            h_ckpt = torch.load(str(self.hgnet_ckpt), map_location=self.device, weights_only=False)
            self.hgnet.load_state_dict(h_ckpt.get("hgnet_state_dict", h_ckpt.get("model_state_dict", h_ckpt)), strict=True)
            self.denorm_points = denorm_points
            self.ready = True
        except Exception as e:
            print(f"[WARN] HGNetRestorer disabled: {type(e).__name__}: {e}")
            self.ready = False

    def __call__(self, frame_bgr: np.ndarray, face_bbox: np.ndarray, face_detected: bool) -> Dict[str, Any]:
        return self.restore(frame_bgr, face_bbox, face_detected)

    @torch.no_grad()
    def restore(self, frame_bgr: np.ndarray, face_bbox: np.ndarray, face_detected: bool) -> Dict[str, Any]:
        frame_bgr = ensure_bgr_frame(frame_bgr)
        if not self.ready or not face_detected:
            return self._fallback()
        try:
            crop_bbox = expand_bbox_xyxy(face_bbox, frame_bgr.shape, pad_ratio=self.crop_pad_ratio, square=True)
            gray112 = crop_resize_gray(frame_bgr, crop_bbox, self.crop_size)
            face = cv2.resize(gray112, (256, 256))
            rgb = np.stack([face] * 3, axis=-1)
            inp = self.NORM(rgb).unsqueeze(0).to(self.device)
            res = self.NORM(cv2.resize(rgb, (64, 64))).unsqueeze(0).to(self.device)
            with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda"), dtype=torch.float16):
                _, ref_hm, *_ = self.orformer(res)
                _, lm = self.hgnet(inp, reference_heatmaps=ref_hm)
            lm_xy = self.denorm_points(lm.float(), 64, 64).cpu().numpy()[0] * (self.crop_size / 64.0)
            frame_xy = crop_landmarks_to_frame_xy(lm_xy, crop_bbox, crop_size=self.crop_size)
            landmarks = np.concatenate([frame_xy, np.zeros((478, 1), dtype=np.float32)], axis=1).astype(np.float32)
            return {"landmarks": landmarks, "detected": True, "crop_bbox": crop_bbox.astype(np.float32)}
        except Exception:
            return self._fallback()

    def _fallback(self) -> Dict[str, Any]:
        return {"landmarks": make_zero_facemesh(), "detected": False, "crop_bbox": np.zeros((4,), dtype=np.float32)}
