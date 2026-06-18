from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union
import sys

import numpy as np
import torch
import yaml


class DMSClassifierWrapper:
    """Wrapper around the existing Model4 classifier.

    Expected unbatched inputs:
        body_seq: (T, 17, 2)
        body_conf_seq: (T, 17)
        face_lm_seq: (T, 478, 3)
        face_detected_seq: (T,)
        face_bbox_seq: (T, 4)
        face_det_score_seq: (T,)
        face_bbox_detected_seq: (T,)
        occ_seq: (T, 5) frame-level [left_eye, right_eye, nose, mouth, crop_valid]
    """

    def __init__(
        self,
        config_path: Union[str, Path],
        checkpoint_path: Union[str, Path],
        classifier_root: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, int]] = None,
    ):
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"DMS config not found: {self.config_path}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"DMS checkpoint not found: {self.checkpoint_path}")
        if classifier_root is None:
            # This bundle reuses the repository's top-level `classifier/` package
            # (identical to the original vendored copy) instead of duplicating it.
            classifier_root = Path(__file__).resolve().parents[2] / "classifier"
        self.classifier_root = Path(classifier_root)
        if str(self.classifier_root) not in sys.path:
            sys.path.insert(0, str(self.classifier_root))

        from src.training.builders import build_model
        from src.data.preprocess_pose import preprocess_pose_clip
        from src.data.preprocess_face import preprocess_face_clip

        self.preprocess_pose_clip = preprocess_pose_clip
        self.preprocess_face_clip = preprocess_face_clip

        with self.config_path.open("r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.device = (f"cuda:{device}" if isinstance(device, int) else (device if device is not None else (self.cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")))
        if isinstance(self.device, str) and self.device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"

        self.model, self.model_meta = build_model(self.cfg, str(self.device))
        ckpt = torch.load(str(self.checkpoint_path), map_location=self.device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
        self.model.load_state_dict(sd, strict=True)
        self.model.to(self.device)
        self.model.eval()
        self.window_size = int(self.cfg.get("window", {}).get("size", 48))

    def predict_window(
        self,
        body_seq: np.ndarray,
        body_conf_seq: np.ndarray,
        face_lm_seq: np.ndarray,
        face_detected_seq: np.ndarray,
        face_bbox_seq: np.ndarray,
        face_det_score_seq: np.ndarray,
        face_bbox_detected_seq: np.ndarray,
        occ_seq: np.ndarray,
    ) -> Dict[str, Any]:
        T = body_seq.shape[0]
        if T != self.window_size:
            raise ValueError(f"Expected window T={self.window_size}, got {T}")

        pose_cfg = self.cfg.get("pose", {})
        face_cfg = self.cfg.get("face", {})
        occ_cfg = self.cfg.get("occ", {})

        x_body = self.preprocess_pose_clip(
            body_seq,
            body_conf_seq,
            use_bone=bool(pose_cfg.get("use_bone", True)),
            use_velocity=bool(pose_cfg.get("use_velocity", True)),
            use_conf_channel=bool(pose_cfg.get("use_conf_channel", True)),
            joint_conf_thres=float(pose_cfg.get("joint_conf_thres", 0.2)),
        )

        # The model4 config uses face.mode=facemesh_full and face.encoder=region_pool,
        # so the loader passes raw V=478 to the model and the model pools internally.
        face_mode = face_cfg.get("mode", "facemesh_full")
        use_region_pool = (face_mode == "facemesh")
        x_face = self.preprocess_face_clip(
            face_lm_seq,
            face_detected_seq,
            face_bbox_seq,
            face_det_score_seq,
            face_bbox_detected_seq,
            use_z=bool(face_cfg.get("use_z", True)),
            use_detected_channel=bool(face_cfg.get("use_detected_channel", True)),
            bbox_det_thres=float(face_cfg.get("bbox_det_thres", 0.25)),
            use_region_pool=use_region_pool,
        )

        if bool(occ_cfg.get("enabled", False)):
            occ_arr = np.asarray(occ_seq, dtype=np.float32)
            if occ_arr.ndim != 2 or occ_arr.shape[1] != int(occ_cfg.get("dim", 5)):
                default_visible = float(occ_cfg.get("default_visible_prob", 0.5))
                default_valid = float(occ_cfg.get("default_crop_valid", 0.0))
                x_occ = np.array([default_visible, default_visible, default_visible, default_visible, default_valid], dtype=np.float32)
            else:
                x_occ = np.nanmean(occ_arr, axis=0).astype(np.float32)
                x_occ = np.nan_to_num(x_occ, nan=float(occ_cfg.get("default_visible_prob", 0.5)))
        else:
            x_occ = np.zeros((int(occ_cfg.get("dim", 5)),), dtype=np.float32)

        xb = torch.from_numpy(x_body).unsqueeze(0).to(self.device)
        xf = torch.from_numpy(x_face).unsqueeze(0).to(self.device)
        xo = torch.from_numpy(x_occ).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            logits = self.model(xb, xf, x_occ=xo)
            out = {}
            for head, z in logits.items():
                prob = torch.softmax(z, dim=-1)[0].detach().cpu().numpy().astype(np.float32)
                pred = int(prob.argmax())
                out[head] = {
                    "pred": pred,
                    "prob": prob,
                    "confidence": float(prob[pred]),
                    "logits": z[0].detach().cpu().numpy().astype(np.float32),
                }
        out["x_occ_window"] = x_occ
        return out
