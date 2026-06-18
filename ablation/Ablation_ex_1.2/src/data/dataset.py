"""Multi-task window dataset — masked labels, distraction + gaze 혼합."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from src.data.clip_builder import ClipLabels, ClipRecord
from src.data.preprocess_face import preprocess_face_clip, window_face_detected_ratio
from src.data.preprocess_face5pt import face5pt_detected_ratio, preprocess_face5pt_clip
from src.data.preprocess_pose import preprocess_pose_clip, window_valid_stats
from src.data.window import sliding_window_mosaic_indices


# missing label 표기 용 sentinel id
IGNORE_LABEL = -100     # torch.nn.CrossEntropyLoss 의 ignore_index 호환


@dataclass
class WindowItem:
    x_body: np.ndarray          # (C_pose, Tw, V_pose)
    x_face: np.ndarray          # (C_face, Tw, V_face)
    x_occ: np.ndarray           # (5,) [left_eye, right_eye, nose, mouth, crop_valid_ratio]
    # 라벨 — 지도 없는 head 는 IGNORE_LABEL
    y_action: int
    y_gaze_fine: int
    y_gaze_weak: int
    y_hands: int
    y_talk: int
    source: str                 # "distraction" | "gaze"
    subject_key: str
    clip_id: str
    window_idx: int
    num_windows_in_clip: int


def labels_to_ids(labels: ClipLabels) -> tuple[int, int, int, int, int]:
    def _or(x): return IGNORE_LABEL if x is None else int(x)
    return (_or(labels.action), _or(labels.gaze_fine), _or(labels.gaze_weak),
            _or(labels.hands), _or(labels.talk))


class _VideoCache:
    def __init__(self):
        self._body: dict[str, tuple] = {}
        self._face: dict[str, tuple] = {}
        self._face5pt: dict[str, tuple] = {}
        self._kps5: dict[str, np.ndarray] = {}
        self._occ: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def body(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        if path not in self._body:
            with np.load(path, allow_pickle=True) as d:
                kp = d["keypoints"].astype(np.float32)
                cf = d["conf"].astype(np.float32)
            self._body[path] = (kp, cf)
        return self._body[path]

    def face(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        if path not in self._face:
            with np.load(path, allow_pickle=True) as d:
                lm = d["landmarks"].astype(np.float32)
                dt = d["detected"].astype(bool)
            self._face[path] = (lm, dt)
        return self._face[path]

    def face5pt_meta(self, path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if path not in self._face5pt:
            with np.load(path, allow_pickle=True) as d:
                bbox = d["bbox"].astype(np.float32)
                sc = d["det_score"].astype(np.float32)
                dt = d["detected"].astype(bool)
            self._face5pt[path] = (bbox, sc, dt)
        return self._face5pt[path]

    def kps5(self, path: str) -> np.ndarray:
        if path not in self._kps5:
            with np.load(path, allow_pickle=True) as d:
                self._kps5[path] = d["kps5"].astype(np.float32)
        return self._kps5[path]

    def occ(self, path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if path not in self._occ:
            with np.load(path, allow_pickle=True) as d:
                probs = d["probs"].astype(np.float32)
                computed = d["computed"].astype(bool)
                crop_valid = d["crop_valid"].astype(np.float32)
            self._occ[path] = (probs, computed, crop_valid)
        return self._occ[path]

    def clear(self):
        self._body.clear(); self._face.clear()
        self._face5pt.clear(); self._kps5.clear(); self._occ.clear()



def build_window_occ_vector_from_arrays(
    probs: np.ndarray,
    computed: np.ndarray,
    crop_valid: np.ndarray,
    face_start: int,
    face_end: int,
    default_visible_prob: float = 0.5,
    default_crop_valid: float = 0.0,
) -> np.ndarray:
    """Aggregate sparse frame-level occ predictions into one window vector.

    Output: (5,) = [left_eye, right_eye, nose, mouth, crop_valid_ratio].
    For stride-sampled occ npz, only computed==1 frames are averaged.
    """
    s = max(0, int(face_start))
    e = min(len(probs) - 1, int(face_end))

    if e < s:
        return np.array(
            [
                default_visible_prob,
                default_visible_prob,
                default_visible_prob,
                default_visible_prob,
                default_crop_valid,
            ],
            dtype=np.float32,
        )

    p = probs[s:e + 1]
    m = computed[s:e + 1]
    v = crop_valid[s:e + 1]

    if m.any():
        occ4 = p[m].mean(axis=0).astype(np.float32)
        valid = np.array([v[m].mean()], dtype=np.float32)
    else:
        occ4 = np.full((4,), default_visible_prob, dtype=np.float32)
        valid = np.array([default_crop_valid], dtype=np.float32)

    return np.concatenate([occ4, valid], axis=0).astype(np.float32)

def preload_multitask_windows(
    clips: list[ClipRecord],
    window_size: int = 48,
    window_stride: int = 24,
    max_windows_per_clip: Optional[int] = None,
    pose_min_valid_frames: int = 16,
    pose_min_valid_ratio: float = 0.35,
    pose_min_valid_joint_ratio: float = 0.15,
    face_min_detected_ratio: float = 0.20,
    joint_conf_thres: float = 0.2,
    face_mode: str = "facemesh_full",      # facemesh | facemesh_full | face5pt
    face_use_z: bool = True,
    face_use_detected_channel: bool = True,
    face_use_det_score_channel: bool = True,
    face_bbox_det_thres: float = 0.25,
    occ_cfg: Optional[dict] = None,
    desc: str = "preload",
    logger=None,
) -> list[WindowItem]:
    """ClipRecord 리스트 → window-level WindowItem 리스트. 멀티태스크 라벨 내장."""
    assert face_mode in ("facemesh", "facemesh_full", "face5pt")
    facemesh_needed = face_mode in ("facemesh", "facemesh_full")
    use_region_pool = (face_mode == "facemesh")

    occ_cfg = occ_cfg or {}
    occ_enabled = bool(occ_cfg.get("enabled", False))
    occ_dim = int(occ_cfg.get("dim", 5))
    default_visible_prob = float(occ_cfg.get("default_visible_prob", 0.5))
    default_crop_valid = float(occ_cfg.get("default_crop_valid", 0.0))

    face_to_occ: dict[str, str] = {}
    if occ_enabled:
        map_path = occ_cfg.get("face_npz_to_occ_npz")
        if not map_path:
            raise ValueError("occ.enabled=true but occ.face_npz_to_occ_npz is missing")
        with open(map_path, "r", encoding="utf-8") as f:
            face_to_occ = json.load(f)

    cache = _VideoCache()
    items: list[WindowItem] = []
    skipped_clip_all = 0
    skipped_win = {"pose_quality": 0, "face_quality": 0}

    pbar = tqdm(clips, desc=desc, ncols=120, ascii=True, mininterval=0.3)
    for c in pbar:
        try:
            kp_all, cf_all = cache.body(c.body_npz)
            bb_all, ds_all, bd_all = cache.face5pt_meta(c.face5pt_npz)
            if facemesh_needed:
                lm_all, fd_all = cache.face(c.face_npz)
            if face_mode == "face5pt":
                kps5_all = cache.kps5(c.face5pt_npz)

            occ_arrays = None
            if occ_enabled:
                occ_path = face_to_occ.get(str(c.face_npz))
                if occ_path is not None:
                    occ_arrays = cache.occ(occ_path)
        except Exception as e:
            if logger is not None:
                logger.warning(f"[preload skip] {c.clip_id}: {e}")
            continue

        clip_len = c.mosaic_end - c.mosaic_start + 1
        win_local = sliding_window_mosaic_indices(clip_len, window_size, window_stride)
        if max_windows_per_clip is not None and len(win_local) > max_windows_per_clip:
            idxs = np.linspace(0, len(win_local) - 1, max_windows_per_clip, dtype=int)
            win_local = [win_local[i] for i in idxs]

        num_wins = len(win_local)
        before = len(items)

        kp_clip = kp_all[c.body_start : c.body_end + 1]
        cf_clip = cf_all[c.body_start : c.body_end + 1]
        bb_clip = bb_all[c.face_start : c.face_end + 1]
        ds_clip = ds_all[c.face_start : c.face_end + 1]
        bd_clip = bd_all[c.face_start : c.face_end + 1]
        if facemesh_needed:
            lm_clip = lm_all[c.face_start : c.face_end + 1]
            fd_clip = fd_all[c.face_start : c.face_end + 1]
        if face_mode == "face5pt":
            kps5_clip = kps5_all[c.face_start : c.face_end + 1]

        ya, yg, ygw, yh, yt = labels_to_ids(c.labels)

        for wi, local_idxs in enumerate(win_local):
            body_idxs = local_idxs
            face_idxs = local_idxs

            kp_win = kp_clip[body_idxs]
            cf_win = cf_clip[body_idxs]
            n_valid, valid_ratio = window_valid_stats(cf_win, min_joint_ratio=pose_min_valid_joint_ratio)
            if n_valid < pose_min_valid_frames or valid_ratio < pose_min_valid_ratio:
                skipped_win["pose_quality"] += 1
                continue

            if facemesh_needed:
                det_ratio = window_face_detected_ratio(fd_clip[face_idxs])
            else:
                det_ratio = face5pt_detected_ratio(bd_clip[face_idxs])
            if det_ratio < face_min_detected_ratio:
                skipped_win["face_quality"] += 1
                continue

            bb_win = bb_clip[face_idxs]
            ds_win = ds_clip[face_idxs]
            bd_win = bd_clip[face_idxs]

            x_body = preprocess_pose_clip(
                kp_win, cf_win,
                use_bone=True, use_velocity=True, use_conf_channel=True,
                joint_conf_thres=joint_conf_thres,
            )
            if facemesh_needed:
                lm_win = lm_clip[face_idxs]
                fd_win = fd_clip[face_idxs]
                x_face = preprocess_face_clip(
                    lm_win, fd_win, bb_win, ds_win, bd_win,
                    use_z=face_use_z, use_detected_channel=face_use_detected_channel,
                    bbox_det_thres=face_bbox_det_thres, use_region_pool=use_region_pool,
                )
            else:
                kps5_win = kps5_clip[face_idxs]
                x_face = preprocess_face5pt_clip(
                    kps5_win, bb_win, ds_win, bd_win,
                    use_detected_channel=face_use_detected_channel,
                    use_det_score_channel=face_use_det_score_channel,
                    bbox_det_thres=face_bbox_det_thres,
                )

            if occ_enabled and occ_arrays is not None:
                occ_probs, occ_computed, occ_crop_valid = occ_arrays
                face_abs_start = int(c.face_start + int(face_idxs[0]))
                face_abs_end = int(c.face_start + int(face_idxs[-1]))
                x_occ = build_window_occ_vector_from_arrays(
                    probs=occ_probs,
                    computed=occ_computed,
                    crop_valid=occ_crop_valid,
                    face_start=face_abs_start,
                    face_end=face_abs_end,
                    default_visible_prob=default_visible_prob,
                    default_crop_valid=default_crop_valid,
                )
            elif occ_enabled:
                x_occ = np.array(
                    [
                        default_visible_prob,
                        default_visible_prob,
                        default_visible_prob,
                        default_visible_prob,
                        default_crop_valid,
                    ],
                    dtype=np.float32,
                )
            else:
                x_occ = np.zeros((occ_dim,), dtype=np.float32)

            items.append(WindowItem(
                x_body=x_body, x_face=x_face, x_occ=x_occ,
                y_action=ya, y_gaze_fine=yg, y_gaze_weak=ygw, y_hands=yh, y_talk=yt,
                source=c.source, subject_key=c.subject_key,
                clip_id=c.clip_id, window_idx=wi, num_windows_in_clip=num_wins,
            ))

        if len(items) == before:
            skipped_clip_all += 1

    cache.clear()
    if logger is not None:
        logger.info(f"{desc}: kept={len(items)} clips_all_skipped={skipped_clip_all} skipped_windows={skipped_win}")
    return items


class MemoryMultitaskDataset(Dataset):
    def __init__(self, items: list[WindowItem]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        it = self.items[idx]
        return {
            "x_body": torch.from_numpy(it.x_body),
            "x_face": torch.from_numpy(it.x_face),
            "x_occ": torch.from_numpy(it.x_occ).float(),
            "y_action": torch.tensor(it.y_action, dtype=torch.long),
            "y_gaze_fine": torch.tensor(it.y_gaze_fine, dtype=torch.long),
            "y_gaze_weak": torch.tensor(it.y_gaze_weak, dtype=torch.long),
            "y_hands": torch.tensor(it.y_hands, dtype=torch.long),
            "y_talk": torch.tensor(it.y_talk, dtype=torch.long),
            "source": it.source,
            "subject_key": it.subject_key,
            "clip_id": it.clip_id,
            "window_idx": torch.tensor(it.window_idx, dtype=torch.long),
            "num_windows_in_clip": torch.tensor(it.num_windows_in_clip, dtype=torch.long),
        }
