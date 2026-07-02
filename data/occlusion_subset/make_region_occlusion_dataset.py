from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm


# ============================================================
# Paths
# ============================================================

SRC_ROOT = Path("/data/shared/DMD_landmarks/facemesh")
VIDEO_ROOT = Path("/data/shared/DMD")

# 새 데이터셋 루트
OUT_ROOT = Path("/data/shared/Occlusion_subset_dataset/region_occlusion_cnn_dataset_v2_facecrop_256")
MANIFEST_PATH = OUT_ROOT / "labels.jsonl"

TARGET_GROUPS = ["gA", "gB", "gC"]
NPZ_PATTERN = "*_ir_face_facemesh.npz"


# ============================================================
# Sampling config
# ============================================================

SEED = 42

FRAMES_PER_VIDEO_PER_REGION = 12
FRAMES_PER_VIDEO_CLEAN = 8

MAX_VIDEOS_TOTAL: Optional[int] = None
MIN_FRAME_GAP = 25

SAVE_SIZE = 256
JPG_QUALITY = 95

# 다양한 synthetic occlusion appearance
APPEARANCE_TYPES = [
    "solid",
    "noise",
    "smooth_noise",
    "stripe",
    "checker",
    "soft_solid",
    "soft_noise",
    "blur_patch",
]

SOLID_GRAY_VALUES = [0, 32, 64, 96, 128, 160, 192, 224, 255]


# ============================================================
# Face crop config
# ============================================================

MIN_VALID_LANDMARKS = 100
MIN_FACE_BOX_SIZE = 25
MAX_FACE_BOX_RATIO = 0.98

# 1.25~1.45 권장. 너무 타이트하면 턱/이마 잘림.
FACE_PAD_FACTOR = 1.35

# 얼굴 crop 중심 보정
CENTER_X_SHIFT_RATIO = 0.00
CENTER_Y_SHIFT_RATIO = 0.03


# ============================================================
# Region / label config
# ============================================================

# 최종 목적용 3-label
LABEL_NAMES = [
    "left_eye",
    "right_eye",
    "mouth",
]

# 학습에 사용할 region
# both_eyes는 label [1,1,0]으로 생성.
REGION_NAMES = [
    "left_eye",
    "right_eye",
    "both_eyes",
    "mouth",
]

# half-face는 일단 제외 권장.
# overlap 기반 label 만들 때 나중에 추가.
# REGION_NAMES += ["left_face", "right_face"]


# ============================================================
# FaceMesh 478 indices
# 주의:
# 여기서는 기존 코드 naming을 그대로 유지.
# 네 데이터에서 left/right가 사람 기준으로 맞는지 overlay로 확인했으면 그대로 사용.
# ============================================================

FACE_REGIONS: Dict[str, List[int]] = {
    "left_eye": [
        33, 7, 163, 144, 145, 153, 154, 155,
        133, 173, 157, 158, 159, 160, 161, 246,
    ],
    "right_eye": [
        362, 382, 381, 380, 374, 373, 390, 249,
        263, 466, 388, 387, 386, 385, 384, 398,
    ],
    "left_iris": [468, 469, 470, 471, 472],
    "right_iris": [473, 474, 475, 476, 477],
    "mouth": [
        61, 146, 91, 181, 84, 17, 314, 405,
        321, 375, 291, 308, 324, 318, 402, 317,
        14, 87, 178, 88, 95, 78, 191, 80,
        81, 82, 13, 312, 311, 310, 415,
    ],
}

EYE_AREA_EXTRA_LEFT = [
    22, 23, 24, 25, 26, 27, 28, 29, 30,
    110, 112, 113, 124, 130, 143, 156,
    189, 190, 221, 222, 223, 224, 225,
]

EYE_AREA_EXTRA_RIGHT = [
    252, 253, 254, 255, 256, 257, 258, 259, 260,
    339, 341, 342, 353, 359, 372, 383,
    413, 414, 441, 442, 443, 444, 445,
]

MOUTH_AREA_EXTRA = [
    0, 11, 12, 15, 16, 18,
    37, 38, 39, 40,
    72, 73, 74, 85, 86, 89, 90, 96,
    164, 165, 167, 175, 179, 180, 199, 200,
    201, 202, 204, 208, 210, 211, 212, 214, 216,
    302, 303, 304, 315, 316, 319, 320, 325,
    391, 392, 393, 394, 403, 404, 406,
    421, 422, 424, 428, 430, 431, 432, 434, 436,
]


def unique(xs: List[int]) -> List[int]:
    return sorted(set(xs))


REGION_LANDMARKS: Dict[str, List[int]] = {
    "left_eye": unique(
        FACE_REGIONS["left_eye"]
        + FACE_REGIONS["left_iris"]
        + EYE_AREA_EXTRA_LEFT
    ),
    "right_eye": unique(
        FACE_REGIONS["right_eye"]
        + FACE_REGIONS["right_iris"]
        + EYE_AREA_EXTRA_RIGHT
    ),
    "both_eyes": unique(
        FACE_REGIONS["left_eye"]
        + FACE_REGIONS["right_eye"]
        + FACE_REGIONS["left_iris"]
        + FACE_REGIONS["right_iris"]
        + EYE_AREA_EXTRA_LEFT
        + EYE_AREA_EXTRA_RIGHT
        + [6, 8, 9, 168, 195, 197]
    ),
    "mouth": unique(
        FACE_REGIONS["mouth"] + MOUTH_AREA_EXTRA
    ),
}


# ============================================================
# NPZ helpers
# ============================================================

LANDMARK_KEY_CANDIDATES = [
    "landmarks",
    "facemesh",
    "face_landmarks",
    "points",
    "coords",
    "arr_0",
]


def find_landmark_key(npz: np.lib.npyio.NpzFile) -> str:
    for k in LANDMARK_KEY_CANDIDATES:
        if k in npz.files:
            arr = npz[k]
            if isinstance(arr, np.ndarray) and arr.ndim >= 3:
                return k

    for k in npz.files:
        arr = npz[k]
        if isinstance(arr, np.ndarray) and arr.ndim >= 3:
            return k

    raise ValueError(f"No landmark-like array found. keys={npz.files}")


def to_tvc(arr: np.ndarray) -> Tuple[np.ndarray, str]:
    arr = np.asarray(arr)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D landmark array, got shape={arr.shape}")

    _, a, b = arr.shape

    # common: (T, V, C)
    if a >= 100 and b <= 16:
        return arr.copy(), "TVC"

    # possible: (T, C, V)
    if b >= 100 and a <= 16:
        return np.transpose(arr, (0, 2, 1)).copy(), "TCV"

    raise ValueError(f"Ambiguous landmark shape: {arr.shape}")


def load_landmarks(src_npz: Path) -> np.ndarray:
    with np.load(str(src_npz), allow_pickle=True) as data:
        key = find_landmark_key(data)
        lm, _ = to_tvc(data[key])
    return lm


def valid_points(points: np.ndarray) -> np.ndarray:
    c = points.shape[-1]
    coord_ch = min(3, c)
    finite = np.isfinite(points[:, :coord_ch]).all(axis=1)
    not_zero = ~np.all(np.isclose(points[:, :coord_ch], 0.0, atol=1e-8), axis=1)
    return finite & not_zero


def landmark_xy_to_frame_xy_float(
    xy: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> np.ndarray:
    pts = xy.copy().astype(np.float32)

    if pts.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]

    if pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)

    max_x = float(np.nanmax(pts[:, 0]))
    max_y = float(np.nanmax(pts[:, 1]))

    # normalized coords
    if max_x <= 2.0 and max_y <= 2.0:
        pts[:, 0] *= frame_w
        pts[:, 1] *= frame_h

    # 너무 멀리 튄 점 제거
    valid = (
        np.isfinite(pts).all(axis=1)
        & (pts[:, 0] >= -0.15 * frame_w)
        & (pts[:, 0] <= 1.15 * frame_w)
        & (pts[:, 1] >= -0.15 * frame_h)
        & (pts[:, 1] <= 1.15 * frame_h)
    )

    pts = pts[valid]
    pts[:, 0] = np.clip(pts[:, 0], 0, frame_w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, frame_h - 1)

    return pts.astype(np.float32)


def landmark_xy_to_frame_xy_int(
    xy: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> np.ndarray:
    pts = landmark_xy_to_frame_xy_float(xy, frame_w, frame_h)
    return np.round(pts).astype(np.int32)


# ============================================================
# File matching
# ============================================================

def collect_original_npz_files() -> List[Path]:
    files: List[Path] = []

    for g in TARGET_GROUPS:
        root = SRC_ROOT / "distraction" / "dmd" / g
        if not root.exists():
            print(f"[WARN] missing group root: {root}")
            continue

        files.extend(sorted(root.rglob(NPZ_PATTERN)))

    return files


def find_matching_video(src_npz: Path) -> Optional[Path]:
    rel = src_npz.relative_to(SRC_ROOT)
    video_dir = VIDEO_ROOT / rel.parent
    name = src_npz.name

    candidates: List[Path] = []

    if name.endswith("_ir_face_facemesh.npz"):
        base = name.replace("_ir_face_facemesh.npz", "")
        candidates.extend([
            video_dir / f"{base}_ir_face.mp4",
            video_dir / f"{base}_ir_body.mp4",
            video_dir / f"{base}_rgb_body.mp4",
        ])

    if video_dir.exists():
        stem = name.replace("_ir_face_facemesh.npz", "")
        candidates.extend(sorted(video_dir.glob(f"{stem}*.mp4")))

    seen = set()
    unique_candidates = []

    for p in candidates:
        if p not in seen:
            seen.add(p)
            unique_candidates.append(p)

    for p in unique_candidates:
        if p.exists():
            return p

    return None


def stable_shuffle(xs: List[Tuple[Path, Path]], seed: int) -> List[Tuple[Path, Path]]:
    rng = random.Random(seed)
    ys = list(xs)
    rng.shuffle(ys)
    return ys


# ============================================================
# Face crop geometry
# ============================================================

def build_face_crop_bbox(
    lm_frame: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> Optional[Tuple[int, int, int, int, Dict]]:
    valid = valid_points(lm_frame)
    pts_raw = lm_frame[valid]

    if pts_raw.shape[0] < MIN_VALID_LANDMARKS:
        return None

    pts = landmark_xy_to_frame_xy_float(pts_raw[:, :2], frame_w, frame_h)

    if pts.shape[0] < MIN_VALID_LANDMARKS:
        return None

    x1 = float(np.min(pts[:, 0]))
    y1 = float(np.min(pts[:, 1]))
    x2 = float(np.max(pts[:, 0]))
    y2 = float(np.max(pts[:, 1]))

    bw = x2 - x1
    bh = y2 - y1

    if bw < MIN_FACE_BOX_SIZE or bh < MIN_FACE_BOX_SIZE:
        return None

    if bw > frame_w * MAX_FACE_BOX_RATIO or bh > frame_h * MAX_FACE_BOX_RATIO:
        return None

    cx = (x1 + x2) / 2.0 + bw * CENTER_X_SHIFT_RATIO
    cy = (y1 + y2) / 2.0 + bh * CENTER_Y_SHIFT_RATIO

    side = max(bw, bh) * FACE_PAD_FACTOR

    sx1 = int(round(cx - side / 2.0))
    sy1 = int(round(cy - side / 2.0))
    sx2 = int(round(cx + side / 2.0))
    sy2 = int(round(cy + side / 2.0))

    sx1 = max(0, sx1)
    sy1 = max(0, sy1)
    sx2 = min(frame_w, sx2)
    sy2 = min(frame_h, sy2)

    if sx2 <= sx1 or sy2 <= sy1:
        return None

    crop_w = sx2 - sx1
    crop_h = sy2 - sy1

    if crop_w < MIN_FACE_BOX_SIZE or crop_h < MIN_FACE_BOX_SIZE:
        return None

    info = {
        "face_raw_bbox_xyxy": [x1, y1, x2, y2],
        "face_crop_xyxy": [sx1, sy1, sx2, sy2],
        "face_crop_w": crop_w,
        "face_crop_h": crop_h,
        "face_pad_factor": FACE_PAD_FACTOR,
        "num_valid_landmarks": int(pts.shape[0]),
    }

    return sx1, sy1, sx2, sy2, info


def crop_frame_to_256_bgr(
    frame_bgr: np.ndarray,
    crop_xyxy: Tuple[int, int, int, int],
) -> np.ndarray:
    x1, y1, x2, y2 = crop_xyxy
    crop = frame_bgr[y1:y2, x1:x2]

    if crop.size == 0:
        raise RuntimeError("Empty face crop")

    crop_256 = cv2.resize(crop, (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_AREA)
    return crop_256


def transform_points_frame_to_crop256(
    pts_frame_xy: np.ndarray,
    crop_xyxy: Tuple[int, int, int, int],
) -> np.ndarray:
    x1, y1, x2, y2 = crop_xyxy
    crop_w = max(1, x2 - x1)
    crop_h = max(1, y2 - y1)

    pts = pts_frame_xy.astype(np.float32).copy()
    pts[:, 0] = (pts[:, 0] - x1) * (SAVE_SIZE / crop_w)
    pts[:, 1] = (pts[:, 1] - y1) * (SAVE_SIZE / crop_h)

    pts[:, 0] = np.clip(pts[:, 0], 0, SAVE_SIZE - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, SAVE_SIZE - 1)

    return pts.astype(np.float32)


# ============================================================
# Region polygon generation in crop-256 coordinate
# ============================================================

def bbox_poly_from_points_crop(
    pts: np.ndarray,
    pad_x: int,
    pad_y: int,
    y_percentile: Optional[float] = None,
) -> np.ndarray:
    if len(pts) < 3:
        return np.zeros((0, 2), dtype=np.int32)

    use_pts = pts

    if y_percentile is not None:
        y_cut = np.percentile(pts[:, 1], y_percentile)
        filtered = pts[pts[:, 1] <= y_cut]
        if len(filtered) >= 3:
            use_pts = filtered

    x1 = int(np.floor(use_pts[:, 0].min())) - pad_x
    y1 = int(np.floor(use_pts[:, 1].min())) - pad_y
    x2 = int(np.ceil(use_pts[:, 0].max())) + pad_x
    y2 = int(np.ceil(use_pts[:, 1].max())) + pad_y

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(SAVE_SIZE - 1, x2)
    y2 = min(SAVE_SIZE - 1, y2)

    return np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.int32,
    )


def hull_poly_from_points_crop(
    pts: np.ndarray,
    padding: int,
) -> np.ndarray:
    if len(pts) < 3:
        return np.zeros((0, 2), dtype=np.int32)

    offsets = np.array([
        [0, 0],
        [-padding, 0],
        [padding, 0],
        [0, -padding],
        [0, padding],
        [-padding, -padding],
        [-padding, padding],
        [padding, -padding],
        [padding, padding],
    ], dtype=np.float32)

    expanded = []
    for p in pts:
        expanded.append(p[None, :] + offsets)

    expanded = np.vstack(expanded)

    expanded[:, 0] = np.clip(expanded[:, 0], 0, SAVE_SIZE - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, SAVE_SIZE - 1)

    hull = cv2.convexHull(np.round(expanded).astype(np.int32).reshape(-1, 1, 2))
    return hull.reshape(-1, 2)


def get_region_poly_crop256(
    lm_frame: np.ndarray,
    region: str,
    frame_w: int,
    frame_h: int,
    crop_xyxy: Tuple[int, int, int, int],
) -> np.ndarray:
    if region not in REGION_LANDMARKS:
        raise KeyError(f"Unknown region: {region}")

    idx = [i for i in REGION_LANDMARKS[region] if i < lm_frame.shape[0]]
    pts_raw = lm_frame[idx]

    valid = valid_points(pts_raw)
    pts_raw = pts_raw[valid]

    if len(pts_raw) < 3:
        return np.zeros((0, 2), dtype=np.int32)

    pts_frame = landmark_xy_to_frame_xy_float(pts_raw[:, :2], frame_w, frame_h)
    pts_crop = transform_points_frame_to_crop256(pts_frame, crop_xyxy)

    if len(pts_crop) < 3:
        return np.zeros((0, 2), dtype=np.int32)

    if region in ["left_eye", "right_eye"]:
        return bbox_poly_from_points_crop(
            pts_crop,
            pad_x=18,
            pad_y=14,
            y_percentile=92.0,
        )

    if region == "both_eyes":
        return bbox_poly_from_points_crop(
            pts_crop,
            pad_x=22,
            pad_y=16,
            y_percentile=92.0,
        )

    if region == "mouth":
        return hull_poly_from_points_crop(
            pts_crop,
            padding=12,
        )

    raise ValueError(f"Unhandled region: {region}")


# ============================================================
# Mask appearance
# ============================================================

def polygon_mask(shape_hw: Tuple[int, int], poly: np.ndarray) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)

    if len(poly) >= 3:
        cv2.fillConvexPoly(mask, poly.reshape(-1, 1, 2), 255)

    return mask


def random_gray(rng: random.Random) -> int:
    return rng.choice(SOLID_GRAY_VALUES)


def make_pattern(
    h: int,
    w: int,
    appearance: str,
    rng: random.Random,
    src_gray: Optional[np.ndarray] = None,
) -> np.ndarray:
    if appearance == "solid":
        v = random_gray(rng)
        return np.full((h, w), v, dtype=np.uint8)

    if appearance == "noise":
        lo = rng.randint(0, 120)
        hi = rng.randint(135, 255)
        if lo > hi:
            lo, hi = hi, lo

        return np.random.default_rng(rng.randint(0, 10**9)).integers(
            lo, hi + 1,
            size=(h, w),
            dtype=np.uint8,
        )

    if appearance == "smooth_noise":
        small_h = max(4, h // rng.choice([8, 12, 16]))
        small_w = max(4, w // rng.choice([8, 12, 16]))

        noise = np.random.default_rng(rng.randint(0, 10**9)).integers(
            0, 256,
            size=(small_h, small_w),
            dtype=np.uint8,
        )

        noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
        noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=rng.uniform(1.0, 3.0))
        return noise.astype(np.uint8)

    if appearance == "stripe":
        bg = rng.randint(20, 230)
        fg = rng.randint(20, 230)
        period = rng.choice([6, 8, 10, 12, 16])
        thickness = rng.choice([2, 3, 4, 5])
        angle = rng.choice(["vertical", "horizontal", "diag"])

        img = np.full((h, w), bg, dtype=np.uint8)

        if angle == "vertical":
            for x in range(0, w, period):
                img[:, x:x + thickness] = fg
        elif angle == "horizontal":
            for y in range(0, h, period):
                img[y:y + thickness, :] = fg
        else:
            for k in range(-h, w, period):
                for t in range(thickness):
                    x0 = max(0, k + t)
                    y0 = max(0, -k - t)
                    x1 = min(w - 1, k + h + t)
                    y1 = min(h - 1, h)
                    cv2.line(img, (x0, y0), (x1, y1), int(fg), 1)

        return img

    if appearance == "checker":
        v1 = rng.randint(0, 130)
        v2 = rng.randint(130, 255)
        cell = rng.choice([6, 8, 10, 12, 16])

        img = np.zeros((h, w), dtype=np.uint8)

        for yy in range(0, h, cell):
            for xx in range(0, w, cell):
                val = v1 if ((yy // cell + xx // cell) % 2 == 0) else v2
                img[yy:yy + cell, xx:xx + cell] = val

        img = cv2.GaussianBlur(img, (3, 3), 0)
        return img

    if appearance == "soft_solid":
        v = random_gray(rng)
        img = np.full((h, w), v, dtype=np.uint8)

        noise = np.random.default_rng(rng.randint(0, 10**9)).normal(
            loc=0,
            scale=rng.uniform(3, 12),
            size=(h, w),
        )

        img = np.clip(img.astype(np.float32) + noise, 0, 255)
        return img.astype(np.uint8)

    if appearance == "soft_noise":
        return make_pattern(h, w, "smooth_noise", rng, src_gray)

    if appearance == "blur_patch":
        if src_gray is None:
            return make_pattern(h, w, "smooth_noise", rng, src_gray)

        patch = cv2.resize(src_gray, (w, h), interpolation=cv2.INTER_LINEAR)
        k = rng.choice([9, 13, 17, 21, 25])
        patch = cv2.GaussianBlur(patch, (k, k), 0)

        shift = rng.randint(-60, 60)
        patch = np.clip(patch.astype(np.int16) + shift, 0, 255).astype(np.uint8)

        return patch

    raise ValueError(f"Unknown appearance: {appearance}")


def apply_pattern_mask_to_crop(
    crop_bgr_256: np.ndarray,
    poly_crop: np.ndarray,
    appearance: str,
    rng: random.Random,
) -> np.ndarray:
    out = crop_bgr_256.copy()
    h, w = out.shape[:2]

    if len(poly_crop) < 3:
        return out

    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

    hard_mask = polygon_mask((h, w), poly_crop)
    if hard_mask.max() == 0:
        return out

    pattern = make_pattern(h, w, appearance, rng, src_gray=gray)

    if appearance in ["soft_solid", "soft_noise", "blur_patch"]:
        alpha = cv2.GaussianBlur(hard_mask, (0, 0), sigmaX=rng.uniform(2.0, 5.0))
    else:
        alpha = hard_mask

    alpha_f = (alpha.astype(np.float32) / 255.0)[..., None]

    pattern_bgr = cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR).astype(np.float32)
    out_f = out.astype(np.float32)

    mixed = out_f * (1.0 - alpha_f) + pattern_bgr * alpha_f
    mixed = np.clip(mixed, 0, 255).astype(np.uint8)

    return mixed


def to_gray_256(frame_bgr_256: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr_256, cv2.COLOR_BGR2GRAY)
    return gray


# ============================================================
# Frame sampling
# ============================================================

def choose_frame_indices(
    total_frames: int,
    n: int,
    min_gap: int,
    rng: random.Random,
) -> List[int]:
    if total_frames <= 0:
        return []

    if total_frames < n * min_gap:
        min_gap = max(1, total_frames // max(n, 1))

    candidates = list(range(0, total_frames))
    rng.shuffle(candidates)

    selected: List[int] = []

    for idx in candidates:
        if all(abs(idx - s) >= min_gap for s in selected):
            selected.append(idx)
            if len(selected) >= n:
                break

    if len(selected) < n:
        selected = np.linspace(
            0,
            total_frames - 1,
            num=min(n, total_frames),
            dtype=int,
        ).tolist()

    return sorted(set(int(x) for x in selected))


def read_specific_frames(
    video_path: Path,
    frame_indices: List[int],
) -> Dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    out: Dict[int, np.ndarray] = {}

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            out[int(idx)] = frame

    cap.release()
    return out


# ============================================================
# Label helpers
# ============================================================

def make_label(region: Optional[str]) -> Dict[str, int]:
    label = {k: 0 for k in LABEL_NAMES}

    if region is None:
        return label

    if region == "left_eye":
        label["left_eye"] = 1
    elif region == "right_eye":
        label["right_eye"] = 1
    elif region == "both_eyes":
        label["left_eye"] = 1
        label["right_eye"] = 1
    elif region == "mouth":
        label["mouth"] = 1
    else:
        raise ValueError(f"Unknown region for 3-label setup: {region}")

    return label


def label_vector(label: Dict[str, int]) -> List[int]:
    return [int(label[k]) for k in LABEL_NAMES]


# ============================================================
# Save helpers
# ============================================================

def save_image(path: Path, img_gray: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(
        str(path),
        img_gray,
        [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY],
    )

    if not ok:
        raise RuntimeError(f"Failed to save image: {path}")


# ============================================================
# Main processing
# ============================================================

def process_video(
    src_npz: Path,
    video_path: Path,
    fw,
    video_index: int,
    rng: random.Random,
) -> Tuple[int, int]:
    landmarks = load_landmarks(src_npz)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    lm_frame_count = int(landmarks.shape[0])
    total_frames = min(video_frame_count, lm_frame_count)

    if total_frames <= 0:
        return 0, 0

    saved = 0
    failed = 0

    rel = src_npz.relative_to(SRC_ROOT)
    rel_stem = "_".join(rel.with_suffix("").parts)

    clean_indices = choose_frame_indices(
        total_frames=total_frames,
        n=FRAMES_PER_VIDEO_CLEAN,
        min_gap=MIN_FRAME_GAP,
        rng=rng,
    )

    region_to_indices: Dict[str, List[int]] = {}

    for region in REGION_NAMES:
        region_to_indices[region] = choose_frame_indices(
            total_frames=total_frames,
            n=FRAMES_PER_VIDEO_PER_REGION,
            min_gap=MIN_FRAME_GAP,
            rng=rng,
        )

    all_needed = sorted(
        set(clean_indices + [i for xs in region_to_indices.values() for i in xs])
    )

    frames = read_specific_frames(video_path, all_needed)

    # clean negative 저장
    for fi in clean_indices:
        if fi not in frames:
            failed += 1
            continue

        frame = frames[fi]
        h, w = frame.shape[:2]

        bbox_result = build_face_crop_bbox(
            lm_frame=landmarks[fi],
            frame_w=w,
            frame_h=h,
        )

        if bbox_result is None:
            failed += 1
            continue

        x1, y1, x2, y2, crop_info = bbox_result

        try:
            crop_bgr = crop_frame_to_256_bgr(frame, (x1, y1, x2, y2))
            gray = to_gray_256(crop_bgr)
        except Exception:
            failed += 1
            continue

        out_name = f"{rel_stem}_f{fi:06d}_clean_facecrop256.jpg"
        out_path = OUT_ROOT / "images" / "clean" / out_name

        save_image(out_path, gray)

        label = make_label(None)

        meta = {
            "image_path": str(out_path),
            "src_npz": str(src_npz),
            "video_path": str(video_path),
            "frame_idx": int(fi),
            "region": "clean",
            "appearance": "none",
            "labels": label,
            "label_vector": label_vector(label),
            "label_names": LABEL_NAMES,
            "video_index": int(video_index),
            "crop_info": crop_info,
        }

        fw.write(json.dumps(meta, ensure_ascii=False) + "\n")
        saved += 1

    # masked positives 저장
    for region, indices in region_to_indices.items():
        for fi in indices:
            if fi not in frames:
                failed += 1
                continue

            frame = frames[fi]
            h, w = frame.shape[:2]

            bbox_result = build_face_crop_bbox(
                lm_frame=landmarks[fi],
                frame_w=w,
                frame_h=h,
            )

            if bbox_result is None:
                failed += 1
                continue

            x1, y1, x2, y2, crop_info = bbox_result
            crop_xyxy = (x1, y1, x2, y2)

            try:
                crop_bgr = crop_frame_to_256_bgr(frame, crop_xyxy)

                poly_crop = get_region_poly_crop256(
                    lm_frame=landmarks[fi],
                    region=region,
                    frame_w=w,
                    frame_h=h,
                    crop_xyxy=crop_xyxy,
                )

                if len(poly_crop) < 3:
                    failed += 1
                    continue

                appearance = rng.choice(APPEARANCE_TYPES)

                masked_crop = apply_pattern_mask_to_crop(
                    crop_bgr_256=crop_bgr,
                    poly_crop=poly_crop,
                    appearance=appearance,
                    rng=rng,
                )

                gray = to_gray_256(masked_crop)

            except Exception:
                failed += 1
                continue

            out_name = f"{rel_stem}_f{fi:06d}_{region}_{appearance}_facecrop256.jpg"
            out_path = OUT_ROOT / "images" / region / appearance / out_name

            save_image(out_path, gray)

            label = make_label(region)

            meta = {
                "image_path": str(out_path),
                "src_npz": str(src_npz),
                "video_path": str(video_path),
                "frame_idx": int(fi),
                "region": region,
                "appearance": appearance,
                "labels": label,
                "label_vector": label_vector(label),
                "label_names": LABEL_NAMES,
                "video_index": int(video_index),
                "crop_info": crop_info,
                "poly_crop256": poly_crop.astype(int).tolist(),
            }

            fw.write(json.dumps(meta, ensure_ascii=False) + "\n")
            saved += 1

    return saved, failed


def main() -> None:
    rng = random.Random(SEED)
    np.random.seed(SEED)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "images").mkdir(parents=True, exist_ok=True)

    src_files = collect_original_npz_files()
    print(f"[INFO] found npz files: {len(src_files)}")

    valid_pairs: List[Tuple[Path, Path]] = []

    for src_npz in tqdm(src_files, desc="match videos"):
        video_path = find_matching_video(src_npz)
        if video_path is not None:
            valid_pairs.append((src_npz, video_path))

    print(f"[INFO] valid npz-video pairs: {len(valid_pairs)}")

    if len(valid_pairs) == 0:
        raise RuntimeError("No valid npz-video pairs found.")

    valid_pairs = stable_shuffle(valid_pairs, SEED)

    if MAX_VIDEOS_TOTAL is not None:
        valid_pairs = valid_pairs[:MAX_VIDEOS_TOTAL]

    print(f"[INFO] videos to process: {len(valid_pairs)}")
    print(f"[INFO] output root: {OUT_ROOT}")
    print(f"[INFO] manifest: {MANIFEST_PATH}")
    print(f"[INFO] save size: {SAVE_SIZE}")
    print(f"[INFO] label names: {LABEL_NAMES}")
    print(f"[INFO] regions: {REGION_NAMES}")
    print(f"[INFO] appearance types: {APPEARANCE_TYPES}")

    total_saved = 0
    total_failed = 0

    with MANIFEST_PATH.open("w", encoding="utf-8") as fw:
        for vi, (src_npz, video_path) in enumerate(tqdm(valid_pairs, desc="process videos")):
            try:
                saved, failed = process_video(
                    src_npz=src_npz,
                    video_path=video_path,
                    fw=fw,
                    video_index=vi,
                    rng=rng,
                )

                total_saved += saved
                total_failed += failed

            except Exception as e:
                total_failed += 1

                err = {
                    "src_npz": str(src_npz),
                    "video_path": str(video_path),
                    "error": f"{type(e).__name__}: {e}",
                }

                fw.write(json.dumps(err, ensure_ascii=False) + "\n")
                print(f"[ERROR] {src_npz}: {type(e).__name__}: {e}")

    print("\n[DONE]")
    print(f"saved images : {total_saved}")
    print(f"failed       : {total_failed}")
    print(f"output root  : {OUT_ROOT}")
    print(f"manifest     : {MANIFEST_PATH}")


if __name__ == "__main__":
    main()