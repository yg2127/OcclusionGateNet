from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm


# ============================================================
# Paths
# ============================================================

SRC_ROOT = Path("/data/shared/DMD_landmarks/facemesh")
OCC_ROOT = Path("/data/shared/DMD_landmarks/facemesh_occluded_subsets_v3")
VIDEO_ROOT = Path("/data/shared/DMD")

OUT_ROOT = Path("/data/shared/DMD_landmarks/facemesh_masked_videos_v3")

TARGET_GROUPS = ["gA", "gB", "gC"]
NPZ_PATTERN = "*_ir_face_facemesh.npz"

VARIANTS = [
    "sunglasses_both_100",
    "sunglasses_left_100",
    "sunglasses_right_100",
    "lower_face_without_nose_100",
    "left_face_half_100",
    "right_face_half_100",
    #"upper_face_half_100",
]

# 전체 처리하려면 None
MAX_NPZ_PER_VARIANT = None

SEED = 42

# 원본/마스킹 비교 영상을 좌우로 붙여 저장할지
SAVE_SIDE_BY_SIDE = False

# 선글라스 계열은 hull 대신 bbox로 처리
USE_SUNGLASSES_BBOX = True

# 마스크 경계선 디버그 표시
DRAW_MASK_BORDER = False

# 검정 마스크 padding
HULL_PADDING_PX = 8

SUNGLASSES_PAD_X = 22
SUNGLASSES_PAD_Y = 16
SUNGLASSES_Y_KEEP_PERCENTILE = 90.0

# 영상 길이 제한. 전체 저장하려면 None
MAX_FRAMES_PER_VIDEO: Optional[int] = None

# 좌표가 0~1이면 영상 크기로 스케일
AUTO_SCALE_NORMALIZED_COORDS = True


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
            if arr.ndim >= 3:
                return k

    for k in npz.files:
        arr = npz[k]
        if arr.ndim >= 3:
            return k

    raise ValueError(f"No landmark-like array found. keys={npz.files}")


def to_tvc(arr: np.ndarray) -> Tuple[np.ndarray, str]:
    arr = np.asarray(arr)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D landmark array, got shape={arr.shape}")

    _, a, b = arr.shape

    if a >= 100 and b <= 16:
        return arr.copy(), "TVC"

    if b >= 100 and a <= 16:
        return np.transpose(arr, (0, 2, 1)).copy(), "TCV"

    raise ValueError(f"Ambiguous landmark shape: {arr.shape}")


def load_landmarks(src_npz: Path, occ_npz: Path) -> Tuple[np.ndarray, np.ndarray]:
    src_data = np.load(src_npz, allow_pickle=True)
    occ_data = np.load(occ_npz, allow_pickle=True)

    src_key = find_landmark_key(src_data)
    occ_key = find_landmark_key(occ_data)

    src_lm, _ = to_tvc(src_data[src_key])
    occ_lm, _ = to_tvc(occ_data[occ_key])

    if src_lm.shape != occ_lm.shape:
        raise ValueError(f"Shape mismatch: src={src_lm.shape}, occ={occ_lm.shape}")

    return src_lm, occ_lm


def is_zero_like(points: np.ndarray) -> np.ndarray:
    c = points.shape[-1]
    coord_ch = min(3, c)

    return np.all(
        np.isclose(points[:, :coord_ch], 0.0, atol=1e-8),
        axis=1,
    )


def valid_original_points(points: np.ndarray) -> np.ndarray:
    c = points.shape[-1]
    coord_ch = min(3, c)

    return ~np.all(
        np.isclose(points[:, :coord_ch], 0.0, atol=1e-8),
        axis=1,
    )


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


def sample_files(files: List[Path], n: Optional[int], seed: int) -> List[Path]:
    if n is None or len(files) <= n:
        return files

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(files), size=n, replace=False)
    return [files[int(i)] for i in sorted(idx)]


def find_matching_video(src_npz: Path) -> Optional[Path]:
    """
    facemesh npz 경로에서 원본 영상 경로를 추정.

    예:
      /data/shared/DMD_landmarks/facemesh/distraction/dmd/gB/7/s1/xxx_ir_face_facemesh.npz

    후보:
      /data/shared/DMD/distraction/dmd/gB/7/s1/xxx_ir_face.mp4
      /data/shared/DMD/distraction/dmd/gB/7/s1/xxx_ir_body.mp4
      /data/shared/DMD/distraction/dmd/gB/7/s1/xxx_rgb_body.mp4
    """
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

    if name.endswith("_face_facemesh.npz"):
        base = name.replace("_face_facemesh.npz", "")
        candidates.extend([
            video_dir / f"{base}_face.mp4",
            video_dir / f"{base}_ir_face.mp4",
            video_dir / f"{base}_ir_body.mp4",
            video_dir / f"{base}_rgb_body.mp4",
        ])

    # fallback
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


# ============================================================
# Mask geometry
# ============================================================

def landmark_xy_to_frame_xy(
    xy: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> np.ndarray:
    pts = xy.copy().astype(np.float32)

    if pts.size == 0:
        return pts.astype(np.int32)

    finite = np.isfinite(pts).all(axis=1)
    if not finite.any():
        return np.zeros((0, 2), dtype=np.int32)

    vals = pts[finite]
    max_x = np.nanmax(vals[:, 0])
    max_y = np.nanmax(vals[:, 1])

    if AUTO_SCALE_NORMALIZED_COORDS and max_x <= 2.0 and max_y <= 2.0:
        pts[:, 0] *= frame_w
        pts[:, 1] *= frame_h

    pts[:, 0] = np.clip(pts[:, 0], 0, frame_w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, frame_h - 1)

    return pts.astype(np.int32)


def dilate_points_for_hull(points: np.ndarray, padding: int) -> np.ndarray:
    if len(points) == 0 or padding <= 0:
        return points

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
    ], dtype=np.int32)

    expanded = []
    for p in points:
        expanded.append(p[None, :] + offsets)

    return np.vstack(expanded)


def make_sunglasses_bbox_poly(
    pts: np.ndarray,
    frame_w: int,
    frame_h: int,
    pad_x: int = SUNGLASSES_PAD_X,
    pad_y: int = SUNGLASSES_PAD_Y,
    keep_percentile: float = SUNGLASSES_Y_KEEP_PERCENTILE,
) -> np.ndarray:
    """
    선글라스 계열 전용.
    아래로 튄 점을 percentile로 제거하고 bbox를 만들어 검정 마스크 처리.
    """
    if len(pts) < 3:
        return np.zeros((0, 2), dtype=np.int32)

    y = pts[:, 1]
    y_cut = np.percentile(y, keep_percentile)
    filtered = pts[y <= y_cut]

    if len(filtered) < 3:
        filtered = pts

    x1 = int(filtered[:, 0].min()) - pad_x
    y1 = int(filtered[:, 1].min()) - pad_y
    x2 = int(filtered[:, 0].max()) + pad_x
    y2 = int(filtered[:, 1].max()) + pad_y

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_w - 1, x2)
    y2 = min(frame_h - 1, y2)

    return np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.int32,
    )


def apply_black_mask_to_frame(
    frame: np.ndarray,
    orig_frame_lm: np.ndarray,
    occ_frame_lm: np.ndarray,
    variant: str,
) -> Tuple[np.ndarray, int]:
    out = frame.copy()
    h, w = out.shape[:2]

    orig_valid = valid_original_points(orig_frame_lm)
    occ_zero = is_zero_like(occ_frame_lm)

    # 원본에는 있었는데 occluded에서 0이 된 landmark
    missing_mask = orig_valid & occ_zero
    missing_count = int(missing_mask.sum())

    if missing_count == 0:
        return out, 0

    missing_xy = orig_frame_lm[:, :2][missing_mask]
    missing_pts = landmark_xy_to_frame_xy(missing_xy, frame_w=w, frame_h=h)

    if len(missing_pts) < 3:
        return out, missing_count

    if variant.startswith("sunglasses") and USE_SUNGLASSES_BBOX:
        poly = make_sunglasses_bbox_poly(
            pts=missing_pts,
            frame_w=w,
            frame_h=h,
        )

        if len(poly) >= 3:
            cv2.fillConvexPoly(out, poly.reshape(-1, 1, 2), (0, 0, 0))

            if DRAW_MASK_BORDER:
                cv2.polylines(
                    out,
                    [poly.reshape(-1, 1, 2)],
                    isClosed=True,
                    color=(0, 0, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

    else:
        hull_pts = dilate_points_for_hull(missing_pts, HULL_PADDING_PX)
        hull_pts[:, 0] = np.clip(hull_pts[:, 0], 0, w - 1)
        hull_pts[:, 1] = np.clip(hull_pts[:, 1], 0, h - 1)

        hull = cv2.convexHull(hull_pts.reshape(-1, 1, 2))
        cv2.fillConvexPoly(out, hull, (0, 0, 0))

        if DRAW_MASK_BORDER:
            cv2.polylines(
                out,
                [hull],
                isClosed=True,
                color=(0, 0, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

    return out, missing_count


# ============================================================
# Video writing
# ============================================================

def safe_video_writer(
    save_path: Path,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {save_path}")

    return writer


def process_one_video(
    src_npz: Path,
    occ_npz: Path,
    video_path: Path,
    variant: str,
) -> Path:
    src_lm, occ_lm = load_landmarks(src_npz, occ_npz)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0

    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lm_frame_count = src_lm.shape[0]

    total_frames = min(video_frame_count, lm_frame_count)

    if MAX_FRAMES_PER_VIDEO is not None:
        total_frames = min(total_frames, MAX_FRAMES_PER_VIDEO)

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Cannot read first frame: {video_path}")

    h, w = first_frame.shape[:2]

    if SAVE_SIDE_BY_SIDE:
        out_w = w * 2
        out_h = h
    else:
        out_w = w
        out_h = h

    rel = src_npz.relative_to(SRC_ROOT)
    save_dir = OUT_ROOT / variant / rel.parent
    save_name = src_npz.stem.replace("_ir_face_facemesh", "") + f"_{variant}_masked.mp4"
    save_path = save_dir / save_name

    writer = safe_video_writer(save_path, fps=fps, width=out_w, height=out_h)

    # rewind
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for fi in tqdm(range(total_frames), desc=f"{variant}:{src_npz.stem}", leave=False):
        ok, frame = cap.read()
        if not ok:
            break

        masked, missing_count = apply_black_mask_to_frame(
            frame=frame,
            orig_frame_lm=src_lm[fi],
            occ_frame_lm=occ_lm[fi],
            variant=variant,
        )

        if SAVE_SIDE_BY_SIDE:
            vis = np.concatenate([frame, masked], axis=1)

            cv2.putText(
                vis,
                "original",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                "masked",
                (w + 20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            vis = masked

        writer.write(vis)

    writer.release()
    cap.release()

    return save_path


# ============================================================
# Main
# ============================================================

def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_src_files = collect_original_npz_files()

    print(f"[INFO] SRC_ROOT   = {SRC_ROOT}")
    print(f"[INFO] OCC_ROOT   = {OCC_ROOT}")
    print(f"[INFO] VIDEO_ROOT = {VIDEO_ROOT}")
    print(f"[INFO] OUT_ROOT   = {OUT_ROOT}")
    print(f"[INFO] source npz = {len(all_src_files)}")

    if len(all_src_files) == 0:
        raise RuntimeError("No source npz files found.")

    # ------------------------------------------------------------
    # 1. 원본 영상이 실제로 존재하는 npz만 먼저 필터링
    # ------------------------------------------------------------
    valid_src_files = []
    missing_video_files = []

    for src_npz in tqdm(all_src_files, desc="check matching videos"):
        video_path = find_matching_video(src_npz)
        if video_path is None:
            missing_video_files.append(src_npz)
        else:
            valid_src_files.append(src_npz)

    print(f"[INFO] valid src files with video: {len(valid_src_files)}")
    print(f"[INFO] missing video files       : {len(missing_video_files)}")

    if len(valid_src_files) == 0:
        raise RuntimeError("No source npz files have matching videos.")

    # ------------------------------------------------------------
    # 2. 모든 variant가 공유할 공통 샘플을 한 번만 뽑음
    # ------------------------------------------------------------
    common_src_files = sample_files(
        valid_src_files,
        n=MAX_NPZ_PER_VARIANT,
        seed=SEED,
    )

    print("[INFO] common sampled files:")
    for p in common_src_files:
        print(f"  {p.relative_to(SRC_ROOT)}")

    total_ok = 0
    total_missing_occ = 0
    total_missing_video = 0
    total_failed = 0

    # ------------------------------------------------------------
    # 3. 모든 variant가 같은 common_src_files를 사용
    # ------------------------------------------------------------
    for variant in VARIANTS:
        print(f"\n[INFO] variant: {variant}")

        ok = 0
        missing_occ = 0
        missing_video = 0
        failed = 0

        for src_npz in tqdm(common_src_files, desc=variant):
            rel = src_npz.relative_to(SRC_ROOT)
            occ_npz = OCC_ROOT / variant / rel

            if not occ_npz.exists():
                missing_occ += 1
                print(f"[WARN] missing occ npz: {occ_npz}")
                continue

            video_path = find_matching_video(src_npz)
            if video_path is None:
                missing_video += 1
                print(f"[WARN] missing video for: {src_npz}")
                continue

            try:
                save_path = process_one_video(
                    src_npz=src_npz,
                    occ_npz=occ_npz,
                    video_path=video_path,
                    variant=variant,
                )
                ok += 1
                print(f"[SAVED] {save_path}")

            except Exception as e:
                failed += 1
                print(f"[ERROR] {src_npz}: {type(e).__name__}: {e}")

        total_ok += ok
        total_missing_occ += missing_occ
        total_missing_video += missing_video
        total_failed += failed

        print(
            f"[DONE] {variant}: "
            f"ok={ok}, missing_occ={missing_occ}, "
            f"missing_video={missing_video}, failed={failed}"
        )

    print("\n[ALL DONE]")
    print(f"Saved to            : {OUT_ROOT}")
    print(f"total ok            : {total_ok}")
    print(f"total missing occ   : {total_missing_occ}")
    print(f"total missing video : {total_missing_video}")
    print(f"total failed        : {total_failed}")


if __name__ == "__main__":
    main()