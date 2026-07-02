"""Gaze occlusion 학습용 7-region 그룹 정의.

Classification_model_Only_Pose/constants/face_regions.py 의 10-region anatomical
정의를 재조합해서 gaze 친화적인 7 region 으로 압축:

    1. left_eye   = LEFT_EYE  + LEFT_BROW       (왼쪽 눈 + 눈썹, gaze 핵심)
    2. right_eye  = RIGHT_EYE + RIGHT_BROW
    3. nose       = NOSE                         (head pose 보조 신호)
    4. mouth      = MOUTH_OUTER + MOUTH_INNER
    5. contour    = LEFT_CHEEK_JAW + RIGHT_CHEEK_JAW  (외곽/볼·턱)
    6. upper_face = FACE_OVAL 상반 (이마/관자놀이) — y 좌표 평균 기준 상위 절반
    7. lower_face = FACE_OVAL 하반 (턱)

upper/lower_face 는 다른 region 과 겹치는 landmark 가 없도록 FACE_OVAL 자체만
y-coordinate 로 분할. mediapipe canonical face mesh 의 평균 좌표 (refined) 사용.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

# 10-region anatomical 정의(face_regions.py)를 같은 폴더에서 로드.
# (모듈명 충돌 방지 위해 importlib 로 직접 로드)
import importlib.util as _ilu
_fr_path = Path(__file__).resolve().parent / "face_regions.py"
_spec = _ilu.spec_from_file_location("_occ_subset_face_regions", _fr_path)
_mod = _ilu.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
LEFT_EYE = _mod.LEFT_EYE
RIGHT_EYE = _mod.RIGHT_EYE
LEFT_BROW = _mod.LEFT_BROW
RIGHT_BROW = _mod.RIGHT_BROW
NOSE = _mod.NOSE
MOUTH_OUTER = _mod.MOUTH_OUTER
MOUTH_INNER = _mod.MOUTH_INNER
FACE_OVAL = _mod.FACE_OVAL
LEFT_CHEEK_JAW = _mod.LEFT_CHEEK_JAW
RIGHT_CHEEK_JAW = _mod.RIGHT_CHEEK_JAW


NUM_LANDMARKS = 478

# FACE_OVAL 36 점을 mediapipe canonical face mesh 의 평균 y 좌표 (대략 0.45) 기준 분할.
# 정확한 분할은 cache 데이터의 mean coord 로 데이터 주도로 잡지만, 그 전 fallback 으로
# 다음 hardcoded 분할도 제공 (mediapipe 표준 인덱스 anatomy 참고).
_OVAL_UPPER_HARDCODED = sorted([
    # 이마/관자놀이 (상부 半)
    10, 338, 297, 332, 284, 251, 389, 356, 454,
    127, 234, 21, 54, 103, 67, 109, 162,
])
_OVAL_LOWER_HARDCODED = sorted(set(FACE_OVAL) - set(_OVAL_UPPER_HARDCODED))


FACE_REGIONS_7: Dict[str, List[int]] = {
    "left_eye":   sorted(set(LEFT_EYE + LEFT_BROW)),
    "right_eye":  sorted(set(RIGHT_EYE + RIGHT_BROW)),
    "nose":       sorted(set(NOSE)),
    "mouth":      sorted(set(MOUTH_OUTER + MOUTH_INNER)),
    "contour":    sorted(set(LEFT_CHEEK_JAW + RIGHT_CHEEK_JAW)),
    "upper_face": _OVAL_UPPER_HARDCODED,
    "lower_face": _OVAL_LOWER_HARDCODED,
}
REGION_NAMES: List[str] = list(FACE_REGIONS_7.keys())
NUM_REGIONS = len(REGION_NAMES)
assert NUM_REGIONS == 7

# Gaze 용 region biased prior (양 눈에 중점, contour/upper/lower 는 보조)
REGION_PRIOR = {
    "left_eye":  0.30,
    "right_eye": 0.30,
    "nose":      0.15,
    "mouth":     0.05,
    "contour":   0.05,
    "upper_face": 0.08,
    "lower_face": 0.07,
}
REGION_PRIOR_VEC = torch.tensor([REGION_PRIOR[n] for n in REGION_NAMES], dtype=torch.float32)


# 선택적: canonical mean 좌표 캐시(있을 때만 patch 관련 함수에서 사용).
_CANONICAL_PATH = Path(__file__).resolve().parent / "canonical_face_mean.npz"


def load_canonical_mean() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """canonical_face_mean.npz 로드.

    Returns:
        mean_norm : (478, 2) float32 in [0,1] — predictor coord → normalize
        lo        : (2,)     — predictor coord 의 min (affine bias)
        hi        : (2,)     — predictor coord 의 max
    """
    if not _CANONICAL_PATH.exists():
        raise FileNotFoundError(
            f"{_CANONICAL_PATH} 없음 — cache 만든 뒤 한 번 compute 필요"
        )
    d = np.load(_CANONICAL_PATH)
    return d["mean_coord_norm"], d["affine_lo"], d["affine_hi"]


def canonical_patch_idx(grid_size: int = 7) -> torch.Tensor:
    """canonical mean position 의 landmark 별 fixed patch index (478,) long."""
    mean_norm, _, _ = load_canonical_mean()
    coords = torch.from_numpy(mean_norm)         # (478, 2)
    px = (coords[:, 0] * grid_size).floor().clamp(0, grid_size - 1).long()
    py = (coords[:, 1] * grid_size).floor().clamp(0, grid_size - 1).long()
    return py * grid_size + px                    # (478,) ∈ [0, P)


def normalize_predictor_coord(coords: torch.Tensor) -> torch.Tensor:
    """predictor 출력 coord (..., 478, 2) 를 cache 전체 affine 으로 [0,1] 정규화.

    학습 시 frame 별 coord 가 약간 다른 head pose 정보를 갖고 있다면 사용 가능.
    Phase 1 에서는 fixed canonical patch_idx 가 우선이라 이 함수는 옵션.
    """
    _, lo, hi = load_canonical_mean()
    lo_t = torch.as_tensor(lo, dtype=coords.dtype, device=coords.device)
    hi_t = torch.as_tensor(hi, dtype=coords.dtype, device=coords.device)
    return (coords - lo_t) / (hi_t - lo_t + 1e-6)


def build_landmark_to_region(num_landmarks: int = NUM_LANDMARKS) -> torch.Tensor:
    """(num_landmarks, K) float — landmark i 가 region k 에 속하면 1.

    중복 허용 (한 landmark 가 여러 region 에 속할 수 있음 — eg. eye landmark 가
    LEFT_EYE 와 LEFT_BROW 둘 다는 분리 시 한 번만 카운트). 여기서는 1-hot+다중.
    """
    M = torch.zeros(num_landmarks, NUM_REGIONS, dtype=torch.float32)
    for k, name in enumerate(REGION_NAMES):
        for i in FACE_REGIONS_7[name]:
            M[i, k] = 1.0
    return M


def landmarks_to_patch_indices(
    coords: torch.Tensor, grid_size: int = 7
) -> torch.Tensor:
    """(B, V, 2) in [0,1] → (B, V) long patch index ∈ [0, grid_size²)."""
    px = (coords[..., 0] * grid_size).floor().clamp(0, grid_size - 1).long()
    py = (coords[..., 1] * grid_size).floor().clamp(0, grid_size - 1).long()
    return py * grid_size + px


def compute_patch_visibility_gt(
    coords: torch.Tensor,
    visibility: torch.Tensor,
    grid_size: int = 7,
):
    """coords (B, V, 2), visibility (B, V) → patch_vis (B, P), patch_mask (B, P) bool.

    P = grid_size². patch 안의 landmark vis 평균. landmark 없는 patch 는 mask=False.
    """
    B, V = visibility.shape
    P = grid_size * grid_size
    idx = landmarks_to_patch_indices(coords, grid_size)            # (B, V) ∈ [0,P)
    sum_v = torch.zeros(B, P, device=visibility.device, dtype=visibility.dtype)
    cnt   = torch.zeros(B, P, device=visibility.device, dtype=visibility.dtype)
    sum_v.scatter_add_(1, idx, visibility)
    cnt.scatter_add_(1, idx, torch.ones_like(visibility))
    mask = cnt > 0
    patch_vis = torch.zeros_like(sum_v)
    patch_vis[mask] = sum_v[mask] / cnt[mask]
    return patch_vis, mask


def compute_patch_visibility_gt_canonical(
    visibility: torch.Tensor,
    grid_size: int = 7,
    canonical_idx: torch.Tensor | None = None,
):
    """canonical fixed patch_idx 사용 버전. coords 가 없어도 됨.

    visibility (..., 478) → patch_vis (..., 49), patch_mask (..., 49) bool.
    모든 frame 동일한 landmark→patch 매핑. predictor coord 좌표 정밀도와 무관.
    """
    if canonical_idx is None:
        canonical_idx = canonical_patch_idx(grid_size)              # (478,)
    canonical_idx = canonical_idx.to(visibility.device)
    P = grid_size * grid_size

    # (..., V) → leading dims merged
    shape_in = visibility.shape
    V = shape_in[-1]
    flat = visibility.reshape(-1, V)                                # (N, V)
    N = flat.shape[0]

    idx = canonical_idx.unsqueeze(0).expand(N, V)                   # (N, V)
    sum_v = torch.zeros(N, P, device=flat.device, dtype=flat.dtype)
    cnt   = torch.zeros(N, P, device=flat.device, dtype=flat.dtype)
    sum_v.scatter_add_(1, idx, flat)
    cnt.scatter_add_(1, idx, torch.ones_like(flat))
    mask = cnt > 0
    patch_vis = torch.zeros_like(sum_v)
    patch_vis[mask] = sum_v[mask] / cnt[mask]

    out_shape = shape_in[:-1] + (P,)
    return patch_vis.reshape(out_shape), mask.reshape(out_shape)


def build_region_membership_canonical(grid_size: int = 7) -> torch.Tensor:
    """canonical mean position 기반 fixed region membership (K, P).

    매 frame 동일. region 별 landmark 들이 어느 patch 들에 분포되는지.
    """
    cidx = canonical_patch_idx(grid_size)                           # (478,)
    P = grid_size * grid_size
    membership = torch.zeros(NUM_REGIONS, P, dtype=torch.float32)
    for k, name in enumerate(REGION_NAMES):
        for i in FACE_REGIONS_7[name]:
            membership[k, int(cidx[i])] += 1.0
    membership = membership / membership.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return membership


def compute_region_membership_from_coords(
    coords: torch.Tensor,
    grid_size: int = 7,
) -> torch.Tensor:
    """coords (V, 2) — 보통 학습 데이터의 평균 landmark 위치.
    Returns: (K, P) — region k 의 landmark 들이 patch p 에 어떻게 분포되는지.
    각 region 의 합 = 1 (normalized).
    """
    V = coords.shape[0]
    P = grid_size * grid_size
    K = NUM_REGIONS
    membership = torch.zeros(K, P, dtype=torch.float32)
    px = (coords[:, 0] * grid_size).floor().clamp(0, grid_size - 1).long()
    py = (coords[:, 1] * grid_size).floor().clamp(0, grid_size - 1).long()
    patch_idx = py * grid_size + px                                # (V,)
    for k, name in enumerate(REGION_NAMES):
        for i in FACE_REGIONS_7[name]:
            membership[k, int(patch_idx[i])] += 1.0
    membership = membership / membership.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return membership
