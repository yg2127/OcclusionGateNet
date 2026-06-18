"""MediaPipe FaceMesh (refined, 478 landmarks) → 해부학적 region group 매핑.

인덱스는 MediaPipe 공식 FACEMESH_* connection 정의에서 파생 (refined mesh 포함).
각 region 은 그룹 내 landmark 를 평균 풀링해 K=10개의 region 토큰을 만든다.
"""
from __future__ import annotations

from typing import Dict, List


# ---- MediaPipe FaceMesh 공식 landmark 그룹 ----
# 값은 중복을 허용해 connection pair를 펼친 뒤 set으로 압축해 사용.

LEFT_EYE: List[int] = sorted(set([
    # upper lid
    246, 161, 160, 159, 158, 157, 173,
    # lower lid
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    # iris + refined (if present in refined mesh)
    468, 469, 470, 471, 472,
]))

RIGHT_EYE: List[int] = sorted(set([
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    466, 388, 387, 386, 385, 384, 398,
    473, 474, 475, 476, 477,
]))

LEFT_BROW: List[int] = sorted(set([
    70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
]))

RIGHT_BROW: List[int] = sorted(set([
    300, 293, 334, 296, 336, 285, 295, 282, 283, 276,
]))

NOSE: List[int] = sorted(set([
    168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 141, 370,
    98, 97, 326, 327,
    # bridge + nostrils
    45, 51, 115, 220, 219, 218, 237,
    275, 281, 344, 440, 439, 438, 457,
    129, 358, 102, 331,
]))

MOUTH_OUTER: List[int] = sorted(set([
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
]))

MOUTH_INNER: List[int] = sorted(set([
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
]))

FACE_OVAL: List[int] = sorted(set([
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
]))

LEFT_CHEEK_JAW: List[int] = sorted(set([
    117, 118, 119, 120, 121, 128, 126, 142, 36, 205,
    50, 123, 147, 213, 192, 214, 207, 206, 216, 212,
]))

RIGHT_CHEEK_JAW: List[int] = sorted(set([
    346, 347, 348, 349, 350, 357, 355, 371, 266, 425,
    280, 352, 376, 433, 416, 434, 427, 426, 436, 432,
]))


FACE_REGIONS: Dict[str, List[int]] = {
    "left_eye":        LEFT_EYE,
    "right_eye":       RIGHT_EYE,
    "left_brow":       LEFT_BROW,
    "right_brow":      RIGHT_BROW,
    "nose":            NOSE,
    "mouth_outer":     MOUTH_OUTER,
    "mouth_inner":     MOUTH_INNER,
    "face_oval":       FACE_OVAL,
    "left_cheek_jaw":  LEFT_CHEEK_JAW,
    "right_cheek_jaw": RIGHT_CHEEK_JAW,
}


REGION_NAMES: List[str] = list(FACE_REGIONS.keys())
NUM_REGIONS: int = len(REGION_NAMES)
assert NUM_REGIONS == 10, f"expected K=10 regions, got {NUM_REGIONS}"


# ---- 검증 & 인덱스 매트릭스 ----

def _validate():
    for name, idxs in FACE_REGIONS.items():
        if len(idxs) == 0:
            raise ValueError(f"region {name}: empty")
        for i in idxs:
            if not (0 <= i < 478):
                raise ValueError(f"region {name}: index {i} out of [0, 478)")

_validate()


def region_member_matrix():
    """return numpy (K=10, 478) float32 mask. row sum = region size."""
    import numpy as np
    M = np.zeros((NUM_REGIONS, 478), dtype=np.float32)
    for ri, name in enumerate(REGION_NAMES):
        for li in FACE_REGIONS[name]:
            M[ri, li] = 1.0
    return M


def region_mean_matrix():
    """return numpy (K=10, 478) float32, mean-pool operator. row sum = 1."""
    M = region_member_matrix()
    s = M.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return M / s
