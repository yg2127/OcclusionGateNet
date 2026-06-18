"""
FaceMesh 478 landmarks -> semantic face region nodes.

Input:
    x_face: (N, C, T, V=478)

Output:
    x_region: (N, C, T, R)

Supported schemes:
    dms_7:
        left_eye, right_eye, nose, mouth, face_oval, left_face, right_face

    dms_10:
        left_eye, right_eye, left_brow, right_brow, nose,
        mouth_outer, mouth_inner, face_oval, left_cheek_jaw, right_cheek_jaw
"""

from __future__ import annotations

import torch
import torch.nn as nn


DMS_7_REGIONS: dict[str, list[int]] = {
    "left_eye": [
        33, 7, 163, 144, 145, 153, 154, 155,
        133, 173, 157, 158, 159, 160, 161, 246,
    ],
    "right_eye": [
        362, 382, 381, 380, 374, 373, 390, 249,
        263, 466, 388, 387, 386, 385, 384, 398,
    ],
    "nose": [
        1, 2, 4, 5, 6, 19, 45, 94, 97, 98,
        115, 168, 195, 197, 220, 275, 326, 327, 344,
    ],
    "mouth": [
        61, 146, 91, 181, 84, 17, 314, 405,
        321, 375, 291, 185, 40, 39, 37, 0,
        267, 269, 270, 409, 78, 95, 88, 178,
        87, 14, 317, 402, 318, 324, 308,
    ],
    "face_oval": [
        10, 338, 297, 332, 284, 251, 389, 356,
        454, 323, 361, 288, 397, 365, 379, 378,
        400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21,
        54, 103, 67, 109,
    ],
    "left_face": [
        127, 234, 93, 132, 58, 172, 136, 150,
        149, 176, 148, 152, 21, 54, 103, 67,
        109, 10, 162,
    ],
    "right_face": [
        356, 454, 323, 361, 288, 397, 365, 379,
        378, 400, 377, 152, 389, 251, 284, 332,
        297, 338, 10,
    ],
}


DMS_10_REGIONS: dict[str, list[int]] = {
    "left_eye": sorted(set([
        246, 161, 160, 159, 158, 157, 173,
        33, 7, 163, 144, 145, 153, 154, 155, 133,
        468, 469, 470, 471, 472,
    ])),
    "right_eye": sorted(set([
        263, 249, 390, 373, 374, 380, 381, 382, 362,
        466, 388, 387, 386, 385, 384, 398,
        473, 474, 475, 476, 477,
    ])),
    "left_brow": sorted(set([
        70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
    ])),
    "right_brow": sorted(set([
        300, 293, 334, 296, 336, 285, 295, 282, 283, 276,
    ])),
    "nose": sorted(set([
        168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 141, 370,
        98, 97, 326, 327,
        45, 51, 115, 220, 219, 218, 237,
        275, 281, 344, 440, 439, 438, 457,
        129, 358, 102, 331,
    ])),
    "mouth_outer": sorted(set([
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
    ])),
    "mouth_inner": sorted(set([
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
        308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
    ])),
    "face_oval": sorted(set([
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
        361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
        176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
        162, 21, 54, 103, 67, 109,
    ])),
    "left_cheek_jaw": sorted(set([
        117, 118, 119, 120, 121, 128, 126, 142, 36, 205,
        50, 123, 147, 213, 192, 214, 207, 206, 216, 212,
    ])),
    "right_cheek_jaw": sorted(set([
        346, 347, 348, 349, 350, 357, 355, 371, 266, 425,
        280, 352, 376, 433, 416, 434, 427, 426, 436, 432,
    ])),
}


REGION_SCHEMES: dict[str, dict[str, list[int]]] = {
    "dms_7": DMS_7_REGIONS,
    "dms_10": DMS_10_REGIONS,
}


class FaceRegionPool(nn.Module):
    """
    Pool full FaceMesh landmarks into semantic region nodes.

    Args:
        scheme:
            "dms_7" or "dms_10".
        reduce:
            "mean" or "max".

    Input:
        x: (N, C, T, V)

    Return:
        out: (N, C, T, R)
    """

    def __init__(self, scheme: str = "dms_10", reduce: str = "mean"):
        super().__init__()

        if scheme not in REGION_SCHEMES:
            raise ValueError(
                f"Unknown face region scheme: {scheme}. "
                f"Available: {list(REGION_SCHEMES.keys())}"
            )

        if reduce not in ("mean", "max"):
            raise ValueError(f"Unknown reduce: {reduce}")

        self.scheme = scheme
        self.reduce = reduce
        self.region_names = list(REGION_SCHEMES[scheme].keys())
        self.regions = REGION_SCHEMES[scheme]

        self._validate_regions()

    @property
    def num_regions(self) -> int:
        return len(self.region_names)

    def _validate_regions(self) -> None:
        for name, idxs in self.regions.items():
            if len(idxs) == 0:
                raise ValueError(f"region {name}: empty")
            for i in idxs:
                if not (0 <= i < 478):
                    raise ValueError(f"region {name}: index {i} out of [0, 478)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, C, T, V=478)
        """
        if x.ndim != 4:
            raise ValueError(f"Expected x_face shape (N, C, T, V), got {tuple(x.shape)}")

        n, c, t, v = x.shape

        max_idx = max(max(idxs) for idxs in self.regions.values())
        if v <= max_idx:
            raise ValueError(
                f"FaceRegionPool requires V > {max_idx}, but got V={v}. "
                "This module expects full FaceMesh landmarks."
            )

        outs = []

        for name in self.region_names:
            idxs = self.regions[name]
            xr = x[:, :, :, idxs]  # (N, C, T, n_region)

            if self.reduce == "mean":
                xr = xr.mean(dim=-1)  # (N, C, T)
            elif self.reduce == "max":
                xr = xr.max(dim=-1).values
            else:
                raise RuntimeError(f"Unsupported reduce: {self.reduce}")

            outs.append(xr)

        out = torch.stack(outs, dim=-1)  # (N, C, T, R)
        return out