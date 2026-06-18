"""Utilities for mapping window-level occlusion vectors to face-region reliability.

The current dataset provides x_occ as a compact window-level vector.  Most
fusion experiments need a per-region reliability vector with shape (N, R).
This helper intentionally keeps the mapping configurable so new occlusion-label
formats do not require model-code rewrites.
"""
from __future__ import annotations

from typing import Sequence

import torch


def _default_region_occ_indices(num_regions: int, occ_dim: int) -> list[int]:
    """Return a conservative default region->occ mapping.

    Assumption for the common occ_dim=5 case:
        0: left_eye visibility
        1: right_eye visibility
        2: nose / mid-face visibility
        3: mouth / lower-face visibility
        4: crop_valid or global quality

    Region definitions differ by scheme, so unmapped regions use neutral
    reliability rather than being forced by an uncertain label.
    """
    if occ_dim <= 0:
        return [-1] * num_regions

    if occ_dim >= 5:
        if num_regions >= 10:
            # dms_10-style rough mapping: eyes, mid-face, mouth/lower-face,
            # and neutral for broad/uncertain regions.
            base = [0, 1, 2, 2, 3, 3, -1, -1, -1, -1]
        elif num_regions == 7:
            base = [0, 1, 2, 2, 3, 3, -1]
        elif num_regions == 5:
            base = [0, 1, 2, 3, -1]
        else:
            base = list(range(min(num_regions, 4))) + [-1] * max(0, num_regions - 4)
        return base[:num_regions] + [-1] * max(0, num_regions - len(base))

    # Generic fallback: first regions consume first occ entries, others neutral.
    return [i if i < occ_dim else -1 for i in range(num_regions)]


def resolve_region_occ_indices(
    *,
    num_regions: int,
    occ_dim: int,
    region_occ_indices: Sequence[int] | None = None,
) -> list[int]:
    if region_occ_indices is None:
        return _default_region_occ_indices(num_regions, occ_dim)

    out = [int(x) for x in region_occ_indices]
    if len(out) < num_regions:
        out = out + [-1] * (num_regions - len(out))
    return out[:num_regions]


def occ_to_region_reliability(
    x_occ: torch.Tensor | None,
    *,
    num_regions: int,
    occ_dim: int,
    region_occ_indices: Sequence[int] | None = None,
    default_visible: float = 1.0,
    mask_strength: float = 1.0,
    min_reliability: float = 0.0,
    max_reliability: float = 1.0,
) -> torch.Tensor | None:
    """Convert x_occ (N, occ_dim) into region reliability (N, R).

    region_occ_indices:
        List of length R.  Each value is an index into x_occ.  -1 means the
        region is not directly supervised by occ labels and receives
        default_visible.

    mask_strength:
        1.0 means use visibility directly. 0.0 means neutral all-ones mask.
        Intermediate values interpolate between no mask and full mask.
    """
    if x_occ is None:
        return None
    if x_occ.ndim != 2:
        raise ValueError(f"x_occ must be (N, occ_dim), got {tuple(x_occ.shape)}")

    n = x_occ.shape[0]
    device = x_occ.device
    dtype = x_occ.dtype
    indices = resolve_region_occ_indices(
        num_regions=num_regions,
        occ_dim=occ_dim,
        region_occ_indices=region_occ_indices,
    )

    rel = torch.full((n, num_regions), float(default_visible), device=device, dtype=dtype)
    for r, idx in enumerate(indices):
        if 0 <= idx < x_occ.shape[1]:
            rel[:, r] = x_occ[:, idx]

    rel = rel.clamp(min=float(min_reliability), max=float(max_reliability))

    # Interpolate toward neutral reliability=1.0. This is useful when hard
    # masking is too aggressive in early experiments.
    strength = float(mask_strength)
    rel = 1.0 + strength * (rel - 1.0)
    return rel.clamp_min(float(min_reliability))
