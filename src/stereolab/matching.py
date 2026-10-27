"""OpenCV stereo baselines and reciprocal disparity filtering."""

from __future__ import annotations

import cv2
import numpy as np


def left_right_consistency(left: np.ndarray, right: np.ndarray, tolerance: float = 1.0) -> np.ndarray:
    """Keep left disparities whose right-view match returns to the same pixel."""
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("Left and right disparities must be matching 2D arrays")
    height, width = left.shape
    x = np.broadcast_to(np.arange(width), (height, width))
    y = np.broadcast_to(np.arange(height)[:, None], (height, width))
    valid_left = np.isfinite(left) & (left > 0)
    right_x = np.rint(x - np.nan_to_num(left, nan=0.0)).astype(np.int32)
    inside = (right_x >= 0) & (right_x < width)
    sampled_right = right[y, np.clip(right_x, 0, width - 1)]
    return valid_left & inside & np.isfinite(sampled_right) & (sampled_right < 0) & (np.abs(left + sampled_right) <= tolerance)


def _matcher(method: str, num_disp: int, block_size: int, right_view: bool = False):
    minimum = -num_disp + 1 if right_view else 0
    if method == "bm":
        matcher = cv2.StereoBM_create(numDisparities=num_disp, blockSize=max(5, block_size))
        matcher.setMinDisparity(minimum)
        matcher.setTextureThreshold(5)
        matcher.setUniquenessRatio(5)
        matcher.setSpeckleWindowSize(50)
        matcher.setSpeckleRange(2)
        return matcher
    if method == "sgbm":
        return cv2.StereoSGBM_create(
            minDisparity=minimum,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * block_size * block_size,
            P2=32 * block_size * block_size,
            disp12MaxDiff=1,
            uniquenessRatio=5,
            speckleWindowSize=50,
            speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
    raise ValueError(f"Unknown stereo method: {method}")


def match_stereo(
    left: np.ndarray,
    right: np.ndarray,
    method: str = "sgbm",
    num_disp: int = 64,
    block_size: int = 5,
    consistency: bool = False,
    tolerance: float = 1.0,
) -> np.ndarray:
    """Compute left-view disparity in pixels, marking unreliable pixels as NaN."""
    if left.shape != right.shape or left.ndim != 2 or left.dtype != np.uint8 or right.dtype != np.uint8:
        raise ValueError("Expected same-shaped uint8 grayscale stereo images")
    if num_disp < 16 or num_disp % 16 or block_size < 3 or block_size % 2 == 0:
        raise ValueError("num_disp must be a multiple of 16; block_size must be odd and at least 3")
    raw = _matcher(method, num_disp, block_size).compute(left, right).astype(np.float32) / 16.0
    disparity = np.where(raw > 0, raw, np.nan)
    if consistency:
        right_raw = _matcher(method, num_disp, block_size, right_view=True).compute(right, left).astype(np.float32) / 16.0
        right_disparity = np.where(right_raw < -0.5, right_raw, np.nan)
        disparity[~left_right_consistency(disparity, right_disparity, tolerance)] = np.nan
    return disparity
