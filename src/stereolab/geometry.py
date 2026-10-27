"""Calibrated depth and point-cloud export in millimetres."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import Calibration


def depth_from_disparity(disparity: np.ndarray, calibration: Calibration) -> np.ndarray:
    """Calculate Z = baseline * focal / (disparity + principal-point offset)."""
    denominator = np.asarray(disparity, dtype=np.float32) + calibration.doffs_px
    depth = np.full(denominator.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(denominator) & (denominator > 0)
    depth[valid] = calibration.baseline_mm * calibration.focal_px / denominator[valid]
    return depth


def write_ply(
    path: Path,
    disparity: np.ndarray,
    rgb: np.ndarray,
    calibration: Calibration,
    stride: int = 1,
    max_depth_mm: float = 20_000.0,
) -> int:
    """Export valid colored 3D points to an ASCII PLY file."""
    if disparity.shape != rgb.shape[:2] or rgb.shape[2] != 3 or stride < 1:
        raise ValueError("Disparity and RGB image must have matching geometry")
    depth = depth_from_disparity(disparity, calibration)
    ys, xs = np.indices(disparity.shape)
    keep = np.isfinite(depth) & (depth > 0) & (depth <= max_depth_mm)
    if stride > 1:
        keep &= (xs % stride == 0) & (ys % stride == 0)
    x = xs[keep]
    y = ys[keep]
    z = depth[keep]
    colors = rgb[keep]
    world_x = (x - calibration.cx_px) * z / calibration.focal_px
    world_y = (y - calibration.cy_px) * z / calibration.focal_px
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(
            "ply\nformat ascii 1.0\n"
            f"element vertex {len(z)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        )
        for px, py, pz, color in zip(world_x, world_y, z, colors):
            stream.write(f"{px:.3f} {py:.3f} {pz:.3f} {int(color[0])} {int(color[1])} {int(color[2])}\n")
    return len(z)
