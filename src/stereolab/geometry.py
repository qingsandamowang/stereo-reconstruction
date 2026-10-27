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
    max_depth_mm: float | None = 20_000.0,
) -> int:
    """Export valid colored 3D points to an ASCII PLY file."""
    points, colors = point_cloud(disparity, rgb, calibration, stride, max_depth_mm)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(
            "ply\nformat ascii 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        )
        for (px, py, pz), color in zip(points, colors):
            stream.write(f"{px:.3f} {py:.3f} {pz:.3f} {int(color[0])} {int(color[1])} {int(color[2])}\n")
    return len(points)


def point_cloud(
    disparity: np.ndarray,
    rgb: np.ndarray,
    calibration: Calibration,
    stride: int = 1,
    max_depth_mm: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite positive-depth XYZ (mm) and RGB, with explicit optional clipping."""
    if disparity.ndim != 2 or rgb.ndim != 3 or disparity.shape != rgb.shape[:2] or rgb.shape[2] != 3:
        raise ValueError("Disparity and RGB image must have matching geometry")
    if disparity.shape != (calibration.height, calibration.width):
        raise ValueError("Image dimensions differ from calibration")
    if not isinstance(stride, int) or stride < 1:
        raise ValueError("Point stride must be a positive integer")
    if max_depth_mm is not None and (not np.isfinite(max_depth_mm) or max_depth_mm <= 0):
        raise ValueError("Maximum depth must be finite and positive")
    depth = depth_from_disparity(disparity, calibration)
    ys, xs = np.indices(disparity.shape)
    keep = np.isfinite(depth) & (depth > 0)
    if max_depth_mm is not None:
        keep &= depth <= max_depth_mm
    keep &= (xs % stride == 0) & (ys % stride == 0)
    z = depth[keep]
    points = np.column_stack(((xs[keep] - calibration.cx_px) * z / calibration.focal_px,
                              (ys[keep] - calibration.cy_px) * z / calibration.fy_px, z))
    finite = np.isfinite(points).all(axis=1)
    return points[finite], rgb[keep][finite]
