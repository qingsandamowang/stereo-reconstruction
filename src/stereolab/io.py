"""ETH3D/Middlebury-compatible data decoding and calibration parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Calibration:
    focal_px: float
    cx_px: float
    cy_px: float
    doffs_px: float
    baseline_mm: float
    width: int
    height: int
    ndisp: int
    focal_y_px: float | None = None

    def __post_init__(self):
        numeric = (self.focal_px, self.fy_px, self.cx_px, self.cy_px, self.doffs_px, self.baseline_mm)
        if not np.isfinite(numeric).all() or self.focal_px <= 0 or self.fy_px <= 0 or self.baseline_mm <= 0:
            raise ValueError("Calibration requires finite values, positive focal lengths and positive baseline in mm")
        if any(not isinstance(value, (int, np.integer)) or value <= 0 for value in (self.width, self.height, self.ndisp)):
            raise ValueError("Calibration width, height and ndisp must be positive integers")

    @property
    def fy_px(self) -> float:
        return self.focal_px if self.focal_y_px is None else self.focal_y_px


def read_pfm(path: Path) -> np.ndarray:
    """Read a single-channel Portable Float Map with bottom-up pixel rows."""
    with path.open("rb") as stream:
        if stream.readline().strip() != b"Pf":
            raise ValueError(f"Expected grayscale PFM: {path}")
        dimensions = stream.readline().split()
        if len(dimensions) != 2:
            raise ValueError(f"Malformed PFM dimensions: {path}")
        width, height = map(int, dimensions)
        scale = float(stream.readline().strip())
        dtype = "<f4" if scale < 0 else ">f4"
        values = np.frombuffer(stream.read(), dtype=dtype)
        if values.size != width * height:
            raise ValueError(f"Malformed PFM pixel count: {path}")
        return np.flipud(values.reshape(height, width)).astype(np.float32, copy=True)


def parse_calibration(path: Path) -> Calibration:
    """Read the Middlebury v3 calib.txt used by ETH3D two-view scenes."""
    contents = path.read_text(encoding="utf-8")
    fields = dict(re.findall(r"(?m)^\s*([A-Za-z0-9]+)\s*=\s*(.+)$", contents))
    required = {"cam0", "doffs", "baseline", "width", "height", "ndisp"}
    if not required.issubset(fields):
        raise ValueError(f"Missing calibration fields: {sorted(required - fields.keys())}")
    def camera(key):
        try:
            matrix = np.array([[float(value) for value in row.split()] for row in fields[key].strip().strip("[]").split(";")])
        except ValueError as error:
            raise ValueError(f"Invalid camera matrix: {path}") from error
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            raise ValueError(f"Invalid camera matrix: {path}")
        if not np.allclose(matrix[2], [0, 0, 1]) or not np.allclose([matrix[0, 1], matrix[1, 0]], [0, 0]):
            raise ValueError("Camera matrices must describe a rectified pinhole pair without skew")
        return matrix

    cam0 = camera("cam0")
    if "cam1" in fields:
        cam1 = camera("cam1")
        if not np.allclose(cam0[[0, 1], [0, 1]], cam1[[0, 1], [0, 1]], rtol=1e-5) or not np.isclose(cam0[1, 2], cam1[1, 2], atol=1e-3):
            raise ValueError("Camera matrices are inconsistent with a horizontally rectified pair")
        if not np.isclose(cam1[0, 2] - cam0[0, 2], float(fields["doffs"]), atol=0.02, rtol=1e-5):
            raise ValueError("Calibration doffs differs from rectified camera principal-point offset")
    return Calibration(
        focal_px=float(cam0[0, 0]),
        cx_px=float(cam0[0, 2]),
        cy_px=float(cam0[1, 2]),
        doffs_px=float(fields["doffs"]),
        baseline_mm=float(fields["baseline"]),
        width=int(fields["width"]),
        height=int(fields["height"]),
        ndisp=int(fields["ndisp"]),
        focal_y_px=float(cam0[1, 1]),
    )


def read_image(path: Path, color: bool = False) -> np.ndarray:
    """Decode an image even when the workspace path contains Chinese text."""
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(data, flag)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if color else image


def non_occluded_pixels(encoded_mask: np.ndarray) -> np.ndarray:
    """ETH3D mask: 255=visible in both views; 128=visible only in left."""
    if encoded_mask.ndim != 2:
        raise ValueError("Expected a single-channel ETH3D occlusion mask")
    return encoded_mask == 255
