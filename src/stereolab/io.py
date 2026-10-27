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
    fields = dict(re.findall(r"(?m)^([A-Za-z0-9]+)=(.+)$", contents))
    required = {"cam0", "doffs", "baseline", "width", "height", "ndisp"}
    if not required.issubset(fields):
        raise ValueError(f"Missing calibration fields: {sorted(required - fields.keys())}")
    rows = fields["cam0"].strip("[]").split(";")
    if len(rows) != 3:
        raise ValueError(f"Invalid camera matrix: {path}")
    first = [float(value) for value in rows[0].split()]
    second = [float(value) for value in rows[1].split()]
    if len(first) != 3 or len(second) != 3:
        raise ValueError(f"Invalid camera matrix: {path}")
    return Calibration(
        focal_px=first[0],
        cx_px=first[2],
        cy_px=second[2],
        doffs_px=float(fields["doffs"]),
        baseline_mm=float(fields["baseline"]),
        width=int(fields["width"]),
        height=int(fields["height"]),
        ndisp=int(fields["ndisp"]),
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
