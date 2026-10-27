"""Generate a seeded translated-texture stereo demo; no download or dataset required."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .reconstruct import reconstruct


def run_demo(out: Path) -> dict:
    """A planar, Lambertian synthetic sanity check, not evidence of real-scene accuracy."""
    inputs = out / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    height, width, shift = 120, 256, 12
    left = np.random.default_rng(20261027).integers(0, 256, (height, width), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:, :-shift] = left[:, shift:]
    visible = np.full_like(left, 255)
    visible[:, :shift] = 0
    for name, array in (("left.png", left), ("right.png", right), ("visible.png", visible)):
        (inputs / name).write_bytes(cv2.imencode(".png", array)[1].tobytes())
    (inputs / "calib.txt").write_text("cam0=[180 0 128; 0 180 60; 0 0 1]\n"
                                      "cam1=[180 0 128; 0 180 60; 0 0 1]\n"
                                      "doffs=0\nbaseline=60\nwidth=256\nheight=120\nndisp=64\n", encoding="utf-8")
    truth = np.full((height, width), shift, dtype="<f4")
    (inputs / "gt.pfm").write_bytes(f"Pf\n{width} {height}\n-1.0\n".encode("ascii") + np.flipud(truth).tobytes())
    report = reconstruct(inputs / "left.png", inputs / "right.png", inputs / "calib.txt", out,
                         method="sgbm_wls", ground_truth=inputs / "gt.pfm", mask=inputs / "visible.png")
    report["synthetic_demo"] = {"seed": 20261027, "shift_px": shift, "expected_depth_mm": 900,
                                 "scope": "one textured fronto-parallel plane; exact horizontal translation; not natural-scene evidence"}
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/synthetic"))
    args = parser.parse_args()
    cv2.setNumThreads(1)
    report = run_demo(args.out)
    print(f"Synthetic translation: 12 px; expected depth: 900 mm; measured median: {report['depth_median_mm']:.2f} mm")
    print(f"Open offline viewer: {(args.out / 'viewer.html').resolve()}")


if __name__ == "__main__":
    main()
