"""Reconstruct a rectified stereo pair into disparity, depth and an offline 3D viewer."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .geometry import depth_from_disparity, point_cloud, write_ply
from .io import non_occluded_pixels, parse_calibration, read_image, read_pfm
from .matching import match_stereo
from .metrics import depth_metrics, disparity_metrics
from .viewer import write_viewer


def _preview(path: Path, values: np.ndarray) -> None:
    valid = np.isfinite(values) & (values > 0)
    maximum = float(np.percentile(values[valid], 99)) if valid.any() else 1.0
    scaled = np.clip(np.nan_to_num(values, nan=0, posinf=0, neginf=0) / maximum, 0, 1)
    image = cv2.applyColorMap(np.uint8(scaled * 255), cv2.COLORMAP_TURBO)
    image[~valid] = 0
    path.write_bytes(cv2.imencode(".png", image)[1].tobytes())


def reconstruct(left: Path, right: Path, calibration: Path, out: Path, *, method: str = "sgbm",
                num_disp: int | None = None, block_size: int = 7, consistency: bool = False,
                wls_lambda: float = 8000.0, wls_sigma: float = 0.8, max_points: int = 20_000,
                stride: int = 1, max_depth_mm: float | None = None, ground_truth: Path | None = None,
                mask: Path | None = None) -> dict:
    """Run calibrated reconstruction. Accuracy exists only when explicit ground truth is supplied."""
    calib = parse_calibration(calibration)
    gray_left, gray_right = read_image(left), read_image(right)
    rgb = read_image(left, color=True)
    expected = (calib.height, calib.width)
    if gray_left.shape != expected or gray_right.shape != expected:
        raise ValueError("Left/right image dimensions must match each other and the calibration")
    if max_points < 1 or stride < 1:
        raise ValueError("Maximum viewer points and point stride must be positive")
    if max_depth_mm is not None and (not np.isfinite(max_depth_mm) or max_depth_mm <= 0):
        raise ValueError("Maximum depth must be finite and positive")
    if mask is not None and ground_truth is None:
        raise ValueError("A ground-truth mask requires --ground-truth")
    truth = read_pfm(ground_truth) if ground_truth is not None else None
    valid_mask = non_occluded_pixels(read_image(mask)) if mask is not None else np.ones(expected, dtype=bool)
    if (truth is not None and truth.shape != expected) or valid_mask.shape != expected:
        raise ValueError("Ground-truth/mask dimensions differ from calibration")
    # calib.ndisp describes a range, not an OpenCV multiple-of-16 setting.
    requested = max(16, ((calib.ndisp + 15) // 16) * 16)
    feasible = ((calib.width - max(5, block_size) // 2 - 1) // 16) * 16
    search = num_disp if num_disp is not None else min(requested, feasible)
    if search + max(5, block_size) // 2 >= calib.width:
        raise ValueError("Disparity search is too wide for the image; use a smaller --num-disp multiple of 16")
    start = perf_counter()
    disparity = match_stereo(gray_left, gray_right, method, search, block_size, consistency,
                             wls_lambda=wls_lambda, wls_sigma=wls_sigma)
    matching_ms = (perf_counter() - start) * 1000
    depth = depth_from_disparity(disparity, calib)
    points, colors = point_cloud(disparity, rgb, calib, stride, max_depth_mm)
    accuracy = disparity_metrics(truth, disparity, valid_mask) if truth is not None else None
    if accuracy is not None:
        accuracy.update(depth_metrics(truth, disparity, valid_mask, calib))
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "disparity.npy", disparity, allow_pickle=False)
    np.save(out / "depth_mm.npy", depth, allow_pickle=False)
    _preview(out / "disparity.png", disparity)
    _preview(out / "depth.png", depth)
    ply_points = write_ply(out / "cloud.ply", disparity, rgb, calib, stride, max_depth_mm)
    viewer_points = write_viewer(out / "viewer.html", points, colors, max_points, title="Stereo reconstruction")
    valid_depth = depth[np.isfinite(depth) & (depth > 0)]
    report = {"method": method, "parameters": {"num_disp": search, "block_size": block_size,
              "consistency": consistency, **({"wls_lambda": wls_lambda, "wls_sigma": wls_sigma} if method == "sgbm_wls" else {})},
              "calibration": asdict(calib), "units": "mm", "matching_ms": matching_ms,
              "opencv": cv2.__version__, "opencv_threads": cv2.getNumThreads(),
              "inputs_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
                                (("left", left), ("right", right), ("calibration", calibration))},
              "valid_disparity_pixels": int(np.isfinite(disparity).sum()),
              "prediction_fraction_of_image": float(np.isfinite(disparity).mean()),
              "valid_depth_pixels": len(valid_depth),
              "depth_median_mm": float(np.median(valid_depth)) if len(valid_depth) else None,
              "ply_points": ply_points, "viewer_points": viewer_points, "viewer_limit": max_points,
              "cloud_stride": stride, "cloud_max_depth_mm": max_depth_mm,
              "search_policy": "explicit --num-disp" if num_disp is not None else "calib.ndisp rounded to 16 and capped to image width; use --num-disp to change the search/coverage tradeoff",
              "preview_scale": "each PNG uses its own 99th percentile; numerical data is in the NPY files",
              "accuracy_status": "evaluated against supplied ground truth" if truth is not None else "not evaluated; no ground truth supplied"}
    if accuracy is not None:
        report["accuracy"] = accuracy
        report["accuracy_mask"] = "mask == 255 and finite positive GT" if mask is not None else "all finite positive GT; no visibility mask supplied"
        report["depth_accuracy_policy"] = "derived from supplied GT disparity and calibration; excludes nonpositive/nonfinite GT depth; reports coverage; not hardware measurement accuracy"
        report["inputs_sha256"]["ground_truth"] = hashlib.sha256(ground_truth.read_bytes()).hexdigest()
        if mask is not None:
            report["inputs_sha256"]["mask"] = hashlib.sha256(mask.read_bytes()).hexdigest()
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("left", "right", "calibration", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--method", choices=("bm", "sgbm", "sgbm_wls"), default="sgbm")
    parser.add_argument("--num-disp", type=int, help="Multiple of 16; default rounds calib.ndisp up and caps to image width")
    parser.add_argument("--block-size", type=int, default=7)
    parser.add_argument("--consistency", action="store_true")
    parser.add_argument("--wls-lambda", type=float, default=8000)
    parser.add_argument("--wls-sigma", type=float, default=0.8)
    parser.add_argument("--max-points", type=int, default=20_000, help="Maximum embedded viewer points; PLY uses --stride")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-depth-mm", type=float, help="Optional cloud-only far-plane clipping; no default distance cutoff")
    parser.add_argument("--ground-truth", type=Path, help="Optional grayscale PFM disparity, in pixels")
    parser.add_argument("--mask", type=Path, help="Optional ETH3D mask: only 255 is evaluated")
    parser.add_argument("--threads", type=int, default=1)
    args = vars(parser.parse_args())
    threads = args.pop("threads")
    if threads < 1:
        parser.error("--threads must be positive")
    cv2.setNumThreads(threads)
    try:
        report = reconstruct(**args)
    except (ValueError, OSError, RuntimeError, cv2.error) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"Open offline viewer: {(args['out'] / 'viewer.html').resolve()}")


if __name__ == "__main__":
    main()
