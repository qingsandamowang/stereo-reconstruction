"""Validation-only selection and held-out ETH3D evaluation of four stereo methods."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .geometry import write_ply
from .io import Calibration, non_occluded_pixels, parse_calibration, read_image, read_pfm
from .matching import match_stereo
from .metrics import depth_metrics, disparity_metrics


@dataclass
class Scene:
    name: str
    left: np.ndarray
    right: np.ndarray
    rgb: np.ndarray
    ground_truth: np.ndarray
    non_occluded: np.ndarray
    calibration: Calibration


def scene_split(names: list[str]) -> tuple[list[str], list[str]]:
    """Reserve entire capture families for held-out testing."""
    validation_families = {"delivery_area", "electro", "facade"}
    validation, test = [], []
    for name in sorted(names):
        match = re.fullmatch(r"(.+)_\d+[ls]", name)
        if not match:
            raise ValueError(f"Unexpected ETH3D scene name: {name}")
        (validation if match.group(1) in validation_families else test).append(name)
    if not validation or not test:
        raise ValueError("Both validation and test scene groups are required")
    return validation, test


def aggregate_metrics(rows: list[dict]) -> dict:
    """Weight coverage and errors by their corresponding valid pixel counts."""
    count_gt = sum(row["valid_gt_pixels"] for row in rows)
    count_pred = sum(row["valid_prediction_pixels"] for row in rows)
    if count_gt == 0:
        raise ValueError("No valid ground-truth pixels")
    aggregate = {
        "scenes": len(rows),
        "valid_gt_pixels": count_gt,
        "valid_prediction_pixels": count_pred,
        "coverage": count_pred / count_gt,
        "epe_valid_px": sum((row["epe_valid_px"] or 0) * row["valid_prediction_pixels"] for row in rows) / count_pred if count_pred else None,
    }
    for threshold in (1, 2, 3):
        key = f"bad{threshold}"
        aggregate[f"{key}_all"] = sum(row[f"{key}_all"] * row["valid_gt_pixels"] for row in rows) / count_gt
        aggregate[f"{key}_valid"] = sum((row[f"{key}_valid"] or 0) * row["valid_prediction_pixels"] for row in rows) / count_pred if count_pred else None
        aggregate[f"{key}_scene_mean"] = statistics.mean(row[f"{key}_all"] for row in rows)
    if all("depth_valid_gt_pixels" in row for row in rows):
        depth_gt = sum(row["depth_valid_gt_pixels"] for row in rows)
        depth_pred = sum(row["depth_valid_prediction_pixels"] for row in rows)
        aggregate.update({"depth_valid_gt_pixels": depth_gt, "depth_valid_prediction_pixels": depth_pred,
                          "depth_gt_rejected_pixels": sum(row["depth_gt_rejected_pixels"] for row in rows),
                          "depth_coverage": depth_pred / depth_gt if depth_gt else None,
                          "depth_mae_valid_mm": sum((row["depth_mae_valid_mm"] or 0) * row["depth_valid_prediction_pixels"] for row in rows) / depth_pred if depth_pred else None,
                          "depth_rmse_valid_mm": float(np.sqrt(sum((row["depth_rmse_valid_mm"] or 0) ** 2 * row["depth_valid_prediction_pixels"] for row in rows) / depth_pred)) if depth_pred else None})
    return aggregate


def _load_scene(directory: Path) -> Scene:
    calibration = parse_calibration(directory / "calib.txt")
    left = read_image(directory / "im0.png")
    right = read_image(directory / "im1.png")
    rgb = read_image(directory / "im0.png", color=True)
    gt = read_pfm(directory / "disp0GT.pfm")
    non_occluded = non_occluded_pixels(read_image(directory / "mask0nocc.png"))
    expected = (calibration.height, calibration.width)
    if not all(array.shape[:2] == expected for array in (left, right, rgb, gt, non_occluded)):
        raise ValueError(f"Image, mask and calibration dimensions differ in {directory}")
    return Scene(directory.name, left, right, rgb, gt, non_occluded, calibration)


def _evaluate_scene(scene: Scene, method: str, num_disp: int, block_size: int, consistency: bool = False, **filter_parameters) -> tuple[dict, np.ndarray]:
    start = perf_counter()
    disparity = match_stereo(scene.left, scene.right, method, num_disp, block_size, consistency, **filter_parameters)
    runtime_ms = (perf_counter() - start) * 1000
    row = disparity_metrics(scene.ground_truth, disparity, scene.non_occluded)
    row.update(depth_metrics(scene.ground_truth, disparity, scene.non_occluded, scene.calibration))
    row.update({"scene": scene.name, "runtime_ms": runtime_ms})
    return row, disparity


def _choose_parameters(scenes: list[Scene], method: str, blocks: list[int]) -> tuple[dict, list[dict]]:
    candidates = []
    for num_disp in (64, 96, 128):
        for block_size in blocks:
            rows = [_evaluate_scene(scene, method, num_disp, block_size)[0] for scene in scenes]
            aggregate = aggregate_metrics(rows)
            candidates.append({"num_disp": num_disp, "block_size": block_size, "validation": aggregate, "per_scene": rows,
                               "median_matching_ms": statistics.median(row["runtime_ms"] for row in rows)})
    selected = min(candidates, key=lambda item: (item["validation"]["bad2_scene_mean"], item["num_disp"], item["block_size"]))
    return {"num_disp": selected["num_disp"], "block_size": selected["block_size"]}, candidates


def _choose_wls_parameters(scenes: list[Scene], base_parameters: dict) -> tuple[dict, list[dict]]:
    """Freeze the validation-selected SGBM base, then select the WLS filter on validation only."""
    candidates = []
    for wls_lambda in (8000, 20000):
        for wls_sigma in (0.8, 1.2, 1.6):
            parameters = {**base_parameters, "wls_lambda": wls_lambda, "wls_sigma": wls_sigma}
            rows = [_evaluate_scene(scene, "sgbm_wls", **parameters)[0] for scene in scenes]
            candidates.append({**parameters, "validation": aggregate_metrics(rows), "per_scene": rows,
                               "median_matching_ms": statistics.median(row["runtime_ms"] for row in rows)})
    selected = min(candidates, key=lambda item: (item["validation"]["bad2_scene_mean"], item["wls_lambda"], item["wls_sigma"]))
    return {key: selected[key] for key in ("num_disp", "block_size", "wls_lambda", "wls_sigma")}, candidates


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest(data: Path) -> dict:
    """Support an external data root and extracted-only/source-zip installations."""
    archives = {}
    for filename in ("two_view_training.7z", "two_view_training_gt.7z"):
        path = data.parent / "raw" / filename
        archives[filename] = {"sha256": _sha256(path) if path.is_file() else None,
                              "status": "present" if path.is_file() else "unavailable; extracted scenes hashed below"}
    scenes = {}
    for directory in sorted(data.iterdir()):
        if directory.is_dir():
            scenes[directory.name] = {name: _sha256(directory / name) for name in
                                     ("im0.png", "im1.png", "calib.txt", "disp0GT.pfm", "mask0nocc.png")
                                     if (directory / name).is_file()}
    return {"source_url": "https://www.eth3d.net/datasets", "archives": archives, "scene_files_sha256": scenes}


def _run_selected(scenes: list[Scene], method: str, parameters: dict, consistency: bool = False) -> tuple[dict, dict[str, np.ndarray]]:
    rows = []
    maps = {}
    for scene in scenes:
        row, disparity = _evaluate_scene(scene, method, **parameters, consistency=consistency)
        rows.append(row)
        maps[scene.name] = disparity
    aggregate = aggregate_metrics(rows)
    aggregate.update(
        {
            "method": method,
            "left_right_consistency": consistency,
            "parameters": parameters,
            "median_matching_ms": statistics.median(row["runtime_ms"] for row in rows),
            "per_scene": rows,
        }
    )
    return aggregate, maps


def _colorize(values: np.ndarray, maximum: float, valid: np.ndarray) -> np.ndarray:
    scaled = np.clip(np.nan_to_num(values, nan=0.0, posinf=maximum), 0, maximum)
    color = cv2.applyColorMap(np.uint8(scaled / maximum * 255), cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def _visualize(scene: Scene, predicted: np.ndarray, out: Path) -> int:
    valid_gt = scene.non_occluded & np.isfinite(scene.ground_truth) & (scene.ground_truth > 0)
    valid_prediction = np.isfinite(predicted) & (predicted > 0)
    error = np.abs(predicted - scene.ground_truth)
    panels = [
        cv2.cvtColor(scene.rgb, cv2.COLOR_RGB2BGR),
        _colorize(scene.ground_truth, 64.0, valid_gt),
        _colorize(predicted, 64.0, valid_prediction),
        _colorize(error, 5.0, valid_gt & valid_prediction),
    ]
    for panel, label in zip(panels, ("left", "GT disparity", "SGBM disparity", "error 0-5px")):
        cv2.putText(panel, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{scene.name}_comparison.png").write_bytes(cv2.imencode(".png", np.concatenate(panels, axis=1))[1].tobytes())
    return write_ply(out / f"{scene.name}_cloud.ply", predicted, scene.rgb, scene.calibration, stride=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/eth3d"))
    parser.add_argument("--out", type=Path, default=Path("results/eth3d_v3"))
    parser.add_argument("--threads", type=int, default=1, help="OpenCV CPU threads, recorded for timing reproducibility")
    parser.add_argument("--visuals", action="store_true", help="Generate local-only images and PLY for three held-out scenes")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.out.resolve() == Path("results/eth3d_v2").resolve():
        parser.error("eth3d_v2 is the preserved historical report; choose a different --out")
    cv2.setNumThreads(args.threads)
    names = [directory.name for directory in args.data.iterdir() if directory.is_dir()]
    validation_names, test_names = scene_split(names)
    validation_scenes = [_load_scene(args.data / name) for name in validation_names]
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        **source_manifest(args.data),
        "report_version": 3,
        "environment": {"python": sys.version.split()[0], "opencv": cv2.__version__, "numpy": np.__version__,
                        "platform": platform.platform(), "processor": platform.processor(), "opencv_threads": cv2.getNumThreads()},
        "validation_scenes": validation_names,
        "test_scenes": test_names,
        "metric_mask": "mask0nocc == 255 and finite positive ground-truth disparity",
        "selection_metric": "minimum mean per-scene bad2_all on validation; deterministic numeric parameter tie break",
        "wls_policy": "validation-selected SGBM base, then 6 lambda/sigma candidates; preserve raw positive-disparity mask; no hole filling",
        "depth_metric_policy": "depth derived from GT/predicted disparity and supplied calibration; same visibility/positive GT mask, plus finite positive depth; valid-point MAE/RMSE in mm with depth coverage; not hardware measurement accuracy",
        "timing_protocol": "one measured CPU call per scene; matcher creation + left/right matching + filter included; IO/metrics/export excluded; no warm-up; not a robust latency benchmark",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    candidate_tables = {}
    selected = {}
    for method, blocks in (("bm", [9, 15, 21]), ("sgbm", [3, 5, 7])):
        parameters, candidates = _choose_parameters(validation_scenes, method, blocks)
        candidate_tables[method] = candidates
        selected[method] = parameters
        print(f"validation selected {method}: {parameters}", flush=True)
    selected["sgbm_wls"], candidate_tables["sgbm_wls"] = _choose_wls_parameters(validation_scenes, selected["sgbm"])
    selected["sgbm_consistency"] = selected["sgbm"].copy()
    (args.out / "validation_candidates.json").write_text(json.dumps(candidate_tables, indent=2), encoding="utf-8")
    # Persist this checkpoint before loading/evaluating any held-out images/GT.
    (args.out / "selected_parameters.json").write_text(json.dumps({"selection_split": "validation", "parameters": selected}, indent=2), encoding="utf-8")
    print(f"validation selected WLS: {selected['sgbm_wls']}; all parameters frozen", flush=True)
    test_scenes = [_load_scene(args.data / name) for name in test_names]
    reports = {}
    for name, parameters in selected.items():
        method = "sgbm" if name == "sgbm_consistency" else name
        report, maps = _run_selected(test_scenes, method, parameters, consistency=name == "sgbm_consistency")
        reports[name] = report
        print(f"{name}: coverage={report['coverage']:.3f} EPE={report['epe_valid_px']:.3f} bad1={report['bad1_all']:.3f} bad2={report['bad2_all']:.3f} bad3={report['bad3_all']:.3f} median={report['median_matching_ms']:.1f}ms", flush=True)
        if name == "sgbm":
            selected_maps = maps
    (args.out / "test_report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    if args.visuals:
        point_counts = {}
        for name in ("forest_1s", "playground_2l", "terrains_1s"):
            scene = next(scene for scene in test_scenes if scene.name == name)
            point_counts[name] = _visualize(scene, selected_maps[name], args.out / "visuals")
        (args.out / "point_counts.json").write_text(json.dumps(point_counts, indent=2), encoding="utf-8")
        print("exported points:", point_counts, flush=True)


if __name__ == "__main__":
    main()
