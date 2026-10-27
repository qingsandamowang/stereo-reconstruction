"""Evaluate BM, SGBM and consistency filtering on held-out ETH3D scenes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .geometry import write_ply
from .io import Calibration, non_occluded_pixels, parse_calibration, read_image, read_pfm
from .matching import match_stereo
from .metrics import disparity_metrics


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
    return {
        "scenes": len(rows),
        "valid_gt_pixels": count_gt,
        "valid_prediction_pixels": count_pred,
        "coverage": count_pred / count_gt,
        "epe_valid_px": sum((row["epe_valid_px"] or 0) * row["valid_prediction_pixels"] for row in rows) / count_pred if count_pred else None,
        "bad2_all": sum(row["bad2_all"] * row["valid_gt_pixels"] for row in rows) / count_gt,
        "bad2_scene_mean": statistics.mean(row["bad2_all"] for row in rows),
    }


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


def _evaluate_scene(scene: Scene, method: str, num_disp: int, block_size: int, consistency: bool = False) -> tuple[dict, np.ndarray]:
    start = perf_counter()
    disparity = match_stereo(scene.left, scene.right, method, num_disp, block_size, consistency)
    runtime_ms = (perf_counter() - start) * 1000
    row = disparity_metrics(scene.ground_truth, disparity, scene.non_occluded)
    row.update({"scene": scene.name, "runtime_ms": runtime_ms})
    return row, disparity


def _choose_parameters(scenes: list[Scene], method: str, blocks: list[int]) -> tuple[dict, list[dict]]:
    candidates = []
    for num_disp in (64, 96, 128):
        for block_size in blocks:
            rows = [_evaluate_scene(scene, method, num_disp, block_size)[0] for scene in scenes]
            aggregate = aggregate_metrics(rows)
            candidates.append({"num_disp": num_disp, "block_size": block_size, "validation": aggregate})
    selected = min(candidates, key=lambda item: (item["validation"]["bad2_scene_mean"], item["num_disp"], item["block_size"]))
    return {"num_disp": selected["num_disp"], "block_size": selected["block_size"]}, candidates


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
    parser.add_argument("--out", type=Path, default=Path("results/eth3d_v2"))
    args = parser.parse_args()
    names = [directory.name for directory in args.data.iterdir() if directory.is_dir()]
    validation_names, test_names = scene_split(names)
    validation_scenes = [_load_scene(args.data / name) for name in validation_names]
    test_scenes = [_load_scene(args.data / name) for name in test_names]
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_url": "https://www.eth3d.net/datasets",
        "input_sha256": hashlib.sha256((Path("data/raw") / "two_view_training.7z").read_bytes()).hexdigest(),
        "ground_truth_sha256": hashlib.sha256((Path("data/raw") / "two_view_training_gt.7z").read_bytes()).hexdigest(),
        "validation_scenes": validation_names,
        "test_scenes": test_names,
        "metric_mask": "mask0nocc == 255 and finite positive ground-truth disparity",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    reports = {}
    candidate_tables = {}
    for method, blocks in (("bm", [9, 15, 21]), ("sgbm", [3, 5, 7])):
        parameters, candidates = _choose_parameters(validation_scenes, method, blocks)
        candidate_tables[method] = candidates
        report, maps = _run_selected(test_scenes, method, parameters)
        reports[method] = report
        print(f"{method}: params={parameters} coverage={report['coverage']:.3f} EPE={report['epe_valid_px']:.3f} bad2={report['bad2_all']:.3f}", flush=True)
        if method == "sgbm":
            selected_maps = maps
            selected_parameters = parameters
    consistency_report, _ = _run_selected(test_scenes, "sgbm", selected_parameters, consistency=True)
    reports["sgbm_consistency"] = consistency_report
    print(f"sgbm+consistency: coverage={consistency_report['coverage']:.3f} EPE={consistency_report['epe_valid_px']:.3f} bad2={consistency_report['bad2_all']:.3f}", flush=True)
    (args.out / "validation_candidates.json").write_text(json.dumps(candidate_tables, indent=2), encoding="utf-8")
    (args.out / "test_report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    point_counts = {}
    for name in ("forest_1s", "playground_2l", "terrains_1s"):
        scene = next(scene for scene in test_scenes if scene.name == name)
        point_counts[name] = _visualize(scene, selected_maps[name], args.out / "visuals")
    (args.out / "point_counts.json").write_text(json.dumps(point_counts, indent=2), encoding="utf-8")
    print("exported points:", point_counts, flush=True)


if __name__ == "__main__":
    main()
