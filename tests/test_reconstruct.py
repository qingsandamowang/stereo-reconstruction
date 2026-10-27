import json
from pathlib import Path

import cv2
import numpy as np
import pytest


def stereo_files(tmp_path, width=160, height=80):
    rng = np.random.default_rng(123)
    left = rng.integers(0, 255, (height, width), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:, :-8] = left[:, 8:]
    for name, array in (("left.png", left), ("right.png", right)):
        (tmp_path / name).write_bytes(cv2.imencode(".png", array)[1].tobytes())
    calibration = tmp_path / "calib.txt"
    calibration.write_text(f"cam0=[100 0 80; 0 120 40; 0 0 1]\ndoffs=0\nbaseline=20\nwidth={width}\nheight={height}\nndisp=64\n")
    return tmp_path / "left.png", tmp_path / "right.png", calibration


def test_synthetic_reconstruction_exports_geometry_and_no_accuracy_claim(tmp_path):
    from stereolab.reconstruct import reconstruct

    left, right, calibration = stereo_files(tmp_path)
    out = tmp_path / "out"
    report = reconstruct(left, right, calibration, out, max_points=117)

    disparity = np.load(out / "disparity.npy")
    depth = np.load(out / "depth_mm.npy")
    assert abs(float(np.nanmedian(disparity[:, 80:140])) - 8) < 0.1
    assert abs(float(np.nanmedian(depth[:, 80:140])) - 250) < 2
    assert "accuracy" not in report
    assert report["units"] == "mm"
    assert report["viewer_points"] == 117
    assert report["ply_points"] > 117
    assert "element vertex " + str(report["ply_points"]) in (out / "cloud.ply").read_text()
    html = (out / "viewer.html").read_text(encoding="utf-8")
    assert 'id="cloud-data"' in html
    assert "https://" not in html and "http://" not in html
    assert "pointermove" in html and "wheel" in html
    assert json.loads((out / "report.json").read_text()) == report


def test_custom_inputs_reject_dimension_mismatch_before_matching(tmp_path):
    from stereolab.reconstruct import reconstruct

    left, right, calibration = stereo_files(tmp_path)
    right.write_bytes(cv2.imencode(".png", np.zeros((80, 100), dtype=np.uint8))[1].tobytes())
    with pytest.raises(ValueError, match="dimensions"):
        reconstruct(left, right, calibration, tmp_path / "out")


def test_ground_truth_is_optional_and_supplied_truth_is_evaluated(tmp_path):
    from stereolab.reconstruct import reconstruct

    left, right, calibration = stereo_files(tmp_path)
    gt = tmp_path / "gt.pfm"
    gt.write_bytes(b"Pf\n160 80\n-1.0\n" + np.full((80, 160), 8, dtype="<f4").tobytes())
    report = reconstruct(left, right, calibration, tmp_path / "out", ground_truth=gt)
    assert report["accuracy"]["epe_valid_px"] < 0.1
    assert report["accuracy"]["coverage"] < 1
    assert report["accuracy"]["depth_mae_valid_mm"] < 1
    assert report["accuracy"]["depth_coverage"] < 1


def test_cli_runs_end_to_end_on_synthetic_pair(tmp_path):
    import os
    import subprocess
    import sys

    left, right, calibration = stereo_files(tmp_path)
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    command = [sys.executable, "-m", "stereolab.reconstruct", "--left", str(left), "--right", str(right),
               "--calibration", str(calibration), "--out", str(tmp_path / "out"), "--max-points", "100"]
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "out" / "viewer.html").is_file()


def test_viewer_handles_empty_cloud_and_escapes_title(tmp_path):
    from stereolab.viewer import write_viewer

    target = tmp_path / "empty.html"
    count = write_viewer(target, np.empty((0, 3)), np.empty((0, 3)), title="</script><img src=x>")
    html = target.read_text(encoding="utf-8")
    assert count == 0
    assert "No valid 3D points" in html
    assert "</script><img" not in html


def test_point_cloud_keeps_unusual_valid_calibration_and_ignores_nonfinite_depth():
    from stereolab.geometry import point_cloud
    from stereolab.io import Calibration

    # Principal point may lie outside a cropped image, fx and fy need not be equal.
    calibration = Calibration(100, -200, 500, 5, 20, 2, 1, 64, focal_y_px=200)
    points, colors = point_cloud(np.array([[5, np.nan]], dtype=np.float32),
                                 np.full((1, 2, 3), 100, dtype=np.uint8), calibration)
    assert np.allclose(points, [[400, -500, 200]])
    assert len(colors) == 1


def test_default_search_handles_calibration_range_equal_to_image_width(tmp_path):
    from stereolab.reconstruct import reconstruct

    left, right, calibration = stereo_files(tmp_path)
    calibration.write_text(calibration.read_text().replace("ndisp=64", "ndisp=160"))
    report = reconstruct(left, right, calibration, tmp_path / "out")
    assert report["parameters"]["num_disp"] == 144


def test_generated_demo_has_known_depth_and_explicit_synthetic_provenance(tmp_path):
    from stereolab.demo import run_demo

    report = run_demo(tmp_path / "demo")
    assert report["synthetic_demo"]["shift_px"] == 12
    assert report["synthetic_demo"]["expected_depth_mm"] == 900
    assert report["accuracy"]["epe_valid_px"] < 0.1
    assert abs(report["depth_median_mm"] - 900) < 2


def test_viewer_javascript_rotation_zoom_reset_and_empty_cloud(tmp_path):
    import shutil
    import subprocess
    from stereolab.viewer import write_viewer

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is optional; required only for the Canvas event harness")
    harness = Path(__file__).with_name("viewer_harness.cjs")
    for points in (np.empty((0, 3)), np.array([[0, 0, 100], [10, 5, 120]])):
        path = tmp_path / "viewer.html"
        write_viewer(path, points, np.full_like(points, 100))
        result = subprocess.run([node, str(harness), str(path)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
