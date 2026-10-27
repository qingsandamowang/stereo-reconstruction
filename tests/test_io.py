from pathlib import Path

import numpy as np


def test_read_pfm_restores_top_to_bottom_order(tmp_path: Path):
    from stereolab.io import read_pfm

    path = tmp_path / "disparity.pfm"
    path.write_bytes(b"Pf\n2 2\n-1.0\n" + np.array([[3, 4], [1, 2]], dtype="<f4").tobytes())

    result = read_pfm(path)

    assert np.array_equal(result, np.array([[1, 2], [3, 4]], dtype=np.float32))


def test_parse_calibration_reads_focal_baseline_and_offset(tmp_path: Path):
    from stereolab.io import parse_calibration

    path = tmp_path / "calib.txt"
    path.write_text(
        "cam0=[500 0 200; 0 500 100; 0 0 1]\n"
        "cam1=[500 0 205; 0 500 100; 0 0 1]\n"
        "doffs=5\nbaseline=60\nwidth=400\nheight=200\nndisp=128\n"
    )

    calibration = parse_calibration(path)

    assert calibration.focal_px == 500
    assert calibration.baseline_mm == 60
    assert calibration.doffs_px == 5
    assert calibration.cx_px == 200
    assert calibration.cy_px == 100
    assert calibration.width == 400


def test_non_occluded_mask_accepts_only_pixels_visible_in_both_views():
    from stereolab.io import non_occluded_pixels

    # ETH3D: 0=no GT, 128=left only, 255=visible in both cameras.
    encoded = np.array([[0, 128, 255]], dtype=np.uint8)

    assert np.array_equal(non_occluded_pixels(encoded), np.array([[False, False, True]]))
