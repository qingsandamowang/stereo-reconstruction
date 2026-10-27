import numpy as np


def test_depth_uses_focal_baseline_and_principal_point_offset():
    from stereolab.geometry import depth_from_disparity
    from stereolab.io import Calibration

    calibration = Calibration(500, 200, 100, 5, 60, 2, 1, 128)
    depth = depth_from_disparity(np.array([[25.0, -5.0]], dtype=np.float32), calibration)

    assert depth[0, 0] == 1000.0
    assert np.isnan(depth[0, 1])


def test_write_ply_uses_calibrated_xyz_and_color(tmp_path):
    from stereolab.geometry import write_ply
    from stereolab.io import Calibration

    calibration = Calibration(100, 0, 0, 0, 20, 2, 1, 64)
    disparity = np.array([[10.0, np.nan]], dtype=np.float32)
    color = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    path = tmp_path / "cloud.ply"

    count = write_ply(path, disparity, color, calibration)
    content = path.read_text()

    assert count == 1
    assert "element vertex 1" in content
    assert "0.000 0.000 200.000 10 20 30" in content
