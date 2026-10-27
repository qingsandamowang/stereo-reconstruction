import numpy as np


def test_left_right_consistency_keeps_only_reciprocal_matches():
    from stereolab.matching import left_right_consistency

    left = np.full((1, 6), np.nan, dtype=np.float32)
    right = np.full((1, 6), np.nan, dtype=np.float32)
    left[0, 3] = 2.0
    right[0, 1] = -2.0
    left[0, 4] = 2.0

    result = left_right_consistency(left, right, tolerance=0.5)

    assert result[0, 3]
    assert not result[0, 4]


def test_sgbm_recovers_shift_on_textured_stereo_pair():
    from stereolab.matching import match_stereo

    rng = np.random.default_rng(42)
    left = rng.integers(0, 255, size=(80, 160), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:, :-8] = left[:, 8:]

    disparity = match_stereo(left, right, method="sgbm", num_disp=64, block_size=5)

    assert disparity.shape == left.shape
    assert abs(float(np.nanmedian(disparity[:, 80:140])) - 8.0) < 1.0
