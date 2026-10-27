import numpy as np
import pytest


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


def test_consistency_never_accepts_right_invalid_sentinel(monkeypatch):
    import stereolab.matching as matching

    class FakeMatcher:
        def __init__(self, right_view):
            self.right_view = right_view

        def compute(self, left, right):
            # With right minDisparity=-63, OpenCV's invalid value is -64.
            return np.full(left.shape, (-64 if self.right_view else 63) * 16, dtype=np.int16)

    monkeypatch.setattr(matching, "_matcher", lambda method, num_disp, block_size, right_view=False: FakeMatcher(right_view))
    image = np.zeros((3, 160), dtype=np.uint8)

    disparity = matching.match_stereo(image, image, consistency=True)

    assert not np.isfinite(disparity).any()


def test_wls_recovers_shift_without_filling_raw_invalid_border():
    from stereolab.matching import match_stereo

    rng = np.random.default_rng(42)
    left = rng.integers(0, 255, size=(80, 160), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:, :-8] = left[:, 8:]
    raw = match_stereo(left, right)
    filtered = match_stereo(left, right, method="sgbm_wls", wls_lambda=8000, wls_sigma=1.2)

    assert abs(float(np.nanmedian(filtered[:, 80:140])) - 8) < 1
    assert not np.isfinite(filtered[~np.isfinite(raw)]).any()


def test_wls_missing_contrib_has_actionable_error(monkeypatch):
    import stereolab.matching as matching

    monkeypatch.delattr(matching.cv2, "ximgproc", raising=False)
    image = np.zeros((20, 160), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="opencv-contrib-python"):
        matching.match_stereo(image, image, method="sgbm_wls")


@pytest.mark.parametrize("value", [63.0, 0.25])
def test_consistency_keeps_valid_negative_right_disparities(monkeypatch, value):
    import stereolab.matching as matching

    class FakeMatcher:
        def __init__(self, right_view):
            self.right_view = right_view

        def compute(self, left, right):
            return np.full(left.shape, (-value if self.right_view else value) * 16, dtype=np.int16)

    monkeypatch.setattr(matching, "_matcher", lambda method, num_disp, block_size, right_view=False: FakeMatcher(right_view))
    image = np.zeros((3, 160), dtype=np.uint8)
    disparity = matching.match_stereo(image, image, consistency=True)
    assert np.isfinite(disparity[:, 80:]).all()


def test_consistency_ignores_infinite_and_out_of_image_values_without_cast_warning():
    import warnings
    from stereolab.matching import left_right_consistency

    left = np.array([[np.inf, np.nan, 1e30, 2]], dtype=np.float32)
    right = np.array([[np.nan, -2, np.nan, np.nan]], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        keep = left_right_consistency(left, right)
    assert np.array_equal(keep, [[False, False, False, True]])
