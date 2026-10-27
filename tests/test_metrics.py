import numpy as np


def test_disparity_metrics_penalize_invalid_predictions_in_bad2_all():
    from stereolab.metrics import disparity_metrics

    gt = np.array([[1, 2], [3, 0]], dtype=np.float32)
    predicted = np.array([[1, 5], [np.nan, 0]], dtype=np.float32)
    non_occluded = np.ones((2, 2), dtype=bool)

    result = disparity_metrics(gt, predicted, non_occluded)

    assert result["valid_gt_pixels"] == 3
    assert result["coverage"] == 2 / 3
    assert result["epe_valid_px"] == 1.5
    assert result["bad2_all"] == 2 / 3


def test_all_bad_thresholds_count_missing_and_use_strict_greater_than():
    from stereolab.metrics import disparity_metrics

    gt = np.full((1, 5), 10, dtype=np.float32)
    prediction = np.array([[10, 11, 12, 13, np.nan]], dtype=np.float32)
    result = disparity_metrics(gt, prediction, np.ones_like(gt, dtype=bool))

    assert result["bad1_valid"] == 2 / 4
    assert result["bad1_all"] == 3 / 5
    assert result["bad2_valid"] == 1 / 4
    assert result["bad2_all"] == 2 / 5
    assert result["bad3_valid"] == 0
    assert result["bad3_all"] == 1 / 5


def test_derived_depth_metrics_use_calibration_and_report_missing_coverage():
    from stereolab.io import Calibration
    from stereolab.metrics import depth_metrics

    gt = np.array([[10, 20, 10, np.nan, 0]], dtype=np.float32)
    predicted = np.array([[10, 10, np.nan, np.inf, 10]], dtype=np.float32)
    calibration = Calibration(100, 0, 0, 0, 20, 5, 1, 64)
    result = depth_metrics(gt, predicted, np.ones_like(gt, dtype=bool), calibration)
    assert result["depth_valid_gt_pixels"] == 3
    assert result["depth_valid_prediction_pixels"] == 2
    assert result["depth_coverage"] == 2 / 3
    assert result["depth_mae_valid_mm"] == 50
    assert np.isclose(result["depth_rmse_valid_mm"], np.sqrt(5000))


def test_depth_metrics_exclude_nonpositive_offset_denominator_and_masked_gt():
    from stereolab.io import Calibration
    from stereolab.metrics import depth_metrics

    gt = np.array([[5, 20, 30]], dtype=np.float32)
    predicted = np.array([[15, 5, 30]], dtype=np.float32)
    calibration = Calibration(100, 0, 0, -10, 20, 3, 1, 64)
    result = depth_metrics(gt, predicted, np.array([[True, True, False]]), calibration)
    assert result["depth_valid_gt_pixels"] == 1
    assert result["depth_gt_rejected_pixels"] == 1
    assert result["depth_coverage"] == 0
    assert result["depth_mae_valid_mm"] is None
    assert result["depth_rmse_valid_mm"] is None
