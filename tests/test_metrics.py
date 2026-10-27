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
