"""Dense disparity accuracy with explicit missing-prediction coverage."""

from __future__ import annotations

import numpy as np


def disparity_metrics(gt: np.ndarray, prediction: np.ndarray, non_occluded: np.ndarray) -> dict[str, float | int | None]:
    if gt.shape != prediction.shape or gt.shape != non_occluded.shape:
        raise ValueError("Disparity and mask shapes must match")
    valid_gt = np.asarray(non_occluded, dtype=bool) & np.isfinite(gt) & (gt > 0)
    valid_pred = np.isfinite(prediction) & (prediction > 0)
    count_gt = int(np.count_nonzero(valid_gt))
    if count_gt == 0:
        raise ValueError("No valid ground-truth pixels")
    jointly_valid = valid_gt & valid_pred
    count_valid = int(np.count_nonzero(jointly_valid))
    errors = np.abs(prediction[jointly_valid] - gt[jointly_valid])
    bad_valid = int(np.count_nonzero(errors > 2.0))
    missing = count_gt - count_valid
    return {
        "valid_gt_pixels": count_gt,
        "valid_prediction_pixels": count_valid,
        "coverage": count_valid / count_gt,
        "epe_valid_px": float(errors.mean()) if count_valid else None,
        "bad2_valid": bad_valid / count_valid if count_valid else None,
        "bad2_all": (bad_valid + missing) / count_gt,
    }
