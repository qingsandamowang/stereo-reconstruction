"""Dense disparity accuracy with explicit missing-prediction coverage."""

from __future__ import annotations

import numpy as np

from .geometry import depth_from_disparity
from .io import Calibration


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
    missing = count_gt - count_valid
    result = {
        "valid_gt_pixels": count_gt,
        "valid_prediction_pixels": count_valid,
        "coverage": count_valid / count_gt,
        "epe_valid_px": float(errors.mean()) if count_valid else None,
    }
    for threshold in (1, 2, 3):
        bad_valid = int(np.count_nonzero(errors > threshold))
        result[f"bad{threshold}_valid"] = bad_valid / count_valid if count_valid else None
        result[f"bad{threshold}_all"] = (bad_valid + missing) / count_gt
    return result


def depth_metrics(gt: np.ndarray, prediction: np.ndarray, non_occluded: np.ndarray,
                  calibration: Calibration) -> dict[str, float | int | None]:
    """Depth error derived from calibrated GT disparity, not hardware measurement accuracy."""
    if gt.shape != prediction.shape or gt.shape != non_occluded.shape:
        raise ValueError("Disparity and mask shapes must match")
    gt_mask = np.asarray(non_occluded, dtype=bool) & np.isfinite(gt) & (gt > 0)
    gt_depth = depth_from_disparity(gt, calibration)
    pred_depth = depth_from_disparity(prediction, calibration)
    valid_gt = gt_mask & np.isfinite(gt_depth) & (gt_depth > 0)
    valid_pred = np.isfinite(prediction) & (prediction > 0) & np.isfinite(pred_depth) & (pred_depth > 0)
    joint = valid_gt & valid_pred
    count_gt, count_pred = int(valid_gt.sum()), int(joint.sum())
    errors = pred_depth[joint].astype(np.float64) - gt_depth[joint].astype(np.float64)
    return {"depth_valid_gt_pixels": count_gt, "depth_valid_prediction_pixels": count_pred,
            "depth_gt_rejected_pixels": int(gt_mask.sum()) - count_gt,
            "depth_coverage": count_pred / count_gt if count_gt else None,
            "depth_mae_valid_mm": float(np.abs(errors).mean()) if count_pred else None,
            "depth_rmse_valid_mm": float(np.sqrt(np.square(errors).mean())) if count_pred else None}
