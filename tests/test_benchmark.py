def test_scene_split_keeps_related_pairs_together():
    from stereolab.benchmark import scene_split

    validation, test = scene_split([
        "delivery_area_1l", "delivery_area_1s", "electro_1l", "forest_1s", "playground_1l"
    ])

    assert validation == ["delivery_area_1l", "delivery_area_1s", "electro_1l"]
    assert test == ["forest_1s", "playground_1l"]


def test_aggregate_metrics_weights_pixel_counts():
    from stereolab.benchmark import aggregate_metrics

    rows = [
        {"valid_gt_pixels": 10, "valid_prediction_pixels": 5, "epe_valid_px": 1.0,
         "bad1_all": 0.8, "bad2_all": 0.5, "bad3_all": 0.5,
         "bad1_valid": 0.6, "bad2_valid": 0.0, "bad3_valid": 0.0},
        {"valid_gt_pixels": 30, "valid_prediction_pixels": 30, "epe_valid_px": 3.0,
         "bad1_all": 0.2, "bad2_all": 0.1, "bad3_all": 0.0,
         "bad1_valid": 0.2, "bad2_valid": 0.1, "bad3_valid": 0.0},
    ]

    result = aggregate_metrics(rows)

    assert result["coverage"] == 35 / 40
    assert result["epe_valid_px"] == 95 / 35
    assert result["bad2_all"] == 0.2
    assert result["bad1_all"] == 14 / 40
    assert result["bad3_all"] == 5 / 40


def test_wls_selection_logs_every_validation_candidate(monkeypatch):
    from types import SimpleNamespace
    import stereolab.benchmark as benchmark

    seen = []

    def evaluate(scene, method, num_disp, block_size, **kwargs):
        assert scene.name.startswith("validation")
        assert method == "sgbm_wls"
        assert (num_disp, block_size) == (96, 5)
        seen.append((scene.name, kwargs["wls_lambda"], kwargs["wls_sigma"]))
        score = abs(kwargs["wls_lambda"] - 8000) / 100000 + abs(kwargs["wls_sigma"] - 1.2) / 10
        row = {"valid_gt_pixels": 10, "valid_prediction_pixels": 10,
               "epe_valid_px": score, "runtime_ms": 3, "scene": scene.name}
        for threshold in (1, 2, 3):
            row[f"bad{threshold}_all"] = row[f"bad{threshold}_valid"] = score
        return row, None

    monkeypatch.setattr(benchmark, "_evaluate_scene", evaluate)
    selected, candidates = benchmark._choose_wls_parameters(
        [SimpleNamespace(name="validation_a"), SimpleNamespace(name="validation_b")],
        {"num_disp": 96, "block_size": 5},
    )

    assert len(candidates) == 6
    assert len(seen) == 12
    assert selected == {"num_disp": 96, "block_size": 5, "wls_lambda": 8000, "wls_sigma": 1.2}
    assert all(len(candidate["per_scene"]) == 2 for candidate in candidates)


def test_manifest_can_hash_external_archives_or_extracted_only_data(tmp_path):
    from stereolab.benchmark import source_manifest

    data = tmp_path / "external" / "eth3d"
    data.mkdir(parents=True)
    manifest = source_manifest(data)
    assert manifest["archives"]["two_view_training.7z"]["sha256"] is None
    raw = data.parent / "raw"
    raw.mkdir()
    (raw / "two_view_training.7z").write_bytes(b"public archive")
    manifest = source_manifest(data)
    assert len(manifest["archives"]["two_view_training.7z"]["sha256"]) == 64


def test_aggregate_depth_rmse_weights_squared_error_by_valid_depth_count():
    from stereolab.benchmark import aggregate_metrics
    from stereolab.io import Calibration
    from stereolab.metrics import depth_metrics, disparity_metrics
    import numpy as np

    calibration = Calibration(100, 0, 0, 0, 20, 2, 1, 64)
    rows = []
    for prediction in ([[10, np.nan]], [[20, 20]]):
        gt = np.full((1, 2), 10, dtype=np.float32)
        pred = np.array(prediction, dtype=np.float32)
        mask = np.ones_like(gt, dtype=bool)
        rows.append({**disparity_metrics(gt, pred, mask), **depth_metrics(gt, pred, mask, calibration)})
    result = aggregate_metrics(rows)
    assert result["depth_coverage"] == 3 / 4
    assert result["depth_mae_valid_mm"] == 200 / 3
    assert np.isclose(result["depth_rmse_valid_mm"], np.sqrt(20000 / 3))
