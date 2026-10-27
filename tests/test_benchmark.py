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
        {"valid_gt_pixels": 10, "valid_prediction_pixels": 5, "epe_valid_px": 1.0, "bad2_all": 0.5},
        {"valid_gt_pixels": 30, "valid_prediction_pixels": 30, "epe_valid_px": 3.0, "bad2_all": 0.1},
    ]

    result = aggregate_metrics(rows)

    assert result["coverage"] == 35 / 40
    assert result["epe_valid_px"] == 95 / 35
    assert result["bad2_all"] == 0.2
