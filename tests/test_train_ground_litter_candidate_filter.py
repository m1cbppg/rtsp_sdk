import hashlib

import numpy as np
import pytest

from scripts.train_ground_litter_candidate_filter import (
    build_holdout_rows,
    build_training_rows,
    camera_excluded_scores,
    ensemble_scores,
    weighted_metrics,
)


def test_training_rows_use_only_clear_semantic_active_labels():
    base = {"rows": [{
        "sample_id": "base-1", "day": "day1", "device_code": "cam-a",
        "label": 1, "score": 0.4, "source": "semantic", "bbox": [0, 0, 4, 4],
    }]}
    selected = {"selected": [
        {"proposal_id": "a", "source": "semantic_tile", "device_code": "cam-a", "timestamp": "t", "score": 0.2, "bbox": [0, 0, 4, 4]},
        {"proposal_id": "b", "source": "semantic_full", "device_code": "cam-b", "timestamp": "t", "score": 0.2, "bbox": [0, 0, 4, 4]},
        {"proposal_id": "c", "source": "random_grid", "device_code": "cam-b", "timestamp": "t", "score": 0.0, "bbox": [0, 0, 4, 4]},
    ]}
    reviews = {"reviews": {
        "a": {"label": "NON_LITTER"}, "b": {"label": "UNCERTAIN"},
        "c": {"label": "LITTER"},
    }}
    rows, summary = build_training_rows(base, selected, reviews)
    assert [row["sample_id"] for row in rows] == ["base-1", "active_day3:a"]
    assert summary == {
        "base_rows": 1, "active": {"positive": 0, "negative": 1, "excluded": 1},
        "rows": 2, "positive": 1, "negative": 1,
    }


def test_weighted_metrics_restore_stratum_population():
    result = weighted_metrics(
        np.array([2.0, -1.0, 2.0, -1.0]), np.array([1, 1, 0, 0]), 0.0,
        np.array([9.0, 1.0, 1.0, 9.0]),
    )
    assert result["recall"] == pytest.approx(0.9)
    assert result["negative_reduction"] == pytest.approx(0.9)
    assert result["candidate_reduction"] == pytest.approx(0.5)


def test_holdout_identity_and_label_exclusion():
    rows = [
        {"proposal_id": "a", "device_code": "cam", "holdout_weight": 2.0},
        {"proposal_id": "b", "device_code": "cam", "holdout_weight": 2.0},
    ]
    identity = hashlib.sha256("a\nb".encode()).hexdigest()
    selected, excluded = build_holdout_rows(
        {"selected": rows, "identity_sha256": identity},
        {"reviews": {"a": {"label": "LITTER"}, "b": {"label": "BOX_WRONG"}}},
    )
    assert [row["proposal_id"] for row in selected] == ["a"]
    assert excluded["BOX_WRONG"] == 1


def test_artifact_prediction_contract():
    artifact = {
        "weights": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "biases": np.zeros(2, dtype=np.float32),
        "means": np.zeros((2, 2), dtype=np.float32),
        "stds": np.ones((2, 2), dtype=np.float32),
        "held_out_devices": np.array(["cam-a", "cam-b"]),
    }
    features = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert ensemble_scores(features, artifact).tolist() == pytest.approx([0.5, 0.5])
    assert camera_excluded_scores(
        features, ["cam-a", "cam-b"], artifact,
    ).tolist() == pytest.approx([1.0, 1.0])
