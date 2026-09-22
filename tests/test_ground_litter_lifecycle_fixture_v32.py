import importlib.util
from pathlib import Path
import sys


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/build_ground_litter_lifecycle_fixture_v32.py"
)
SPEC = importlib.util.spec_from_file_location(
    "ground_litter_lifecycle_fixture_v32", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _event(event_id, first_seen, confirmed_at, *, closed_at=None,
           closed_reason=None, states=()):
    return {
        "event_id": event_id,
        "first_seen": first_seen,
        "confirmed_at": confirmed_at,
        "closed_at": closed_at,
        "closed_reason": closed_reason,
        "state": "VISIBLE_ANOMALY",
        "anchor_box": [1490, 1020, 1520, 1045],
        "state_history": [
            {"timestamp": first_seen, "state": state, "reason": "test"}
            for state in states
        ],
    }


def test_fixture_segments_are_contiguous_and_total_65_seconds():
    rows = MODULE.fixture_segments()
    assert rows[0]["fixture_start"] == 0.0
    assert rows[-1]["fixture_end"] == 65.0
    assert all(
        left["fixture_end"] == right["fixture_start"]
        for left, right in zip(rows, rows[1:])
    )


def test_validation_accepts_confirm_clear_and_new_same_location_event():
    result = MODULE.validate_lifecycle([
        _event(
            2, 11.0, 18.0,
            closed_at=36.0,
            closed_reason="clean_confirmed",
            states=("VISIBLE_ANOMALY", "OCCLUDED", "CLEAN_PENDING", "CLEARED"),
        ),
        _event(7, 46.0, 50.0, states=("VISIBLE_ANOMALY",)),
    ])
    assert result["passed"] is True
    assert result["baseline_decision"]["status"] == "ACCEPTED_AS_CURRENT_OFFLINE_BASELINE"
    assert result["first_event"]["event_id"] == 2
    assert result["second_event"]["event_id"] == 7


def test_validation_rejects_reappearance_before_clear():
    result = MODULE.validate_lifecycle([
        _event(
            2, 11.0, 18.0,
            closed_at=46.0,
            closed_reason="clean_confirmed",
            states=("VISIBLE_ANOMALY", "OCCLUDED", "CLEARED"),
        ),
        _event(7, 45.0, 50.0, states=("VISIBLE_ANOMALY",)),
    ])
    assert result["passed"] is False
    assert result["checks"]["first_event_cleared_during_clean_segment"] is False
    assert result["checks"]["second_event_started_after_first_cleared"] is False


def test_validation_rejects_duplicate_confirmed_target_event():
    result = MODULE.validate_lifecycle([
        _event(
            2, 11.0, 18.0,
            closed_at=36.0,
            closed_reason="clean_confirmed",
            states=("VISIBLE_ANOMALY", "OCCLUDED", "CLEARED"),
        ),
        _event(7, 46.0, 50.0, states=("VISIBLE_ANOMALY",)),
        _event(8, 53.0, 57.0, states=("VISIBLE_ANOMALY",)),
    ])
    assert result["passed"] is False
    assert result["checks"]["exactly_two_target_confirmed_primary_events"] is False
