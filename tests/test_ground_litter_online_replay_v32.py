import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_ground_litter_online_replay_v32.py"
SPEC = importlib.util.spec_from_file_location("ground_litter_online_replay_v32", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


BOX = [50, 50, 60, 60]


def _frame(
    memory,
    timestamp,
    candidates=(),
    support=None,
    valid=None,
    actors=(),
    environment_state="NORMAL",
):
    if support is None:
        support = np.zeros((120, 120), np.uint8)
    if valid is None:
        valid = np.ones((120, 120), np.uint8) * 255
    return memory.update(
        timestamp=float(timestamp),
        candidates=[{"box": list(box)} for box in candidates],
        support=support,
        valid=valid,
        actors=[list(box) for box in actors],
        environment_state=environment_state,
    )


def _changed():
    support = np.zeros((120, 120), np.uint8)
    support[50:60, 50:60] = 1
    return support


def test_closes_after_five_consecutive_valid_clean_samples():
    memory = MODULE.OnlineEventMemory()
    _frame(memory, 1, [BOX], _changed())
    for timestamp in range(2, 6):
        _frame(memory, timestamp)
    assert memory.events[0].closed_at is None
    _frame(memory, 6)
    assert memory.events[0].closed_at == 6.0
    assert memory.events[0].state == "CLEARED"


def test_environment_change_resets_clear_sequence():
    memory = MODULE.OnlineEventMemory()
    _frame(memory, 1, [BOX], _changed())
    for timestamp in range(2, 6):
        _frame(memory, timestamp)
    _frame(memory, 6, environment_state="ENVIRONMENT_CHANGE")
    _frame(memory, 7)
    assert memory.events[0].closed_at is None
    assert memory.events[0].clear_observed_seconds == 1.0
    for timestamp in range(8, 12):
        _frame(memory, timestamp)
    assert memory.events[0].closed_at == 11.0


def test_large_sampling_gap_restarts_clear_sequence():
    memory = MODULE.OnlineEventMemory()
    for timestamp in range(1, 6):
        _frame(memory, timestamp, [BOX], _changed())
    for timestamp in (6, 7, 8, 9, 20):
        _frame(memory, timestamp)
    assert memory.events[0].closed_at is None
    assert memory.events[0].clear_observed_seconds == 1.0


def test_invalid_ground_cannot_close_event():
    memory = MODULE.OnlineEventMemory()
    _frame(memory, 1, [BOX], _changed())
    invalid = np.zeros((120, 120), np.uint8)
    for timestamp in range(2, 8):
        counts = _frame(memory, timestamp, valid=invalid)
    assert counts["ground_unavailable"] == 1
    assert memory.events[0].closed_at is None
    assert memory.events[0].state == "GROUND_UNAVAILABLE"
    assert memory.events[0].clear_observed_seconds == 0.0


def test_valid_fraction_uses_anchor_footprint_not_support_padding():
    valid = np.zeros((120, 120), np.uint8)
    valid[50:60, 50:60] = 255
    observation = MODULE.anchor_observation(
        np.zeros_like(valid), valid, BOX
    )
    assert observation["valid_fraction"] == 1.0
    assert observation["valid_pixels"] == 100


def test_invalid_ground_does_not_consume_pending_timeout():
    memory = MODULE.OnlineEventMemory()
    _frame(memory, 1, [BOX], _changed())
    invalid = np.zeros((120, 120), np.uint8)
    for timestamp in range(2, 22):
        _frame(memory, timestamp, valid=invalid)
    assert memory.events[0].closed_at is None
    assert memory.events[0].state == "GROUND_UNAVAILABLE"
    _frame(memory, 22, support=_changed())
    assert memory.events[0].closed_at is None
    assert memory.events[0].pending_unmatched_seconds == 1.0


def test_pending_timeout_counts_only_judgeable_missing_samples():
    memory = MODULE.OnlineEventMemory()
    _frame(memory, 1, [BOX], _changed())
    for timestamp in range(2, 16):
        _frame(memory, timestamp, support=_changed())
    assert memory.events[0].closed_at is None
    _frame(memory, 16, support=_changed())
    assert memory.events[0].state == "EXPIRED_PENDING"
    assert memory.events[0].closed_at == 16.0


def test_actor_occlusion_resets_clear_sequence_but_keeps_event():
    memory = MODULE.OnlineEventMemory()
    _frame(memory, 1, [BOX], _changed())
    for timestamp in (2, 3, 4):
        _frame(memory, timestamp)
    _frame(memory, 5, actors=[[40, 40, 80, 90]])
    _frame(memory, 6)
    assert memory.events[0].closed_at is None
    assert memory.events[0].clear_observed_seconds == 1.0


def test_visible_confirmation_remains_cumulative_across_occlusion():
    memory = MODULE.OnlineEventMemory()
    for timestamp in (1, 2, 3):
        _frame(memory, timestamp, [BOX], _changed())
    _frame(memory, 4, support=_changed(), actors=[[40, 40, 80, 90]])
    for timestamp in (5, 6):
        _frame(memory, timestamp, [BOX], _changed())
    assert memory.events[0].confirmed_at == 6.0
    assert memory.events[0].visible_evidence_seconds == 5.0


def test_same_location_after_clear_creates_new_event():
    memory = MODULE.OnlineEventMemory()
    _frame(memory, 1, [BOX], _changed())
    for timestamp in range(2, 7):
        _frame(memory, timestamp)
    assert memory.events[0].state == "CLEARED"
    _frame(memory, 7, [BOX], _changed())
    assert [event.event_id for event in memory.events] == [1, 2]
    assert memory.events[1].state == "VISIBLE_ANOMALY"


def test_converged_events_merge_at_the_association_distance():
    memory = MODULE.OnlineEventMemory()
    first_box = [10, 10, 20, 20]
    second_box = [35, 10, 45, 20]
    first = MODULE.OnlineEvent(
        event_id=1, first_seen=1.0, anchor_box=first_box, boxes=[first_box]
    )
    second = MODULE.OnlineEvent(
        event_id=2, first_seen=1.0, anchor_box=second_box, boxes=[second_box]
    )
    memory.events = [first, second]
    memory._merge_converged_events(2.0)
    assert [event.event_id for event in memory.active] == [1]
    assert second.closed_reason == "merged_into:1"


def test_diagnostic_histories_are_bounded():
    event = MODULE.OnlineEvent(event_id=1, first_seen=0.0, anchor_box=list(BOX))
    for timestamp in range(MODULE.MAX_VISIBLE_TIMESTAMP_HISTORY + 2):
        event.observe(float(timestamp), BOX, 1.0)
        event.transition(float(timestamp), "A" if timestamp % 2 else "B", "test")
    assert len(event.visible_timestamps) == MODULE.MAX_VISIBLE_TIMESTAMP_HISTORY
    assert event.visible_history_truncated is True
    assert len(event.state_history) == MODULE.MAX_STATE_HISTORY
    assert event.state_history_truncated is True
    assert event.visible_observation_count == MODULE.MAX_VISIBLE_TIMESTAMP_HISTORY + 2


def test_closed_event_retention_is_bounded_and_counted():
    memory = MODULE.OnlineEventMemory(max_closed_events=1)
    other_box = [90, 90, 100, 100]
    _frame(memory, 1, [BOX], _changed())
    for timestamp in range(2, 7):
        _frame(memory, timestamp)
    _frame(memory, 7, [other_box], _changed())
    for timestamp in range(8, 13):
        _frame(memory, timestamp)
    assert len(memory.events) == 1
    assert memory.events[0].event_id == 2
    assert memory.events_pruned == 1
    assert memory.confirmed_events_pruned == 0


def test_rejects_duplicate_or_out_of_order_timestamps():
    memory = MODULE.OnlineEventMemory()
    _frame(memory, 1)
    with pytest.raises(ValueError, match="strictly increasing"):
        _frame(memory, 1)
