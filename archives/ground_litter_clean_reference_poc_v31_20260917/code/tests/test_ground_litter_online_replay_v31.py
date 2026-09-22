import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_ground_litter_online_replay_v31.py"
SPEC = importlib.util.spec_from_file_location("ground_litter_online_replay_v31", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _frame(memory, timestamp, candidates=(), support=None, actors=()):
    if support is None:
        support = np.zeros((120, 120), np.uint8)
    return memory.update(
        timestamp=float(timestamp),
        candidates=[{"box": list(box)} for box in candidates],
        support=support,
        valid=np.ones((120, 120), np.uint8) * 255,
        actors=[list(box) for box in actors],
        environment_state="NORMAL",
    )


def test_event_survives_actor_occlusion_and_reappears_with_same_id():
    memory = MODULE.OnlineEventMemory()
    box = [50, 50, 60, 60]
    changed = np.zeros((120, 120), np.uint8)
    changed[50:60, 50:60] = 1
    _frame(memory, 1, [box], changed)
    _frame(memory, 2, [], changed, [[40, 40, 80, 90]])
    _frame(memory, 3, [box], changed)
    assert len(memory.events) == 1
    assert memory.events[0].visible_timestamps == [1.0, 3.0]
    assert any(row["state"] == "OCCLUDED" for row in memory.events[0].state_history)


def test_event_closes_only_after_consecutive_clean_observations():
    memory = MODULE.OnlineEventMemory()
    box = [50, 50, 60, 60]
    changed = np.zeros((120, 120), np.uint8)
    changed[50:60, 50:60] = 1
    clean = np.zeros((120, 120), np.uint8)
    _frame(memory, 1, [box], changed)
    for timestamp in range(2, 6):
        _frame(memory, timestamp, [], clean)
    assert memory.events[0].closed_at is None
    _frame(memory, 6, [], clean)
    assert memory.events[0].closed_at == 6.0
    assert memory.events[0].state == "CLEARED"


def test_residual_without_candidate_does_not_close_event():
    memory = MODULE.OnlineEventMemory()
    box = [50, 50, 60, 60]
    changed = np.zeros((120, 120), np.uint8)
    changed[50:60, 50:60] = 1
    _frame(memory, 1, [box], changed)
    for timestamp in range(2, 10):
        _frame(memory, timestamp, [], changed)
    assert memory.events[0].closed_at is None
    assert memory.events[0].state == "ANOMALY_PENDING"


def test_unconfirmed_event_expires_after_visible_pending_timeout():
    memory = MODULE.OnlineEventMemory()
    box = [50, 50, 60, 60]
    changed = np.zeros((120, 120), np.uint8)
    changed[50:60, 50:60] = 1
    _frame(memory, 1, [box], changed)
    for timestamp in range(2, 17):
        _frame(memory, timestamp, [], changed)
    assert memory.events[0].closed_at == 16.0
    assert memory.events[0].state == "EXPIRED_PENDING"
    assert memory.events[0].closed_reason == "pending_timeout"


def test_multiple_nearby_components_in_one_frame_share_one_event():
    memory = MODULE.OnlineEventMemory()
    changed = np.zeros((120, 120), np.uint8)
    changed[50:60, 50:60] = 1
    changed[51:61, 62:72] = 1
    _frame(memory, 1, [[50, 50, 60, 60]], changed)
    _frame(memory, 2, [[50, 50, 60, 60], [62, 51, 72, 61]], changed)
    assert len(memory.events) == 1
    assert memory.events[0].visible_timestamps == [1.0, 2.0]


def test_converged_active_events_merge_without_double_counting_hits():
    memory = MODULE.OnlineEventMemory()
    changed = np.zeros((120, 120), np.uint8)
    changed[20:90, 20:100] = 1
    # Start far enough apart to create two live events.
    _frame(memory, 1, [[20, 50, 30, 60], [80, 50, 90, 60]], np.zeros_like(changed))
    # Move both anchors near the center. Historical median anchors converge on
    # the fourth observation, avoiding an eager merge during a brief crossing.
    _frame(memory, 2, [[35, 50, 45, 60], [65, 50, 75, 60]], np.zeros_like(changed))
    _frame(memory, 3, [[48, 50, 58, 60], [57, 50, 67, 60]], np.zeros_like(changed))
    _frame(memory, 4, [[50, 50, 60, 60], [56, 50, 66, 60]], np.zeros_like(changed))
    active = memory.active
    assert len(active) == 1
    assert sorted(active[0].visible_timestamps) == [1.0, 2.0, 3.0, 4.0]
    merged = [event for event in memory.events if event.closed_reason == f"merged_into:{active[0].event_id}"]
    assert len(merged) == 1
