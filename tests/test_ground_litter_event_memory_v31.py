import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_ground_litter_event_memory_v31.py"
SPEC = importlib.util.spec_from_file_location("ground_litter_event_memory_v31", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _track(track_id, box):
    return {"track_id": track_id, "median_box": box}


def test_event_grouping_uses_location_without_labels():
    groups = MODULE.group_tracks_into_events([
        _track(1, [100, 100, 112, 110]),
        _track(2, [102, 101, 114, 111]),
        _track(3, [300, 300, 312, 310]),
    ])
    assert [[row["track_id"] for row in group] for group in groups] == [[1, 2], [3]]


def test_event_grouping_rejects_incompatible_scale():
    groups = MODULE.group_tracks_into_events([
        _track(1, [100, 100, 104, 104]),
        _track(2, [98, 98, 122, 122]),
    ])
    assert len(groups) == 2


def test_context_support_marks_touching_large_component():
    support = np.zeros((200, 200), np.uint8)
    valid = np.ones((200, 200), np.uint8) * 255
    support[70:95, 90:100] = 1
    result = MODULE.measure_context_support(support, valid, [100, 100, 110, 110])
    assert result["touching_support_component_px"] == 250
    assert result["touching_external_support_px"] >= 200


def test_context_support_ignores_distant_component_for_touching_metric():
    support = np.zeros((200, 200), np.uint8)
    valid = np.ones((200, 200), np.uint8) * 255
    support[40:60, 40:60] = 1
    result = MODULE.measure_context_support(support, valid, [100, 100, 110, 110])
    assert result["largest_support_component_px"] == 400
    assert result["touching_support_component_px"] == 0
    assert result["touching_external_support_px"] == 0


def test_context_support_does_not_treat_candidate_itself_as_occlusion():
    support = np.zeros((200, 200), np.uint8)
    valid = np.ones((200, 200), np.uint8) * 255
    support[98:113, 98:113] = 1
    result = MODULE.measure_context_support(support, valid, [100, 100, 110, 110])
    assert result["touching_support_component_px"] == 225
    assert result["touching_external_support_px"] < 100


def test_actor_overlap_helper_uses_candidate_coverage():
    actor_script = Path(__file__).resolve().parents[1] / "scripts/probe_ground_litter_event_actor_v31.py"
    spec = importlib.util.spec_from_file_location("ground_litter_event_actor_v31", actor_script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.max_actor_overlap([10, 10, 20, 20], [[0, 0, 15, 20]]) == 0.5
    assert module.max_actor_overlap([10, 10, 20, 20], []) == 0.0
