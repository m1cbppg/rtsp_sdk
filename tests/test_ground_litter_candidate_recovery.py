from __future__ import annotations

import pytest

from rtsp_annotator.ground_litter_candidate_recovery import (
    CandidateEvidence, DeferredRecoveryTracker, RecoveryOptions,
    box_overlap_fraction,
)


def candidate(name, box=(10, 10, 30, 30), *, semantic=.6, passed=False):
    return CandidateEvidence(name, box, semantic, -10.0, passed)


def test_repeated_high_semantic_candidate_is_recovered_once():
    tracker = DeferredRecoveryTracker(RecoveryOptions())
    assert not tracker.observe(0.0, [candidate("a")]).temporal_recovered
    assert not tracker.observe(.5, [candidate("b", (11, 10, 31, 30))]).temporal_recovered
    result = tracker.observe(1.0, [candidate("c", (12, 10, 32, 30))])
    assert len(result.temporal_recovered) == 1
    assert len(result.active_recovered) == 1
    assert result.temporal_recovered[0].reason == "temporal_repeat_high_semantic"
    next_result = tracker.observe(1.5, [candidate("d")])
    assert not next_result.temporal_recovered
    assert len(next_result.active_recovered) == 1


def test_transient_candidate_is_not_recovered():
    tracker = DeferredRecoveryTracker(RecoveryOptions())
    tracker.observe(0.0, [candidate("a")])
    result = tracker.observe(.5, [])
    assert not result.temporal_recovered


def test_high_semantic_policy_does_not_recover_low_score_track():
    tracker = DeferredRecoveryTracker(RecoveryOptions(semantic_score_threshold=.45))
    for tick in range(4):
        result = tracker.observe(tick * .5, [candidate(str(tick), semantic=.44)])
    assert not result.temporal_recovered


def test_temporal_any_policy_recovers_low_score_track():
    tracker = DeferredRecoveryTracker(RecoveryOptions(semantic_score_threshold=None))
    for tick in range(3):
        result = tracker.observe(tick * .5, [candidate(str(tick), semantic=.1)])
    assert len(result.temporal_recovered) == 1
    assert result.temporal_recovered[0].reason == "temporal_repeat"


def test_expired_track_cannot_be_resurrected():
    tracker = DeferredRecoveryTracker(RecoveryOptions(ttl_seconds=1.0))
    tracker.observe(0.0, [candidate("a")])
    tracker.observe(.5, [candidate("b")])
    result = tracker.observe(2.0, [candidate("c")])
    assert not result.temporal_recovered
    assert result.active_tracks == 1


def test_same_frame_overlap_is_reported_without_deferred_track():
    tracker = DeferredRecoveryTracker(RecoveryOptions())
    result = tracker.observe(0.0, [
        candidate("filtered", (10, 10, 30, 30)),
        candidate("passed", (12, 12, 32, 32), passed=True),
    ])
    assert len(result.overlap_recovered) == 1
    assert result.overlap_recovered[0].candidate_ids == ("filtered", "passed")
    assert result.active_tracks == 0


def test_capacity_is_hard_bounded_and_reports_eviction():
    tracker = DeferredRecoveryTracker(RecoveryOptions(maximum_tracks=2))
    result = tracker.observe(0.0, [
        candidate("a", (0, 0, 10, 10)),
        candidate("b", (100, 100, 110, 110)),
        candidate("c", (200, 200, 210, 210)),
    ])
    assert result.active_tracks == 2
    assert result.evicted_tracks == 1


def test_one_track_cannot_match_two_detections_in_same_tick():
    tracker = DeferredRecoveryTracker(RecoveryOptions())
    tracker.observe(0.0, [candidate("first")])
    result = tracker.observe(.5, [
        candidate("a", (10, 10, 30, 30)),
        candidate("b", (11, 10, 31, 30)),
    ])
    assert result.active_tracks == 2


@pytest.mark.parametrize("timestamp", [0.0, -1.0])
def test_repeated_or_negative_timestamp_is_rejected(timestamp):
    tracker = DeferredRecoveryTracker(RecoveryOptions())
    if timestamp == 0:
        tracker.observe(0.0, [])
    with pytest.raises(ValueError):
        tracker.observe(timestamp, [])


def test_invalid_candidate_is_rejected():
    tracker = DeferredRecoveryTracker(RecoveryOptions())
    with pytest.raises(ValueError):
        tracker.observe(0.0, [candidate("bad", (10, 10, 10, 20))])


def test_actor_overlap_is_measured_against_candidate_area():
    assert box_overlap_fraction((10, 10, 30, 30), (15, 0, 40, 40)) == .75
    assert box_overlap_fraction((10, 10, 30, 30), (40, 40, 50, 50)) == 0
