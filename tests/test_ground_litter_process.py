"""Unit tests for the lossy ground-litter side process and its client."""

from __future__ import annotations

import queue
import unittest
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np

from rtsp_annotator.event_engine import NormalizedRect
from rtsp_annotator.ground_litter_detection import (
    GroundLitterCandidate,
    GroundLitterDetectionOptions,
    GroundLitterResultCache,
    GroundLitterSnapshot,
    GroundLitterZone,
)
from rtsp_annotator.ground_litter_process import (
    GroundLitterProcessClient,
    GroundLitterProcessConfig,
    _put_latest,
    _run_ground_litter_process,
)


def options(enabled: bool = True, **overrides) -> GroundLitterDetectionOptions:
    values = {
        "enabled": enabled,
        "zones": (
            GroundLitterZone(
                region_id="z1",
                polygon=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
                minimum_short_side_px=4,
                minimum_box_area_px=16,
            ),
        ),
        "tile_size_px": 160,
        "minimum_hits": 1,
        "analysis_fps": 1.0,
    }
    values.update(overrides)
    return GroundLitterDetectionOptions(**values)


class FakeProcess:
    def __init__(self, alive: bool = True) -> None:
        self._alive = alive
        self.terminated = False
        self.joined: list[float] = []
        self.started = False

    def start(self) -> None:
        self.started = True

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        self.joined.append(float(timeout or 0.0))
        if timeout:
            # The real process exits after receiving the sentinel.
            self._alive = False

    def terminate(self) -> None:
        self.terminated = True
        self._alive = False


class FakeQueue(queue.Queue):
    """Thread queue with the multiprocessing ``close`` API used on shutdown."""

    def __init__(self, maxsize: int = 0) -> None:
        super().__init__(maxsize=maxsize)
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def cancel_join_thread(self) -> None:
        return None


class FakeContext:
    def __init__(self, process: FakeProcess) -> None:
        self._process = process

    def Queue(self, maxsize: int = 0) -> FakeQueue:
        return FakeQueue(maxsize=maxsize)

    def Process(self, **_kwargs) -> FakeProcess:
        return self._process


class PutLatestTests(unittest.TestCase):
    def test_newest_item_wins_when_the_queue_is_full(self) -> None:
        target: queue.Queue = queue.Queue(maxsize=1)
        _put_latest(target, (0, "old"))
        _put_latest(target, (0, "new"))
        self.assertEqual(target.get_nowait(), (0, "new"))

    def test_empty_queue_is_filled_directly(self) -> None:
        target: queue.Queue = queue.Queue(maxsize=2)
        _put_latest(target, (1, "a"))
        self.assertEqual(target.get_nowait(), (1, "a"))


@dataclass
class FakeDetector:
    candidates_seen: list[int] = None

    def __init__(self) -> None:
        self.has_actor_model = False
        self.calls: list[dict] = []

    def actor_boxes(self, frame, options_):  # pragma: no cover - unused
        return []

    def candidates(
        self,
        frame,
        options_,
        *,
        masks,
        tiles,
        night=False,
        actors=(),
    ):
        self.calls.append(
            {
                "shape": frame.shape,
                "tiles": list(tiles),
                "night": night,
                "actors": [list(item) for item in actors],
            }
        )
        height, width = frame.shape[:2]
        candidates = [
            GroundLitterCandidate(
                rectangle=NormalizedRect(0.1, 0.1, 0.02, 0.02),
                confidence=0.5,
                class_name="Plastic",
                region_id="z1",
            )
        ]
        stats = {
            "raw_candidates": 1,
            "rejected_roi": 2,
            "rejected_actor": 3,
        }
        del height, width
        return candidates, stats


class RunProcessTests(unittest.TestCase):
    def _run(self, items, options_by_pad):
        input_queue: queue.Queue = queue.Queue()
        output_queue: queue.Queue = queue.Queue(maxsize=16)
        for item in items:
            input_queue.put(item)
        input_queue.put(None)
        detector = FakeDetector()
        config = GroundLitterProcessConfig(
            model_path=None,
            device="cpu",
            half=False,
            actor_model_path=None,
            options_by_pad=options_by_pad,
        )
        with patch(
            "rtsp_annotator.ground_litter_process."
            "UltralyticsGroundLitterDetector",
            return_value=detector,
        ):
            _run_ground_litter_process(config, input_queue, output_queue)
        results = []
        while not output_queue.empty():
            results.append(output_queue.get_nowait())
        return detector, results

    def test_frame_is_analyzed_and_snapshot_published(self) -> None:
        frame = np.zeros((200, 200, 3), np.uint8)
        detector, results = self._run(
            [(0, frame, 5.0, False, [(0.5, 0.5, 0.25, 0.25)])],
            {0: options()},
        )
        self.assertTrue(detector.calls)
        self.assertEqual(detector.calls[0]["shape"], (200, 200, 3))
        self.assertEqual(
            detector.calls[0]["actors"],
            [[100.0, 100.0, 150.0, 150.0]],
        )
        snapshots = [item for _pad, item in results if item.state == "running"]
        self.assertEqual(len(snapshots), 1)
        snapshot = snapshots[0]
        self.assertEqual(snapshot.count, 1)
        self.assertEqual(snapshot.analyzed_frames, 1)
        self.assertEqual(snapshot.raw_candidates, 1)
        self.assertEqual(snapshot.rejected_roi, 2)
        self.assertEqual(snapshot.rejected_actor, 3)
        self.assertGreaterEqual(snapshot.tile_count, 1)
        self.assertIsNotNone(snapshot.updated_at)

    def test_starting_snapshot_is_emitted_for_each_pad(self) -> None:
        _detector, results = self._run([], {0: options(), 1: options(False)})
        states = {pad: item.state for pad, item in results}
        self.assertEqual(states[0], "starting")
        self.assertEqual(states[1], "starting")

    def test_night_flag_and_resolution_change_are_forwarded(self) -> None:
        frame = np.zeros((200, 200, 3), np.uint8)
        other = np.zeros((300, 320, 3), np.uint8)
        detector, _results = self._run(
            [
                (0, frame, 1.0, True, ()),
                (0, other, 2.0, False, ()),
            ],
            {0: options()},
        )
        self.assertTrue(detector.calls[0]["night"])
        self.assertFalse(detector.calls[1]["night"])
        self.assertNotEqual(
            detector.calls[0]["tiles"],
            detector.calls[1]["tiles"],
        )

    def test_unknown_pad_is_ignored(self) -> None:
        frame = np.zeros((120, 120, 3), np.uint8)
        detector, results = self._run(
            [(7, frame, 1.0, False, ())],
            {0: options()},
        )
        self.assertEqual(detector.calls, [])
        self.assertEqual(
            [item.state for _pad, item in results],
            ["starting"],
        )

    def test_frame_that_breaks_the_detector_reports_error_not_crash(self) -> None:
        class ExplodingDetector(FakeDetector):
            def candidates(self, *_args, **_kwargs):
                raise RuntimeError("boom")

        input_queue: queue.Queue = queue.Queue()
        output_queue: queue.Queue = queue.Queue(maxsize=16)
        input_queue.put((0, np.zeros((80, 80, 3), np.uint8), 1.0, False, ()))
        input_queue.put(None)
        config = GroundLitterProcessConfig(
            model_path=None,
            device="cpu",
            half=False,
            actor_model_path=None,
            options_by_pad={0: options()},
        )
        with patch(
            "rtsp_annotator.ground_litter_process."
            "UltralyticsGroundLitterDetector",
            return_value=ExplodingDetector(),
        ):
            _run_ground_litter_process(config, input_queue, output_queue)
        states = [item.state for _pad, item in _drain(output_queue)]
        self.assertIn("error", states)


def _drain(target: queue.Queue) -> list:
    items = []
    while not target.empty():
        items.append(target.get_nowait())
    return items


class ClientTests(unittest.TestCase):
    def _client(self, options_by_pad, *, alive: bool = True):
        cache = GroundLitterResultCache()
        process = FakeProcess(alive)
        context = FakeContext(process)
        config = GroundLitterProcessConfig(
            model_path=None,
            device="cpu",
            half=False,
            actor_model_path=None,
            options_by_pad=options_by_pad,
        )
        with patch(
            "rtsp_annotator.ground_litter_process.multiprocessing."
            "get_context",
            return_value=context,
        ):
            client = GroundLitterProcessClient(config, cache)
        return client, cache, process

    def test_accepts_rate_limits_each_pad_separately(self) -> None:
        client, _cache, _process = self._client({0: options(), 1: options()})
        with patch(
            "rtsp_annotator.ground_litter_process.time.monotonic",
            return_value=100.0,
        ):
            self.assertTrue(client.accepts(0))
            self.assertFalse(client.accepts(0))
            self.assertTrue(client.accepts(1))
        with patch(
            "rtsp_annotator.ground_litter_process.time.monotonic",
            return_value=101.0,
        ):
            self.assertTrue(client.accepts(0))
        client.shutdown()

    def test_disabled_or_unknown_pad_is_never_accepted(self) -> None:
        client, _cache, _process = self._client(
            {0: options(False), 1: options()}
        )
        with patch(
            "rtsp_annotator.ground_litter_process.time.monotonic",
            return_value=50.0,
        ):
            self.assertFalse(client.accepts(0))
            self.assertFalse(client.accepts(9))
            self.assertTrue(client.accepts(1))
        self.assertEqual(client.enabled_pads, (1,))
        client.shutdown()

    def test_dead_process_marks_error_and_rejects_frames(self) -> None:
        client, cache, _process = self._client({0: options()}, alive=False)
        with patch(
            "rtsp_annotator.ground_litter_process.time.monotonic",
            return_value=10.0,
        ):
            self.assertFalse(client.accepts(0))
        snapshot = cache.snapshot(0)
        self.assertEqual(snapshot.state, "error")
        self.assertIn("已退出", snapshot.message)
        client.shutdown()

    def test_submit_and_drain_round_trip(self) -> None:
        client, cache, _process = self._client({0: options()})
        frame = np.zeros((40, 40, 3), np.uint8)
        self.assertTrue(
            client.submit(0, frame, timestamp=1.0, night=True, actors=())
        )
        self.assertFalse(client.submit(5, frame, timestamp=1.0))
        client._output_queue.put(
            (0, GroundLitterSnapshot(state="running", result_version=3))
        )
        client.drain_results()
        self.assertEqual(cache.snapshot(0).result_version, 3)
        client.shutdown()

    def test_submit_fails_when_the_input_queue_is_full(self) -> None:
        client, _cache, _process = self._client({0: options()})
        frame = np.zeros((40, 40, 3), np.uint8)
        for _ in range(8):
            client.submit(0, frame, timestamp=1.0)
        self.assertFalse(client.submit(0, frame, timestamp=2.0))
        client.shutdown()

    def test_shutdown_sends_sentinel_and_closes_queues(self) -> None:
        client, _cache, process = self._client({0: options()})
        client.shutdown()
        self.assertEqual(process.joined, [5.0])
        self.assertFalse(process.terminated)


if __name__ == "__main__":
    unittest.main()
