from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from rtsp_annotator.event_engine import (
    EventEngine,
    GarbageSnapshot,
    NormalizedRect,
    TrackedObject,
)
from rtsp_annotator.event_delivery import WebhookDispatcher
from rtsp_annotator.events import (
    EventDetectionOptions,
    EventRepository,
    EventRoiOptions,
    EventRuleOptions,
    GarbageAnalysisOptions,
    EventRecord,
    WebhookOptions,
)


def event_options(*, garbage: bool = True) -> EventDetectionOptions:
    return EventDetectionOptions(
        enabled=True,
        rois=(
            EventRoiOptions(
                roi_id="shop_entrance",
                polygon=((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)),
                rules=EventRuleOptions(
                    person_dwell_seconds=20,
                    vehicle_dwell_seconds=30,
                    actor_leave_grace_seconds=2,
                    garbage_persistence_seconds=10,
                    minimum_change_area=0.01,
                ),
            ),
        ),
        garbage=GarbageAnalysisOptions(enabled=garbage),
    )


def person(track_id: int = 7) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        class_id=0,
        rectangle=NormalizedRect(0.4, 0.3, 0.1, 0.3),
    )


class EventEngineTests(unittest.TestCase):
    def test_garbage_cannot_be_enabled_without_event_engine(self) -> None:
        with self.assertRaisesRegex(ValueError, "event_detection"):
            EventDetectionOptions(
                enabled=False,
                garbage=GarbageAnalysisOptions(enabled=True),
            ).validate()

    def test_actor_class_sets_cannot_overlap(self) -> None:
        with self.assertRaisesRegex(ValueError, "不能重叠"):
            EventDetectionOptions(
                enabled=False,
                person_classes=(0, 2),
                vehicle_classes=(2,),
            ).validate()

    def test_dwell_emits_once_and_uses_bottom_center(self) -> None:
        engine = EventEngine(stream_id="stream1", options=event_options())

        first = engine.observe_tracks(timestamp=100, objects=[person()])
        before = engine.observe_tracks(timestamp=119, objects=[person()])
        reached = engine.observe_tracks(timestamp=120, objects=[person()])
        repeated = engine.observe_tracks(timestamp=125, objects=[person()])

        self.assertEqual(first.actor_overlays[0].label, "人员停留 0秒")
        self.assertEqual(before.events, [])
        self.assertEqual(reached.events[0].event_type, "zone_dwell")
        self.assertEqual(reached.actor_overlays[0].state, "confirmed")
        self.assertEqual(repeated.events, [])

    def test_track_can_reenter_after_leave_grace(self) -> None:
        engine = EventEngine(stream_id="stream1", options=event_options())
        engine.observe_tracks(timestamp=0, objects=[person()])
        engine.observe_tracks(timestamp=21, objects=[person()])
        engine.observe_tracks(timestamp=24, objects=[])

        reentered = engine.observe_tracks(timestamp=25, objects=[person()])

        self.assertEqual(reentered.actor_overlays[0].label, "人员停留 0秒")

    def test_added_garbage_after_actor_leaves_is_littering(self) -> None:
        engine = EventEngine(stream_id="stream1", options=event_options())
        baseline = GarbageSnapshot(0, 1, 0.05)
        engine.observe_garbage(roi_id="shop_entrance", snapshot=baseline)
        engine.observe_tracks(timestamp=1, objects=[person(88)])
        engine.observe_tracks(timestamp=2, objects=[])
        candidate = GarbageSnapshot(
            4,
            100,
            0.08,
            (NormalizedRect(0.5, 0.6, 0.1, 0.1),),
            0.8,
            "垃圾袋",
            visual_change_ratio=0.02,
        )

        initial = engine.observe_garbage(
            roi_id="shop_entrance",
            snapshot=candidate,
        )
        confirmed = engine.observe_garbage(
            roi_id="shop_entrance",
            snapshot=GarbageSnapshot(
                14,
                200,
                0.08,
                candidate.regions,
                0.8,
                "垃圾袋",
                visual_change_ratio=0.02,
            ),
        )

        self.assertEqual(initial.events, [])
        self.assertIn("疑似新增垃圾", initial.garbage_overlays[0].label)
        self.assertEqual(confirmed.events[0].event_type, "suspected_littering")
        self.assertEqual(confirmed.events[0].actor_track_id, 88)

    def test_removed_garbage_is_cleanup_not_littering(self) -> None:
        engine = EventEngine(stream_id="stream1", options=event_options())
        engine.observe_garbage(
            roi_id="shop_entrance",
            snapshot=GarbageSnapshot(0, 1, 0.10),
        )
        engine.observe_tracks(timestamp=1, objects=[person()])
        engine.observe_tracks(timestamp=2, objects=[])
        engine.observe_garbage(
            roi_id="shop_entrance",
            snapshot=GarbageSnapshot(
                4,
                2,
                0.04,
                visual_change_ratio=0.02,
            ),
        )

        result = engine.observe_garbage(
            roi_id="shop_entrance",
            snapshot=GarbageSnapshot(
                14,
                3,
                0.04,
                visual_change_ratio=0.02,
            ),
        )

        self.assertEqual(result.events[0].event_type, "garbage_removed")
        self.assertEqual(result.events[0].message, "垃圾已清理")

    def test_added_garbage_without_actor_is_unattended(self) -> None:
        engine = EventEngine(stream_id="stream1", options=event_options())
        engine.observe_garbage(
            roi_id="shop_entrance",
            snapshot=GarbageSnapshot(0, 1, 0.01),
        )
        engine.observe_garbage(
            roi_id="shop_entrance",
            snapshot=GarbageSnapshot(
                4,
                2,
                0.03,
                visual_change_ratio=0.02,
            ),
        )

        result = engine.observe_garbage(
            roi_id="shop_entrance",
            snapshot=GarbageSnapshot(
                14,
                3,
                0.03,
                visual_change_ratio=0.02,
            ),
        )

        self.assertEqual(result.events[0].event_type, "unattended_garbage")

    def test_far_actor_is_not_blamed_for_new_garbage(self) -> None:
        engine = EventEngine(stream_id="stream1", options=event_options())
        engine.observe_garbage(
            roi_id="shop_entrance",
            snapshot=GarbageSnapshot(0, 1, 0.01),
        )
        far_actor = TrackedObject(
            track_id=5,
            class_id=0,
            rectangle=NormalizedRect(0.1, 0.1, 0.1, 0.2),
        )
        engine.observe_tracks(timestamp=1, objects=[far_actor])
        engine.observe_tracks(timestamp=2, objects=[])
        region = (NormalizedRect(0.75, 0.75, 0.05, 0.05),)
        engine.observe_garbage(
            roi_id="shop_entrance",
            snapshot=GarbageSnapshot(
                4,
                2,
                0.03,
                region,
                visual_change_ratio=0.02,
            ),
        )

        result = engine.observe_garbage(
            roi_id="shop_entrance",
            snapshot=GarbageSnapshot(
                14,
                3,
                0.03,
                region,
                visual_change_ratio=0.02,
            ),
        )

        self.assertEqual(result.events[0].event_type, "unattended_garbage")
        self.assertIsNone(result.events[0].actor_track_id)

    def test_semantic_flicker_without_pixel_change_is_ignored(self) -> None:
        engine = EventEngine(stream_id="stream1", options=event_options())
        engine.observe_garbage(
            roi_id="shop_entrance",
            snapshot=GarbageSnapshot(0, 1, 0.01),
        )

        appeared = engine.observe_garbage(
            roi_id="shop_entrance",
            snapshot=GarbageSnapshot(
                4,
                2,
                0.04,
                (NormalizedRect(0.4, 0.4, 0.1, 0.1),),
                0.9,
                "plastic bottle",
                visual_change_ratio=0.0,
            ),
        )

        self.assertEqual(appeared.events, [])
        self.assertEqual(appeared.garbage_overlays, [])


class EventRepositoryTests(unittest.TestCase):
    def test_append_list_get_and_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = EventRepository(Path(directory))
            engine = EventEngine(
                stream_id="stream1",
                options=event_options(garbage=False),
                repository=repository,
            )
            engine.observe_tracks(timestamp=0, objects=[person()])
            emitted = engine.observe_tracks(timestamp=20, objects=[person()])
            event_id = emitted.events[0].event_id

            listed = repository.list(stream_id="stream1")
            fetched = repository.get(event_id)
            reviewed = repository.review(event_id, "confirmed")

        self.assertEqual(len(listed), 1)
        self.assertEqual(fetched["event_id"], event_id)
        self.assertEqual(reviewed["status"], "confirmed")


class WebhookDispatcherTests(unittest.TestCase):
    def test_delivery_runs_outside_caller_thread(self) -> None:
        delivered = threading.Event()
        sender_threads: list[str] = []

        def sender(
            _url: str,
            _payload: dict[str, object],
            _timeout: float,
        ) -> None:
            sender_threads.append(threading.current_thread().name)
            delivered.set()

        dispatcher = WebhookDispatcher(
            WebhookOptions(url="http://alarm.local/event"),
            sender=sender,
        )
        event = EventRecord.create(
            stream_id="stream1",
            event_type="zone_dwell",
            roi_id="entrance",
            message="人员区域停留",
        )

        queued = dispatcher.enqueue(event)
        completed = delivered.wait(timeout=1)
        dispatcher.shutdown()

        self.assertTrue(queued)
        self.assertTrue(completed)
        self.assertEqual(sender_threads, ["event-webhook"])

    def test_repository_write_is_also_off_caller_thread(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = EventRepository(Path(directory))
            dispatcher = WebhookDispatcher(
                WebhookOptions(),
                repository=repository,
            )
            event = EventRecord.create(
                stream_id="stream1",
                event_type="zone_dwell",
                roi_id="entrance",
                message="人员区域停留",
            )

            self.assertTrue(dispatcher.enqueue(event))
            for _attempt in range(100):
                if repository.list(stream_id="stream1"):
                    break
                threading.Event().wait(0.01)
            dispatcher.shutdown()

            self.assertEqual(repository.get(event.event_id)["event_id"], event.event_id)


if __name__ == "__main__":
    unittest.main()
