from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from rtsp_annotator.event_evidence import EventEvidenceWriter
from rtsp_annotator.events import EventRecord, EventRepository


class EventEvidenceWriterTests(unittest.TestCase):
    def test_snapshot_is_atomic_and_updates_event_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = EventRepository(Path(directory))
            event = EventRecord.create(
                stream_id="stream1",
                event_type="suspected_littering",
                roi_id="door",
                message="疑似乱丢垃圾",
            )
            repository.append(event)
            frame = np.zeros((90, 160, 3), dtype=np.uint8)
            frame[20:60, 40:100] = 255

            EventEvidenceWriter(repository).attach_snapshots([event], frame)

            stored = repository.get(event.event_id)
            snapshot = Path(str(stored["snapshot_path"]))
            self.assertTrue(snapshot.is_file())
            self.assertEqual(snapshot.read_bytes()[:2], b"\xff\xd8")
            self.assertEqual(
                repository.media_path(event.event_id, "snapshot_path"),
                snapshot,
            )


if __name__ == "__main__":
    unittest.main()
