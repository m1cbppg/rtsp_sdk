from __future__ import annotations

import queue
import unittest

import cv2
import numpy as np

from rtsp_annotator.event_engine import NormalizedRect
from rtsp_annotator.vessel_detection import (
    VesselCandidate,
    VesselDetectionOptions,
    VesselSnapshot,
    VesselResultCache,
)
from rtsp_annotator.vessel_detection_process import (
    VesselDetectionProcessClient,
    _put_latest,
    _validate_evidence_image,
    _verification_options,
)


class VesselDetectionProcessTests(unittest.TestCase):
    def test_late_inference_from_old_view_is_not_published(self) -> None:
        client = VesselDetectionProcessClient.__new__(VesselDetectionProcessClient)
        client._last_view_generation = {0: 4}
        client._output_queue = queue.Queue()
        client._cache = VesselResultCache()
        client._output_queue.put((0, VesselSnapshot(
            state="running", result_version=9, view_generation=3,
        )))
        client.drain_results()
        self.assertEqual(client._cache.snapshot(0).state, "disabled")
        client._output_queue.put((0, VesselSnapshot(
            state="running", result_version=10, view_generation=4,
        )))
        client.drain_results()
        self.assertEqual(client._cache.snapshot(0).result_version, 10)

    def test_failed_view_event_enqueue_is_retried_on_next_frame(self) -> None:
        class FullQueue:
            def put_nowait(self, _item) -> None:
                raise queue.Full

            def get_nowait(self):
                raise queue.Empty

        client = VesselDetectionProcessClient.__new__(
            VesselDetectionProcessClient
        )
        client._options_by_pad = {0: VesselDetectionOptions()}
        client._last_view_generation = {0: 3}
        client._view_queue = FullQueue()
        client._input_queue = queue.Queue()

        accepted = client.submit(
            0,
            np.zeros((2, 2, 3), dtype=np.uint8),
            timestamp=1.0,
            view_generation=4,
        )

        self.assertFalse(accepted)
        self.assertEqual(client._last_view_generation[0], 3)
        self.assertTrue(client._input_queue.empty())

    def test_put_latest_replaces_stale_item_when_queue_is_full(self) -> None:
        output: queue.Queue = queue.Queue(maxsize=1)
        stale = VesselSnapshot(state="running", result_version=1)
        latest = VesselSnapshot(state="running", result_version=2)
        output.put_nowait((0, stale))

        _put_latest(output, (0, latest))

        _pad, snapshot = output.get_nowait()
        self.assertEqual(snapshot.result_version, 2)

    def test_evidence_validation_decodes_jpeg_and_measures_boat_crop(self) -> None:
        class Detector:
            def detect(self, _frame, _options):
                return [
                    VesselCandidate(
                        rectangle=NormalizedRect(0.3, 0.3, 0.4, 0.3),
                        confidence=0.8,
                        class_id=8,
                    )
                ]

        image = np.zeros((120, 160, 3), dtype=np.uint8)
        image[36:72, 48:112:2] = 255
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)

        result = _validate_evidence_image(
            Detector(),  # type: ignore[arg-type]
            encoded.tobytes(),
            VesselDetectionOptions(),
        )

        self.assertEqual(result.state, "running")
        self.assertEqual(len(result.detections), 1)
        self.assertGreater(result.sharpness_for(1), 0)

    def test_closeup_profile_does_not_mutate_home_view_filters(self) -> None:
        home = VesselDetectionOptions(
            inference_regions=((0.0, 0.4, 1.0, 1.0),),
            roi=((0.0, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)),
            exclude_rois=(
                ((0.9, 0.7), (1.0, 0.7), (1.0, 1.0), (0.9, 1.0)),
            ),
            proposal_roi=(
                (0.0, 0.5),
                (1.0, 0.5),
                (1.0, 0.8),
                (0.0, 0.8),
            ),
            large_box_area_threshold=0.2,
            maximum_box_area=0.8,
            proposal_minimum_motion_ratio=0.1,
        )

        closeup = _verification_options(home)

        self.assertIsNone(closeup.roi)
        self.assertEqual(closeup.exclude_rois, ())
        self.assertEqual(closeup.inference_regions, ((0.0, 0.0, 1.0, 1.0),))
        self.assertEqual(closeup.large_box_area_threshold, 0.2)
        self.assertEqual(closeup.maximum_box_area, 0.8)
        self.assertIsNotNone(closeup.proposal_roi)
        self.assertEqual(closeup.proposal_minimum_motion_ratio, 0.0)
        self.assertIsNotNone(home.roi)
        self.assertEqual(len(home.exclude_rois), 1)
        self.assertEqual(home.inference_regions, ((0.0, 0.4, 1.0, 1.0),))
        self.assertEqual(home.proposal_minimum_motion_ratio, 0.1)


if __name__ == "__main__":
    unittest.main()
