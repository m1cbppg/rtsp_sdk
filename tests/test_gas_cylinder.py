from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rtsp_annotator.event_engine import NormalizedRect
from rtsp_annotator.gas_cylinder import (
    GasCylinderCameraProfile,
    GasCylinderCandidate,
    GasCylinderCoordinator,
    GasCylinderOptions,
    GasCylinderResultCache,
    GasPromptProfile,
    StaticSceneMonitor,
    build_temporal_consensus,
    deduplicate_candidates,
)


def candidate(
    left: float,
    top: float,
    width: float = 0.08,
    height: float = 0.20,
    confidence: float = 0.8,
) -> GasCylinderCandidate:
    return GasCylinderCandidate(
        rectangle=NormalizedRect(left, top, width, height),
        confidence=confidence,
    )


def camera_profile() -> GasCylinderCameraProfile:
    return GasCylinderCameraProfile(
        profile_id="camera_01_ir",
        reference_image=Path("unused.jpg"),
        reference_width=1280,
        reference_height=720,
        roi=((0, 0), (1, 0), (1, 1), (0, 1)),
        exclude_rois=(),
        prompts=(
            GasPromptProfile(
                prompt_id="regular",
                boxes=((0.1, 0.1, 0.2, 0.3),),
            ),
        ),
    )


class FakeDetector:
    def __init__(self, results: list[list[GasCylinderCandidate]]) -> None:
        self.results = results
        self.calls = 0

    def detect(self, _frame: np.ndarray) -> list[GasCylinderCandidate]:
        result = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        return result


class GasCylinderTests(unittest.TestCase):
    def test_profile_loads_normalized_visual_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "reference.jpg").touch()
            (root / "camera_01_ir.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "profile_id": "camera_01_ir",
                        "reference_image": "reference.jpg",
                        "reference_size": [1280, 720],
                        "roi": [[0, 0], [1, 0], [1, 1], [0, 1]],
                        "exclude_rois": [
                            [[0.8, 0], [1, 0], [1, 0.2], [0.8, 0.2]]
                        ],
                        "prompts": [
                            {
                                "id": "regular",
                                "boxes": [[0.1, 0.2, 0.3, 0.6]],
                                "minimum_confidence": 0.03,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            profile = GasCylinderCameraProfile.load(root, "camera_01_ir")

        self.assertEqual(profile.reference_width, 1280)
        self.assertEqual(profile.prompts[0].prompt_id, "regular")
        self.assertEqual(profile.prompts[0].boxes[0], (0.1, 0.2, 0.3, 0.6))
        self.assertEqual(len(profile.exclude_rois), 1)

    def test_consensus_requires_repeated_one_to_one_detections(self) -> None:
        samples = [
            [candidate(0.10, 0.20), candidate(0.50, 0.20)],
            [candidate(0.105, 0.20), candidate(0.505, 0.20)],
            [candidate(0.11, 0.20), candidate(0.51, 0.20)],
            [candidate(0.115, 0.20)],
            [candidate(0.80, 0.20, confidence=0.2)],
        ]

        stable = build_temporal_consensus(
            samples,
            minimum_confirmations=3,
            match_iou=0.3,
            nms_iou=0.5,
        )

        self.assertEqual(len(stable), 2)
        self.assertEqual(sorted(item.support for item in stable), [3, 4])

    def test_high_iou_duplicate_is_removed_without_merging_neighbours(self) -> None:
        detections = [
            candidate(0.15, 0.15, 0.08, 0.18, confidence=0.9),
            candidate(0.152, 0.152, 0.08, 0.18, confidence=0.8),
            candidate(0.32, 0.16, 0.08, 0.18, confidence=0.7),
        ]

        selected = deduplicate_candidates(detections, iou_threshold=0.5)

        self.assertEqual(len(selected), 2)
        self.assertTrue(all(item.rectangle.width < 0.1 for item in selected))

    def test_cache_keeps_ids_when_result_refreshes(self) -> None:
        cache = GasCylinderResultCache()
        first = cache.publish(
            0,
            [candidate(0.1, 0.2), candidate(0.5, 0.2)],
            timestamp=1,
            inference_ms=20,
        )
        second = cache.publish(
            0,
            [candidate(0.105, 0.2), candidate(0.505, 0.2)],
            timestamp=2,
            inference_ms=21,
        )

        self.assertEqual(
            [item.object_id for item in first.detections],
            [item.object_id for item in second.detections],
        )
        self.assertEqual(second.result_version, 2)

    def test_coordinator_samples_only_at_configured_times_then_caches(self) -> None:
        options = GasCylinderOptions(
            enabled=True,
            sample_count=3,
            minimum_confirmations=2,
            sample_interval_seconds=1,
            forced_refresh_seconds=30,
        )
        detector = FakeDetector(
            [
                [candidate(0.1, 0.2)],
                [candidate(0.102, 0.2)],
                [candidate(0.104, 0.2)],
            ]
        )
        cache = GasCylinderResultCache()
        coordinator = GasCylinderCoordinator(
            pad_index=0,
            options=options,
            profile=camera_profile(),
            detector=detector,
            cache=cache,
        )
        frame = np.zeros((90, 160, 3), dtype=np.uint8)

        coordinator.process(frame, timestamp=0)
        coordinator.process(frame, timestamp=0.5)
        coordinator.process(frame, timestamp=1)
        coordinator.process(frame, timestamp=2)
        for timestamp in (3, 4, 5, 10, 20):
            coordinator.process(frame, timestamp=timestamp)

        snapshot = cache.snapshot(0)
        self.assertEqual(detector.calls, 3)
        self.assertEqual(snapshot.state, "stable")
        self.assertEqual(snapshot.count, 1)
        self.assertEqual(snapshot.detections[0].support, 3)

    def test_scene_monitor_ignores_global_brightness_change(self) -> None:
        monitor = StaticSceneMonitor(camera_profile())
        dark = np.full((90, 160, 3), 20, dtype=np.uint8)
        bright = np.full((90, 160, 3), 120, dtype=np.uint8)
        monitor.commit(dark)

        observation = monitor.observe(bright)

        self.assertEqual(observation.change_ratio, 0)


if __name__ == "__main__":
    unittest.main()
