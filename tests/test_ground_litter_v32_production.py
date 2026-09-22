from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from rtsp_annotator.ground_litter_detection import (
    GroundLitterDetectionOptions,
    GroundLitterZone,
)
from rtsp_annotator.ground_litter_v32 import (
    CleanReferenceProfileV32,
    CleanReferenceV32Processor,
    PROFILE_KIND,
    align_profile,
)


def options(**overrides) -> GroundLitterDetectionOptions:
    values = {
        "enabled": True,
        "mode": "clean_reference_v32",
        "profile_id": "camera_01_v32",
        "analysis_fps": 1.0,
        "startup_suppress_seconds": 0.0,
        "normal_stability_samples": 1,
        "zones": (
            GroundLitterZone(
                region_id="walkway",
                polygon=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
                minimum_short_side_px=1,
                minimum_box_area_px=1,
            ),
        ),
    }
    values.update(overrides)
    result = GroundLitterDetectionOptions(**values)
    result.validate()
    return result


def write_profile(root: Path, *, corrupt: bool = False) -> Path:
    directory = root / "test_profile"
    directory.mkdir(parents=True)
    reference = np.full((100, 120, 3), 100, np.uint8)
    valid = np.full((100, 120), 255, np.uint8)
    tolerance = np.zeros((100, 120), np.uint8)
    paths = {
        "reference.png": reference,
        "valid_mask.png": valid,
        "daylight_tolerance.png": tolerance,
    }
    for name, image in paths.items():
        assert cv2.imwrite(str(directory / name), image)
    checksums = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in paths
    }
    if corrupt:
        checksums["reference.png"] = "0" * 64
    (directory / "profile.json").write_text(
        json.dumps({
            "kind": PROFILE_KIND,
            "profile_id": "test_profile",
            "reference_size": [120, 100],
            "sha256": checksums,
        }),
        encoding="utf-8",
    )
    return directory


def test_clean_reference_options_require_safe_profile_id() -> None:
    with pytest.raises(ValueError, match="profile_id"):
        GroundLitterDetectionOptions(
            enabled=True,
            mode="clean_reference_v32",
            zones=options().zones,
        ).validate()
    with pytest.raises(ValueError, match="profile_id"):
        GroundLitterDetectionOptions(
            enabled=True,
            mode="clean_reference_v32",
            profile_id="../camera",
            zones=options().zones,
        ).validate()


def test_profile_load_verifies_checksums(tmp_path: Path) -> None:
    write_profile(tmp_path)
    loaded = CleanReferenceProfileV32.load(tmp_path, "test_profile")
    assert loaded.reference.shape == (100, 120, 3)

    bad = tmp_path / "bad"
    bad.mkdir()
    source = write_profile(bad, corrupt=True)
    source.rename(bad / "broken")
    metadata = json.loads((bad / "broken" / "profile.json").read_text())
    metadata["profile_id"] = "broken"
    (bad / "broken" / "profile.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="校验失败"):
        CleanReferenceProfileV32.load(bad, "broken")


@pytest.mark.parametrize(
    "profile_id",
    ("camera_01_v32", "camera_01_v32_1080p", "camera_01_v32_afternoon_1080p"),
)
def test_packaged_camera_profile_aligns_to_its_reviewed_reference(
    profile_id: str,
) -> None:
    root = Path(__file__).resolve().parents[1] / "models/litter/profiles"
    profile = CleanReferenceProfileV32.load(root, profile_id)
    reference, valid, tolerance, diagnostics = align_profile(
        profile, profile.reference.copy()
    )
    assert reference.shape == profile.reference.shape
    assert valid.shape == profile.valid.shape
    assert tolerance.shape == profile.tolerance.shape
    assert diagnostics["inliers"] >= 20
    assert diagnostics["reprojection_median_px"] <= 0.1


def test_processor_only_draws_confirmed_supported_events_and_reuses_no_id() -> None:
    frame = np.zeros((40, 40, 3), np.uint8)
    profile = CleanReferenceProfileV32(
        "camera_01_v32",
        Path("."),
        frame.copy(),
        np.full((40, 40), 255, np.uint8),
        np.zeros((40, 40), np.uint8),
        {},
    )
    processor = CleanReferenceV32Processor(options(), profile)
    changed = np.zeros((40, 40), np.uint8)
    changed[5:15, 5:15] = 1
    clean = np.zeros_like(changed)
    candidate = {"box": [5, 5, 15, 15], "anomaly_score": 0.8}
    proposals = [
        *[([candidate], changed) for _ in range(5)],
        *[([], clean) for _ in range(5)],
        *[([candidate], changed) for _ in range(5)],
    ]

    with patch(
        "rtsp_annotator.ground_litter_v32.align_profile",
        return_value=(frame.copy(), np.full((40, 40), 255, np.uint8), clean, {}),
    ), patch(
        "rtsp_annotator.ground_litter_v32.protected_normalize",
        return_value=(frame.copy(), np.full((40, 40), 255, np.uint8), {"state": "NORMAL"}),
    ), patch(
        "rtsp_annotator.ground_litter_v32.propose_v32",
        side_effect=proposals,
    ):
        snapshots = [
            processor.update(frame, timestamp=float(timestamp))
            for timestamp in range(1, 16)
        ]

    assert [snapshot.count for snapshot in snapshots[:4]] == [0, 0, 0, 0]
    assert snapshots[4].count == 1
    first_id = snapshots[4].detections[0].object_id
    assert all(snapshot.count == 0 for snapshot in snapshots[5:10])
    assert [event.state for event in processor.memory.events[:1]] == ["CLEARED"]
    assert snapshots[-1].count == 1
    assert snapshots[-1].detections[0].object_id != first_id


def test_environment_change_abstains_and_never_draws() -> None:
    frame = np.zeros((40, 40, 3), np.uint8)
    profile = CleanReferenceProfileV32(
        "camera_01_v32", Path("."), frame, np.full((40, 40), 255, np.uint8),
        np.zeros((40, 40), np.uint8), {},
    )
    processor = CleanReferenceV32Processor(options(confirm_visible_seconds=1), profile)
    with patch(
        "rtsp_annotator.ground_litter_v32.align_profile",
        return_value=(frame, profile.valid, profile.tolerance, {}),
    ), patch(
        "rtsp_annotator.ground_litter_v32.protected_normalize",
        return_value=(frame, profile.valid, {"state": "ENVIRONMENT_CHANGE"}),
    ), patch(
        "rtsp_annotator.ground_litter_v32.propose_v32",
        return_value=([{"box": [5, 5, 15, 15], "anomaly_score": 1.0}], profile.valid),
    ):
        snapshot = processor.update(frame, timestamp=1.0)
    assert snapshot.state == "abstaining"
    assert snapshot.count == 0


def test_startup_suppression_never_builds_event_memory() -> None:
    frame = np.zeros((40, 40, 3), np.uint8)
    profile = CleanReferenceProfileV32(
        "camera_01_v32", Path("."), frame,
        np.full((40, 40), 255, np.uint8), np.zeros((40, 40), np.uint8), {},
    )
    processor = CleanReferenceV32Processor(
        options(startup_suppress_seconds=10, normal_stability_samples=2), profile
    )
    with patch(
        "rtsp_annotator.ground_litter_v32.align_profile",
        return_value=(frame, profile.valid, profile.tolerance, {}),
    ), patch(
        "rtsp_annotator.ground_litter_v32.protected_normalize",
        return_value=(frame, profile.valid, {"state": "NORMAL"}),
    ), patch(
        "rtsp_annotator.ground_litter_v32.propose_v32"
    ) as propose:
        first = processor.update(frame, timestamp=100.0)
        second = processor.update(frame, timestamp=109.9)
    assert first.state == "warming_up"
    assert second.state == "warming_up"
    assert processor.memory.events == []
    propose.assert_not_called()


def test_global_light_change_resets_events_and_requires_stable_normal_samples() -> None:
    frame = np.zeros((40, 40, 3), np.uint8)
    profile = CleanReferenceProfileV32(
        "camera_01_v32", Path("."), frame,
        np.full((40, 40), 255, np.uint8), np.zeros((40, 40), np.uint8), {},
    )
    processor = CleanReferenceV32Processor(
        options(confirm_visible_seconds=1, normal_stability_samples=2), profile
    )
    changed = np.zeros((40, 40), np.uint8)
    changed[5:15, 5:15] = 1
    candidate = {"box": [5, 5, 15, 15], "anomaly_score": 1.0}
    environments = [
        {"state": "NORMAL"},
        {"state": "NORMAL"},
        {"state": "GLOBAL_LIGHT_CHANGE"},
        {"state": "NORMAL"},
        {"state": "NORMAL"},
    ]
    with patch(
        "rtsp_annotator.ground_litter_v32.align_profile",
        return_value=(frame, profile.valid, profile.tolerance, {}),
    ), patch(
        "rtsp_annotator.ground_litter_v32.protected_normalize",
        side_effect=[(frame, profile.valid, item) for item in environments],
    ), patch(
        "rtsp_annotator.ground_litter_v32.propose_v32",
        return_value=([candidate], changed),
    ):
        snapshots = [
            processor.update(frame, timestamp=float(index))
            for index in range(1, 6)
        ]
    assert [item.state for item in snapshots] == [
        "abstaining", "running", "abstaining", "abstaining", "running"
    ]
    assert snapshots[1].count == 1
    assert snapshots[2].count == 0
    assert snapshots[3].count == 0
    assert snapshots[4].count == 1
    assert snapshots[4].detections[0].object_id == 1


def test_processor_upscales_1080p_analysis_and_actor_boxes_to_profile_space() -> None:
    reference = np.zeros((40, 80, 3), np.uint8)
    profile = CleanReferenceProfileV32(
        "camera_01_v32", Path("."), reference,
        np.full((40, 80), 255, np.uint8), np.zeros((40, 80), np.uint8), {},
    )
    processor = CleanReferenceV32Processor(options(), profile)
    processor.memory.update = MagicMock()
    input_frame = np.zeros((20, 40, 3), np.uint8)
    with patch(
        "rtsp_annotator.ground_litter_v32.align_profile",
        return_value=(reference, profile.valid, profile.tolerance, {}),
    ) as align, patch(
        "rtsp_annotator.ground_litter_v32.protected_normalize",
        return_value=(reference, profile.valid, {"state": "NORMAL"}),
    ), patch(
        "rtsp_annotator.ground_litter_v32.propose_v32",
        return_value=([], np.zeros((40, 80), np.uint8)),
    ):
        processor.update(
            input_frame, timestamp=1.0, actors=[[1.0, 2.0, 3.0, 4.0]]
        )
    assert align.call_args.args[1].shape == (40, 80, 3)
    assert processor.memory.update.call_args.kwargs["actors"] == [
        [2.0, 4.0, 6.0, 8.0]
    ]


def test_reviewed_zone_exclusion_filters_change_component() -> None:
    frame = np.zeros((100, 100, 3), np.uint8)
    zone = GroundLitterZone(
        region_id="walkway",
        polygon=((0, 0), (1, 0), (1, 1), (0, 1)),
        exclude_zones=(((0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)),),
        minimum_short_side_px=1,
        minimum_box_area_px=1,
    )
    processor = CleanReferenceV32Processor(
        options(zones=(zone,)),
        CleanReferenceProfileV32(
            "camera_01_v32", Path("."), frame,
            np.full((100, 100), 255, np.uint8),
            np.zeros((100, 100), np.uint8), {},
        ),
    )
    rows = processor._assign_regions([
        {"box": [45, 45, 55, 55]},
        {"box": [10, 10, 20, 20]},
    ], 100, 100)
    assert rows == [{"box": [10, 10, 20, 20], "region_id": "walkway"}]
