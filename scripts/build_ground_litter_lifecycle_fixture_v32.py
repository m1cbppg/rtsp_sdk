#!/usr/bin/env python3
"""Build and validate a same-camera semi-synthetic V3.2 lifecycle fixture."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ground_litter_event_state_v32 import OnlineEventMemory  # noqa: E402
from rtsp_annotator.ground_litter_detection import (  # noqa: E402
    UltralyticsGroundLitterDetector,
)
from run_ground_litter_clean_temporal_poc import (  # noqa: E402
    T01_REGION,
    video_metadata,
    yolo_options,
)
from run_ground_litter_online_replay_v31 import load_json  # noqa: E402
from run_ground_litter_online_replay_v32 import replay_video  # noqa: E402
from run_ground_litter_prior_region_v3 import scan_video  # noqa: E402


FIXTURE_FPS = 5
SEGMENTS = [
    {
        "phase": "clean_before_first_deposit",
        "source": "clean_reference",
        "source_start": 36.0,
        "source_end": 46.0,
        "expected_state": "CLEAN",
    },
    {
        "phase": "first_litter_with_real_occlusion",
        "source": "positive",
        "source_start": 72.0,
        "source_end": 92.0,
        "expected_state": "LITTER_VISIBLE_OR_OCCLUDED",
    },
    {
        "phase": "clean_after_removal",
        "source": "clean_reference",
        "source_start": 36.0,
        "source_end": 51.0,
        "expected_state": "CLEAN",
    },
    {
        "phase": "second_litter_same_location",
        "source": "positive",
        "source_start": 80.0,
        "source_end": 100.0,
        "expected_state": "LITTER_VISIBLE",
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixture_segments() -> list[dict[str, Any]]:
    cursor = 0.0
    rows = []
    for source in SEGMENTS:
        duration = float(source["source_end"] - source["source_start"])
        row = dict(source)
        row["fixture_start"] = cursor
        cursor += duration
        row["fixture_end"] = cursor
        rows.append(row)
    return rows


def build_fixture_video(clean_source: Path, positive_source: Path, output: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to build the lifecycle fixture")
    filters = []
    labels = []
    for index, row in enumerate(SEGMENTS):
        label = f"v{index}"
        input_index = 0 if row["source"] == "clean_reference" else 1
        filters.append(
            f"[{input_index}:v]trim=start={row['source_start']}:end={row['source_end']},"
            f"setpts=PTS-STARTPTS[{label}]"
        )
        labels.append(f"[{label}]")
    filters.append(
        "".join(labels)
        + f"concat=n={len(labels)}:v=1:a=0,fps={FIXTURE_FPS},format=yuv420p[outv]"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(clean_source),
            "-i",
            str(positive_source),
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[outv]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )


def is_primary(event: dict[str, Any]) -> bool:
    return not str(event.get("closed_reason") or "").startswith("merged_into:")


def center_in_t01(event: dict[str, Any]) -> bool:
    left, top, right, bottom = event["anchor_box"]
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    return (
        T01_REGION[0] <= center_x <= T01_REGION[2]
        and T01_REGION[1] <= center_y <= T01_REGION[3]
    )


def validate_lifecycle(events: list[dict[str, Any]]) -> dict[str, Any]:
    segments = fixture_segments()
    first_litter = segments[1]
    clean_after = segments[2]
    second_litter = segments[3]
    target_events = sorted(
        (
            event for event in events
            if is_primary(event)
            and center_in_t01(event)
            and event.get("confirmed_at") is not None
        ),
        key=lambda event: (float(event["first_seen"]), int(event["event_id"])),
    )
    first = next((
        event for event in target_events
        if first_litter["fixture_start"] <= event["first_seen"] < clean_after["fixture_start"]
    ), None)
    second = next((
        event for event in target_events
        if event["first_seen"] >= second_litter["fixture_start"]
    ), None)
    checks = {
        "exactly_two_target_confirmed_primary_events": len(target_events) == 2,
        "first_event_confirmed": first is not None,
        "first_event_observed_real_occlusion": (
            first is not None
            and any(row["state"] == "OCCLUDED" for row in first["state_history"])
        ),
        "first_event_cleared_during_clean_segment": (
            first is not None
            and first.get("closed_reason") == "clean_confirmed"
            and clean_after["fixture_start"]
            <= float(first["closed_at"])
            <= clean_after["fixture_end"]
        ),
        "second_event_confirmed": second is not None,
        "second_event_has_new_id": (
            first is not None and second is not None
            and first["event_id"] != second["event_id"]
        ),
        "second_event_started_after_first_cleared": (
            first is not None and second is not None
            and first.get("closed_at") is not None
            and second["first_seen"] > first["closed_at"]
        ),
    }
    passed = all(checks.values())
    return {
        "kind": "ground_litter_lifecycle_fixture_v32_validation",
        "passed": passed,
        "baseline_decision": {
            "status": (
                "ACCEPTED_AS_CURRENT_OFFLINE_BASELINE"
                if passed else "NOT_ACCEPTED"
            ),
            "production_deployed": False,
            "natural_field_lifecycle_validated": False,
        },
        "checks": checks,
        "first_event": first,
        "second_event": second,
        "target_confirmed_event_ids": [event["event_id"] for event in target_events],
    }


def render_report(manifest: dict[str, Any], validation: dict[str, Any]) -> str:
    rows = "\n".join(
        "| {phase} | {source} | {source_start:.0f}–{source_end:.0f}s | "
        "{fixture_start:.0f}–{fixture_end:.0f}s | {expected_state} |".format(**row)
        for row in manifest["segments"]
    )
    checks = "\n".join(
        f"- {'PASS' if passed else 'FAIL'}：{name}"
        for name, passed in validation["checks"].items()
    )
    first = validation.get("first_event") or {}
    second = validation.get("second_event") or {}
    return f"""# Ground Litter V3.2 半合成生命周期验证

## 结论

**{'PASS' if validation['passed'] else 'FAIL'}**

本素材由同一摄像头真实画面按时间重新排列，用于验证状态机生命周期，不用于估计真实准确率或召回率。

基线决策：`{validation['baseline_decision']['status']}`。V3.2 允许作为当前离线 PoC 和后续集成基线；尚未部署生产，仍需通过未来自然发生的现场事件补充验证。

## 素材时间线

| 阶段 | 来源 | 原视频时间 | 合成视频时间 | 预期画面语义 |
|---|---|---:|---:|---|
{rows}

Clean Reference SHA-256：`{manifest['source_videos']['clean_reference']['sha256']}`  
正样本 SHA-256：`{manifest['source_videos']['positive']['sha256']}`  
合成视频 SHA-256：`{manifest['fixture_video']['sha256']}`

## 验收项

{checks}

## 目标事件

- 第一次事件：ID `{first.get('event_id')}`，首次出现 `{first.get('first_seen')}` 秒，确认 `{first.get('confirmed_at')}` 秒，关闭 `{first.get('closed_at')}` 秒，原因 `{first.get('closed_reason')}`。
- 第二次事件：ID `{second.get('event_id')}`，首次出现 `{second.get('first_seen')}` 秒，确认 `{second.get('confirmed_at')}` 秒，当前状态 `{second.get('state')}`。

## 证据边界

- 投放物、遮挡人员和地面画面均来自真实摄像头。
- “清走”和“再次投放”通过重排真实片段构造，不是自然连续发生的现场录像。
- 本测试通过只表示 V3.2 完整视频链路能正确关闭旧事件并在同位置建立新事件。
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-video", type=Path, required=True)
    parser.add_argument("--positive-video", type=Path, required=True)
    parser.add_argument("--v1-output", type=Path, required=True)
    parser.add_argument("--v3-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--litter-model", type=Path, required=True)
    parser.add_argument("--actor-model", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--force-video", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    video = args.output / "ground_litter_lifecycle_fixture_v32.mp4"
    if args.force_video or not video.exists():
        build_fixture_video(args.clean_video, args.positive_video, video)
    metadata = video_metadata(video)
    arrays = np.load(args.v1_output / "reference_arrays.npz")
    reference = arrays["reference"]
    reference_valid = cv2.imread(
        str(args.v3_output / "reference_valid_mask_v3.png"), cv2.IMREAD_GRAYSCALE
    )
    daylight_tolerance = cv2.imread(
        str(args.v3_output / "daylight_bounded_tolerance_mask.png"),
        cv2.IMREAD_GRAYSCALE,
    )
    if reference_valid is None or daylight_tolerance is None:
        raise FileNotFoundError("V3 reference valid or daylight tolerance mask is missing")

    v3_fixture_root = args.output / "v3"
    scan_video(
        "lifecycle_fixture",
        video,
        v3_fixture_root,
        reference,
        reference_valid,
        daylight_tolerance,
        start=0.0,
        duration=float(metadata["duration_seconds"]),
    )
    options = yolo_options(args.litter_model, args.actor_model)
    detector = UltralyticsGroundLitterDetector(
        model_path=args.litter_model,
        actor_model_path=args.actor_model,
        device=args.device,
    )
    online_output = args.output / "online"
    replay_video(
        name="lifecycle_fixture",
        video=video,
        start=0.0,
        v3_directory=v3_fixture_root / "lifecycle_fixture",
        output=online_output,
        reference=reference,
        reference_valid=reference_valid,
        detector=detector,
        detector_options=options,
    )
    events = load_json(online_output / "events.json")
    validation = validate_lifecycle(events)
    manifest = {
        "kind": "ground_litter_lifecycle_fixture_v32",
        "semi_synthetic": True,
        "source_videos": {
            "clean_reference": {
                "path": str(args.clean_video.resolve()),
                "sha256": sha256(args.clean_video),
            },
            "positive": {
                "path": str(args.positive_video.resolve()),
                "sha256": sha256(args.positive_video),
            },
        },
        "fixture_video": {
            "path": str(video.resolve()),
            "sha256": sha256(video),
            "duration_seconds": metadata["duration_seconds"],
            "fps": FIXTURE_FPS,
        },
        "segments": fixture_segments(),
    }
    (args.output / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "VALIDATION.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "REPORT.md").write_text(
        render_report(manifest, validation), encoding="utf-8"
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return 0 if validation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
