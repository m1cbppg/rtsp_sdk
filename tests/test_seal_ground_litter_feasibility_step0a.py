from __future__ import annotations

import importlib.util
from pathlib import Path

from rtsp_annotator.ground_litter_recording_source import RecordingFile

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "seal_ground_litter_feasibility_step0a.py"
spec = importlib.util.spec_from_file_location("step0a", SCRIPT)
assert spec and spec.loader
step0a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step0a)


def recording(file_id: str, start: str, end: str) -> RecordingFile:
    return RecordingFile(
        file_id=file_id,
        file_name=f"{file_id}.ps",
        record_start=start,
        record_end=end,
        file_size=123,
        file_type="media",
    )


def test_overlap_keeps_boundary_straddlers() -> None:
    start = step0a.parse_time("2026-09-23 08:00:00")
    end = step0a.parse_time("2026-09-23 09:00:00")
    files = [
        recording("before", "2026-09-23 07:50:00", "2026-09-23 07:59:59"),
        recording("left", "2026-09-23 07:58:00", "2026-09-23 08:03:00"),
        recording("inside", "2026-09-23 08:03:00", "2026-09-23 08:58:00"),
        recording("right", "2026-09-23 08:58:00", "2026-09-23 09:03:00"),
        recording("after", "2026-09-23 09:00:00", "2026-09-23 09:05:00"),
    ]
    assert [x.file_id for x in files if step0a.overlaps(x, start, end)] == [
        "left", "inside", "right"
    ]


def test_coverage_merges_overlap_and_reports_gap() -> None:
    start = step0a.parse_time("2026-09-23 08:00:00")
    end = step0a.parse_time("2026-09-23 08:10:00")
    files = [
        recording("a", "2026-09-23 07:59:00", "2026-09-23 08:04:00"),
        recording("b", "2026-09-23 08:03:00", "2026-09-23 08:06:00"),
        recording("c", "2026-09-23 08:06:05", "2026-09-23 08:11:00"),
    ]
    result = step0a.coverage(files, start, end)
    assert result["covered_seconds"] == 595.0
    assert result["max_gap_seconds"] == 5.0
    assert len(result["gaps"]) == 1


def test_archive_path_separates_split_and_camera(tmp_path: Path) -> None:
    path = step0a.archive_path(
        tmp_path,
        "exp1",
        {"split": "sealed_test", "camera_id": "01030"},
        {"file_id": "abc/def", "file_name": "x.ps"},
    )
    assert path == (
        tmp_path / "exp1" / "sealed_test" / "01030" / "raw" / "abc_def__x.ps"
    )
