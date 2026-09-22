"""Offline native-resolution frame sampler for PS/MP4 replay evidence."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2
import time


@dataclass(frozen=True, slots=True)
class ReplaySampleSummary:
    source: str
    output: str
    frames: int
    source_fps: float
    width: int
    height: int
    duration_seconds: float
    requested_offset_seconds: float = 0.0
    content_time_verified: bool = False


def sample_video(source: str | Path, output: str | Path, *, sample_fps: float = 0.5,
                 max_frames: int = 10000) -> ReplaySampleSummary:
    if not 0 < sample_fps <= 10:
        raise ValueError("sample_fps must be in (0, 10]")
    if max_frames < 1:
        raise ValueError("max_frames must be positive")
    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source_path), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise RuntimeError("could not open replay video")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    if not 1 <= source_fps <= 240:
        source_fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = 0.0
    next_sample = 0.0
    frame_index = 0
    written = 0
    try:
        while written < max_frames:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            timestamp = frame_index / source_fps
            frame_index += 1
            duration = timestamp
            if timestamp + 1e-9 < next_sample:
                continue
            target = output_path / f"frame-{written:06d}-{timestamp:010.3f}.jpg"
            if not cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise RuntimeError(f"could not write {target.name}")
            written += 1
            next_sample += 1.0 / sample_fps
    finally:
        capture.release()
    summary = ReplaySampleSummary(str(source_path), str(output_path), written,
                                  source_fps, width, height, duration,
                                  requested_offset_seconds=0.0,
                                  content_time_verified=False)
    (output_path / "manifest.json").write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return summary


def sample_stream(source: str, output: str | Path, *, duration_seconds: float = 60,
                  sample_fps: float = 0.5, max_frames: int = 10000,
                  source_label: str = "<redacted>",
                  start_offset_seconds: float = 0.0) -> ReplaySampleSummary:
    """Sample an already-resolved RTSP URL without persisting the URL."""
    if not source.startswith(("rtsp://", "rtsps://")):
        raise ValueError("source must be an RTSP URL")
    if (duration_seconds <= 0 or max_frames < 1 or not 0 < sample_fps <= 10
            or not isinstance(start_offset_seconds, (int, float))
            or start_offset_seconds < 0):
        raise ValueError("invalid stream sampling limits")
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG, [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 12000,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, 12000,
    ])
    if not capture.isOpened():
        raise RuntimeError("could not open playback RTSP")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    if not 1 <= source_fps <= 240:
        source_fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    started = time.monotonic()
    next_sample = 0.0
    written = 0
    duration = 0.0
    try:
        while written < max_frames and time.monotonic() - started < start_offset_seconds + duration_seconds:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            elapsed = time.monotonic() - started
            if elapsed + 1e-9 < start_offset_seconds:
                continue
            duration = elapsed - start_offset_seconds
            if duration + 1e-9 < next_sample:
                continue
            target = output_path / f"frame-{written:06d}-{duration:010.3f}.jpg"
            if not cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise RuntimeError(f"could not write {target.name}")
            written += 1
            next_sample += 1.0 / sample_fps
    finally:
        capture.release()
    summary = ReplaySampleSummary(source_label, str(output_path), written,
                                  source_fps, width, height, duration,
                                  requested_offset_seconds=float(start_offset_seconds),
                                  content_time_verified=False)
    (output_path / "manifest.json").write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return summary
