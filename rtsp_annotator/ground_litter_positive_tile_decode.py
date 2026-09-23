"""Real raw-PS decode adapter for Step 1C-2.

Imports PyAV **inside** the methods so the module (and the geometry/eligibility tests)
stay usable without an image stack.  Reuses the Step 1C-0 sampling helpers so a Step
1C-2 decode takes exactly the same code path as the Step 1C-0 verification frame:
``SequentialFrameReader.probe_seek`` seeks to the file-relative second and returns the
closest real frame together with its achieved offset.

Run the real generation with the profile virtualenv:

    .venv-profile/bin/python scripts/build_ground_litter_positive_tiles.py generate
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


class SourceFrameDecoder:
    """PyAV implementation of the positive-tile ``DecodeAdapter`` protocol."""

    def __init__(self, *, timeout: float = 90.0) -> None:
        self.timeout = float(timeout)

    # -- frame interval from the real stream (§13) --------------------------- #

    def probe_interval(self, path: Path) -> dict[str, Any]:
        try:
            import av
        except ImportError as exc:                      # pragma: no cover
            return {"ok": False, "error": f"PyAV missing: {exc}"}
        target = Path(path)
        if not target.is_file():
            return {"ok": False, "error": "file missing"}
        try:
            with av.open(str(target), timeout=(10.0, self.timeout)) as container:
                stream = next((s for s in container.streams if s.type == "video"), None)
                if stream is None:
                    return {"ok": False, "error": "no video stream"}
                fps = None
                try:
                    if stream.average_rate:
                        fps = float(stream.average_rate)
                except Exception:
                    fps = None
                times: list[float] = []
                for decoded in container.decode(video=0):
                    moment = decoded.time
                    if moment is None and decoded.pts is not None and stream.time_base:
                        moment = float(decoded.pts * stream.time_base)
                    if moment is not None:
                        times.append(float(moment))
                    if len(times) >= 40:
                        break
                deltas = sorted(b - a for a, b in zip(times, times[1:]) if b > a)
                measured = deltas[len(deltas) // 2] if deltas else None
                interval = None
                if measured and measured > 0:
                    interval = measured
                elif fps and fps > 0:
                    interval = 1.0 / fps
                if interval is None:
                    return {"ok": False, "error": "no frame interval could be derived"}
                return {
                    "ok": True,
                    "width": int(stream.codec_context.width),
                    "height": int(stream.codec_context.height),
                    "fps": fps,
                    "measured_pts_delta_seconds": measured,
                    "frame_interval_seconds": float(interval),
                    "frames_sampled": len(times),
                }
        except Exception as exc:                        # pragma: no cover
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    # -- one frame at a file-relative offset --------------------------------- #

    def decode_frame(self, path: Path, offset_seconds: float) -> dict[str, Any]:
        from rtsp_annotator.ground_litter_profile_sampling import SequentialFrameReader

        try:
            reader = SequentialFrameReader(path, timeout=self.timeout)
            frame, diagnostics = reader.probe_seek(float(offset_seconds))
        except Exception as exc:                        # pragma: no cover
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if frame is None or getattr(frame, "frame", None) is None:
            return {"ok": False,
                    "error": str(diagnostics.get("error") or "no_frame_at_offset")}
        image = frame.frame
        if getattr(image, "size", 0) == 0:
            return {"ok": False, "error": "empty_frame"}
        height, width = image.shape[:2]
        return {
            "ok": True,
            "frame": image,
            "width": int(width),
            "height": int(height),
            "decoded_offset_seconds": float(frame.time_seconds),
            "requested_offset_seconds": float(offset_seconds),
            "seek_diagnostics": {k: v for k, v in diagnostics.items()
                                 if k in ("keyframe_gap_seconds", "decoded_frames",
                                          "wall_seconds", "mode")},
        }


def load_default_decoder() -> SourceFrameDecoder:
    return SourceFrameDecoder()
