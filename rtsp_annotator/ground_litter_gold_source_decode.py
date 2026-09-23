"""Real decode adapter for Step 1C-0, built on the existing sampling helpers.

Deliberately imports PyAV/cv2 **inside** the methods: the repository virtualenv used
for unit tests has no cv2, and Step 1C-0 must stay testable there.  Run the real
adapter with the profile virtualenv (``.venv-profile``) which has av + cv2 + numpy.

Reuses rather than reimplements:
``ground_litter_profile_sampling.probe_recording`` (width/height/codec/duration) and
``SequentialFrameReader.probe_seek`` (seek to a file-relative second and return the
closest real frame plus the achieved offset).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


class ProfileDecodeAdapter:
    """PyAV + cv2 implementation of the recovery ``DecodeAdapter`` protocol."""

    def __init__(self, *, timeout: float = 60.0, quality: int = 95) -> None:
        self.timeout = float(timeout)
        self.quality = int(quality)

    def probe(self, path: Path) -> dict[str, Any]:
        from rtsp_annotator.ground_litter_profile_sampling import probe_recording

        result = probe_recording(path, timeout=self.timeout)
        return {
            "ok": bool(result.ok),
            "width": int(result.width),
            "height": int(result.height),
            "duration_seconds": float(result.duration_seconds),
            "frame_count": int(result.frame_count),
            "codec": str(result.codec),
            "error": str(result.error or ""),
        }

    def sample_frame(self, path: Path, offset_seconds: float) -> dict[str, Any]:
        import cv2
        from rtsp_annotator.ground_litter_profile_sampling import SequentialFrameReader

        reader = SequentialFrameReader(path, timeout=self.timeout)
        frame, diagnostics = reader.probe_seek(float(offset_seconds))
        if frame is None:
            return {"ok": False,
                    "error": str(diagnostics.get("error") or "no_frame_at_offset")}
        image = frame.frame
        if image is None or getattr(image, "size", 0) == 0:
            return {"ok": False, "error": "empty_frame"}
        ok, buffer = cv2.imencode(".jpg", image,
                                  [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if not ok:
            return {"ok": False, "error": "jpeg_encode_failed"}
        height, width = image.shape[:2]
        return {
            "ok": True,
            "frame_jpeg": buffer.tobytes(),
            "width": int(width),
            "height": int(height),
            "decoded_offset_seconds": float(frame.time_seconds),
            "decoded_source": str(frame.source),
        }


def load_default_adapter() -> ProfileDecodeAdapter:
    return ProfileDecodeAdapter()
