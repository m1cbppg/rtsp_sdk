"""Profile Bank / 工厂测试共用夹具。

只生成合成数据，不访问网络、不读真实录像。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from rtsp_annotator.ground_litter_profile_bank import (
    default_camera_geometry, default_matcher_config, publish_version,
)

# 明显是假地址，但保留签名参数形态，用于断言「签名 URL 不落盘」。
FAKE_SIGNED_URL = (
    "https://media.example.invalid/ps/file.ps"
    "?Token=FAKE_TOKEN_VALUE&Signature=FAKE_SIGNATURE_VALUE&TimeStamp=1758000000"
)
FAKE_TOKEN_MARKER = "FAKE_SIGNATURE_VALUE"


class FakeResponse:
    """最小 HTTP 响应桩：context manager + read + status + headers。"""

    def __init__(self, body: bytes = b"", status: int = 200,
                 headers: Mapping[str, str] | None = None) -> None:
        self._body = body
        self._offset = 0
        self.status = status
        self.headers = dict(headers or {})

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            chunk = self._body[self._offset:]
            self._offset = len(self._body)
            return chunk
        chunk = self._body[self._offset:self._offset + size]
        self._offset += len(chunk)
        return chunk

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_exc: Any) -> bool:
        return False


def make_response(body: bytes = b"", *, status: int = 200,
                  headers: Mapping[str, str] | None = None) -> FakeResponse:
    return FakeResponse(body, status=status, headers=headers)


def synthetic_noise(width: int, height: int, *, bias: bool = False) -> dict[str, np.ndarray]:
    shape = (height, width)
    base = np.zeros(shape, np.float32)
    payload = {
        "seed_signature_threshold": base + 3.0,
        "seed_luminance_threshold": base + 100.0,
        "support_signature_threshold": base + 1.2,
        "support_luminance_threshold": base + 25.0,
        "seed_signature_median": base.copy(),
        "seed_luminance_median": base.copy(),
        "seed_signature_mad": base.copy(),
        "seed_luminance_mad": base.copy(),
        "seed_signature_q95": base.copy(),
        "seed_luminance_q95": base.copy(),
        "seed_signature_cap": base + 3.8,
        "seed_luminance_cap": base + 130.0,
        "seed_signature_raw": base + (4.2 if bias else 3.0),
        "seed_luminance_raw": base + (140.0 if bias else 100.0),
        "support_blocks": base + 5.0,
        "over_cap_fraction": base + (1.0 if bias else 0.0),
        "bias_flag": np.full(shape, 1 if bias else 0, np.uint8),
        "low_support": np.zeros(shape, np.uint8),
    }
    return payload


def synthetic_descriptor(width: int = 16, height: int = 9, *,
                         luminance: float = 120.0, chroma: float = 8.0,
                         structure: float = 40.0,
                         weight: float = 1.0) -> dict[str, np.ndarray]:
    return {
        "grid_luminance": np.full((height, width), luminance, np.float32),
        "grid_chroma": np.full((height, width), chroma, np.float32),
        "grid_structure": np.full((height, width), structure, np.float32),
        "grid_weight": np.full((height, width), weight, np.float32),
    }


def synthetic_reference(width: int, height: int, *, value: int = 120,
                        gradient: bool = True) -> np.ndarray:
    """合成背景：参数顺序是 (width, height)。可选水平亮度梯度。"""
    canvas = np.full((height, width, 3), value, np.uint8)
    if gradient:
        ramp = np.linspace(-20, 20, width, dtype=np.float32)
        for channel in range(3):
            canvas[..., channel] = np.clip(
                canvas[..., channel].astype(np.float32) + ramp[None, :], 0, 255
            ).astype(np.uint8)
    return canvas


def synthetic_valid(width: int, height: int, *, inset: int = 0) -> np.ndarray:
    """参数顺序是 (width, height)，与 build_synthetic_bank/reference 一致。"""
    mask = np.zeros((height, width), np.uint8)
    mask[inset:height - inset, inset:width - inset] = 255
    return mask


def build_synthetic_bank(
    root: str | Path, bank_id: str, version: str, *, profiles: int = 2,
    width: int = 160, height: int = 120, supersede_existing: bool = False,
    bias: bool = False, prior_suitable: bool = True,
) -> Path:
    entries = []
    envelopes: dict[str, Any] = {}
    for index in range(profiles):
        reference = synthetic_reference(width, height, value=110 + 5 * index)
        valid = synthetic_valid(width, height)
        profile_id = f"p{index + 1:04d}"
        envelopes[profile_id] = {
            "envelope": {
                "enter": 1.5 + 0.1 * index, "hold": 1.8 + 0.1 * index,
                "calibrated": True, "samples": 24,
                "source": "synthetic_calibration_set",
            },
            "calibration": {"source": "synthetic_calibration_set"},
        }
        entries.append({
            "profile_id": profile_id,
            "reference": reference,
            "valid": valid,
            "noise": synthetic_noise(width, height, bias=bias),
            "descriptor": synthetic_descriptor(luminance=110.0 + 5 * index),
            "profile_json": {
                "sample_count": 12, "valid_ground_fraction": 1.0,
                "source": {"kind": "synthetic_test"},
                # 跨模块契约：能力字段（v4 复核后 loader 要求存在）。
                "match_eligible": True,
                "prior_suitable": bool(prior_suitable),
                "calibration_state": (
                    "independent_matched" if prior_suitable
                    else "reference_self_low_support"
                ),
            },
        })
    matcher = default_matcher_config()
    matcher["calibration"] = {
        "source": "synthetic_calibration_set",
        "calibrated_utc": "2026-09-20T00:00:00Z",
        "method": "envelope_from_samples",
        "inputs": {"files": ["synthetic"], "sha256": "synthetic"},
    }
    matcher["profiles"] = envelopes
    return publish_version(
        root, bank_id, version,
        manifest={"notes": "synthetic test bank"},
        camera_geometry=default_camera_geometry(width, height, camera_id=bank_id),
        matcher=matcher,
        profiles=entries,
        supersede_existing=supersede_existing,
    )


def write_test_video(
    path: str | Path, *, frames: int = 12, width: int = 160, height: int = 120,
    fps: float = 5.0, value: int = 120, patch: bool = False,
) -> Path:
    """写一个可被 PyAV/OpenCV 解码的合成视频。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height),
    )
    if not writer.isOpened():  # pragma: no cover - 编解码器缺失环境
        raise RuntimeError("无法创建合成测试视频")
    try:
        for index in range(frames):
            frame = synthetic_reference(width, height, value=value)
            if patch and index % 3 == 0:
                frame[height // 2 - 4:height // 2 + 4,
                      width // 2 - 4:width // 2 + 4] = 240
            writer.write(frame)
    finally:
        writer.release()
    return target


__all__ = [
    "FAKE_SIGNED_URL", "FAKE_TOKEN_MARKER", "FakeResponse", "build_synthetic_bank",
    "make_response", "synthetic_descriptor", "synthetic_noise",
    "synthetic_reference", "synthetic_valid", "write_test_video",
]
