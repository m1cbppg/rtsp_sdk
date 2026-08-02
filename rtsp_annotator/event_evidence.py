from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .events import EventRecord, EventRepository


LOGGER = logging.getLogger(__name__)


class EventEvidenceWriter:
    """Persist event snapshots without being part of the publish pipeline."""

    def __init__(self, repository: EventRepository) -> None:
        self._repository = repository

    def attach_snapshots(
        self,
        events: list[EventRecord],
        frame: Any,
    ) -> None:
        if not events:
            return
        try:
            rgb = _rgb_uint8(frame)
            from PIL import Image

            for event in events:
                directory = (
                    self._repository.root / event.stream_id / "media"
                )
                directory.mkdir(parents=True, exist_ok=True)
                target = directory / f"{event.event_id}.jpg"
                temporary = target.with_suffix(".jpg.tmp")
                Image.fromarray(rgb, mode="RGB").save(
                    temporary,
                    format="JPEG",
                    quality=90,
                    optimize=True,
                )
                temporary.replace(target)
                event.snapshot_path = str(target)
                # Overwrite the initial metadata record atomically now that
                # evidence is available.
                self._repository.append(event)
        except Exception:
            LOGGER.exception("事件截图保存失败，不影响实时流")


def _rgb_uint8(frame: Any) -> np.ndarray:
    value = np.asarray(frame)
    if value.ndim == 4:
        value = value[0]
    if value.ndim == 3 and value.shape[0] in {3, 4}:
        value = np.moveaxis(value[:3], 0, -1)
    if value.ndim != 3 or value.shape[-1] < 3:
        raise ValueError(f"无法生成事件截图，帧形状无效: {value.shape}")
    value = value[..., :3]
    if np.issubdtype(value.dtype, np.floating):
        maximum = float(value.max(initial=0.0))
        if maximum <= 1.5:
            value = value * 255.0
    return np.clip(value, 0, 255).astype(np.uint8)
