from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .event_engine import NormalizedRect


@dataclass(frozen=True, slots=True)
class VisualChange:
    area_ratio: float = 0.0
    regions: tuple[NormalizedRect, ...] = ()


class BackgroundChangeDetector:
    """Low-resolution, illumination-compensated persistent scene change.

    This detector deliberately does not call a change "littering". It only
    provides independent visual evidence which the temporal event engine can
    combine with actors and semantic garbage detections.
    """

    def __init__(
        self,
        polygon: tuple[tuple[float, float], ...],
        *,
        width: int = 160,
        height: int = 90,
        pixel_threshold: float = 0.10,
        baseline_alpha: float = 0.04,
    ) -> None:
        if width < 16 or height < 16:
            raise ValueError("背景分析尺寸过小")
        if not 0 < pixel_threshold < 1:
            raise ValueError("pixel_threshold必须在(0, 1)范围内")
        if not 0 < baseline_alpha <= 1:
            raise ValueError("baseline_alpha必须在(0, 1]范围内")
        self.width = width
        self.height = height
        self.pixel_threshold = pixel_threshold
        self.baseline_alpha = baseline_alpha
        self._mask = _polygon_mask(polygon, width, height)
        if not self._mask.any():
            raise ValueError("背景分析ROI不包含有效像素")
        self._baseline: np.ndarray | None = None

    def observe(
        self,
        frame: np.ndarray,
        *,
        actor_present: bool,
    ) -> VisualChange:
        gray = _normalized_gray(frame, self.width, self.height)
        if self._baseline is None:
            if not actor_present:
                self._baseline = gray.copy()
            return VisualChange()

        signed = gray - self._baseline
        # Exposure/IR switching changes most pixels together. Removing the
        # median ROI delta prevents that global brightness shift becoming a
        # false scene-change alarm.
        illumination_delta = float(np.median(signed[self._mask]))
        difference = np.abs(signed - illumination_delta)
        changed = (difference >= self.pixel_threshold) & self._mask
        changed = _remove_isolated_pixels(changed)
        changed_count = int(np.count_nonzero(changed))
        roi_count = max(int(np.count_nonzero(self._mask)), 1)
        ratio = changed_count / roi_count

        regions: tuple[NormalizedRect, ...] = ()
        if changed_count:
            ys, xs = np.nonzero(changed)
            left = float(xs.min()) / self.width
            top = float(ys.min()) / self.height
            right = float(xs.max() + 1) / self.width
            bottom = float(ys.max() + 1) / self.height
            regions = (
                NormalizedRect(left, top, right - left, bottom - top),
            )

        # Learn slow daylight/weather drift only while the ROI is quiet and no
        # person/vehicle is interacting with it. Objects cannot be absorbed
        # into the baseline before the event engine has evaluated them.
        if not actor_present and ratio < 0.001:
            self._baseline = (
                (1.0 - self.baseline_alpha) * self._baseline
                + self.baseline_alpha * gray
            )
        return VisualChange(area_ratio=ratio, regions=regions)

    def commit(self, frame: np.ndarray) -> None:
        """Accept a confirmed stable scene as the new comparison baseline."""
        self._baseline = _normalized_gray(frame, self.width, self.height)


def _normalized_gray(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    value = np.asarray(frame)
    if value.ndim == 4:
        if value.shape[0] != 1:
            raise ValueError("单次背景分析只能处理一帧")
        value = value[0]
    if value.ndim == 3 and value.shape[0] in {1, 3, 4}:
        value = np.moveaxis(value, 0, -1)
    if value.ndim == 3:
        if value.shape[-1] < 3:
            value = value[..., 0]
        else:
            value = (
                value[..., 0].astype(np.float32) * 0.299
                + value[..., 1].astype(np.float32) * 0.587
                + value[..., 2].astype(np.float32) * 0.114
            )
    elif value.ndim != 2:
        raise ValueError(f"不支持的背景分析帧形状: {value.shape}")
    value = value.astype(np.float32, copy=False)
    if float(value.max(initial=0.0)) > 1.5:
        value = value / 255.0
    y_index = np.linspace(0, value.shape[0] - 1, height).astype(np.intp)
    x_index = np.linspace(0, value.shape[1] - 1, width).astype(np.intp)
    return value[np.ix_(y_index, x_index)]


def _remove_isolated_pixels(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(np.uint8), 1)
    neighbours = np.zeros_like(mask, dtype=np.uint8)
    for y_offset in range(3):
        for x_offset in range(3):
            neighbours += padded[
                y_offset : y_offset + mask.shape[0],
                x_offset : x_offset + mask.shape[1],
            ]
    return mask & (neighbours >= 4)


def _polygon_mask(
    polygon: tuple[tuple[float, float], ...],
    width: int,
    height: int,
) -> np.ndarray:
    x = (np.arange(width, dtype=np.float32) + 0.5) / width
    y = (np.arange(height, dtype=np.float32) + 0.5) / height
    grid_x, grid_y = np.meshgrid(x, y)
    inside = np.zeros((height, width), dtype=bool)
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        crossing = (current_y > grid_y) != (previous_y > grid_y)
        denominator = previous_y - current_y
        if abs(denominator) < 1e-12:
            denominator = 1e-12
        x_at_y = (
            (previous_x - current_x)
            * (grid_y - current_y)
            / denominator
            + current_x
        )
        inside ^= crossing & (grid_x < x_at_y)
        previous_x, previous_y = current_x, current_y
    return inside
