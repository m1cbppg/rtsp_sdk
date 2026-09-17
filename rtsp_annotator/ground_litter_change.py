"""Conservative local background-change evidence for ground-litter candidates.

This module does not decide that a changed patch is litter.  It only answers
whether a candidate's local ground patch differs from a reviewed reference
after removing a broad illumination offset.  The temporal tracker and human
review remain responsible for the event decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class ChangeConfig:
    delta_threshold: float = 25.0
    minimum_change_fraction: float = 0.02
    maximum_change_fraction: float = 1.0
    minimum_component_area: int = 24
    blur_kernel: int = 3

    def validate(self) -> None:
        if not isfinite(self.delta_threshold) or self.delta_threshold <= 0:
            raise ValueError("delta_threshold must be positive")
        if not 0 < self.minimum_change_fraction <= self.maximum_change_fraction <= 1:
            raise ValueError("invalid change fraction bounds")
        if self.minimum_component_area < 1:
            raise ValueError("minimum_component_area must be positive")
        if self.blur_kernel < 1 or self.blur_kernel % 2 == 0:
            raise ValueError("blur_kernel must be a positive odd number")


@dataclass(frozen=True, slots=True)
class ChangeEvidence:
    changed: bool
    change_fraction: float
    largest_component_area: int
    illumination_offset: float
    pixels: int


def compare_patch(reference: np.ndarray, current: np.ndarray,
                  box: tuple[int, int, int, int],
                  config: ChangeConfig = ChangeConfig()) -> ChangeEvidence:
    """Compare one pixel box in two BGR images.

    A broad median luminance offset is removed before thresholding.  This keeps
    normal exposure changes from becoming events while retaining a localized
    object that occupies a compact connected component.
    """
    config.validate()
    if reference is None or current is None or reference.shape != current.shape:
        raise ValueError("reference and current images must have the same shape")
    if reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError("images must be BGR color images")
    x, y, right, bottom = (int(value) for value in box)
    height, width = reference.shape[:2]
    x = max(0, min(width, x)); right = max(0, min(width, right))
    y = max(0, min(height, y)); bottom = max(0, min(height, bottom))
    if right <= x or bottom <= y:
        raise ValueError("box must have positive area")
    # Estimate exposure from the surrounding ring, never from the candidate
    # itself: a solid paper filling its box must not be normalized away.
    margin = max(12, right-x, bottom-y)
    a, b = max(0, x-margin), max(0, y-margin)
    c, d = min(width, right+margin), min(height, bottom+margin)
    old = reference[b:d, a:c]
    new = current[b:d, a:c]
    if config.blur_kernel > 1:
        old = cv2.GaussianBlur(old, (config.blur_kernel, config.blur_kernel), 0)
        new = cv2.GaussianBlur(new, (config.blur_kernel, config.blur_kernel), 0)
    delta = new.astype(np.float32) - old.astype(np.float32)
    ring = np.ones(delta.shape[:2], bool)
    ring[y-b:bottom-b, x-a:right-a] = False
    offsets = np.median(delta[ring], axis=0) if ring.any() else np.median(delta.reshape(-1, 3), axis=0)
    residual = np.max(np.abs(delta[y-b:bottom-b, x-a:right-a] - offsets), axis=2)
    offset = float(np.mean(offsets))
    changed = (residual >= config.delta_threshold).astype(np.uint8)
    changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    pixels = int(changed.size)
    fraction = float(changed.mean()) if pixels else 0.0
    count, _, stats, _ = cv2.connectedComponentsWithStats(changed, 8)
    largest = int(max(stats[1:, cv2.CC_STAT_AREA], default=0)) if count > 1 else 0
    is_changed = (config.minimum_change_fraction <= fraction <= config.maximum_change_fraction
                  and largest >= config.minimum_component_area)
    return ChangeEvidence(is_changed, fraction, largest, offset, pixels)


def propose_changes(reference, current, mask, actors=(), config=ChangeConfig(), *, max_regions=32):
    """Bounded independent ground-object proposals; no garbage classification.

    Reference is explicit, frozen, and never learned from startup. Broad
    lighting/scene changes abstain. Masks exclude non-ground and actor boxes.
    """
    config.validate()
    if reference.shape != current.shape or mask.shape != current.shape[:2]:
        raise ValueError('Change reference/mask dimensions mismatch')
    eligible = mask.astype(bool).copy()
    for x, y, r, b in actors:
        x, y, r, b = map(int, (x, y, r, b))
        eligible[max(0,y):max(0,b), max(0,x):max(0,r)] = False
    if eligible.sum() < 160:
        return [], 'insufficient_visible_ground'
    delta = current.astype(np.float32) - reference.astype(np.float32)
    offset = np.median(delta[eligible], axis=0)
    residual = np.max(np.abs(delta-offset), axis=2)
    changed = ((residual > config.delta_threshold) & eligible).astype(np.uint8)
    if changed.sum() / eligible.sum() > .25:
        return [], 'broad_change_unknown'
    changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(changed, 8)
    regions = sorted(stats[1:], key=lambda s: -int(s[4]))
    boxes = [[int(x), int(y), int(x+w), int(y+h)] for x,y,w,h,area in regions
             if config.minimum_component_area <= area <= 16000 and min(w,h) >= 12]
    return boxes[:max_regions], 'ok' if len(boxes) <= max_regions else 'proposal_limit'


def compare_boxes(reference: np.ndarray, current: np.ndarray,
                  boxes: list[tuple[int, int, int, int]],
                  config: ChangeConfig = ChangeConfig()) -> tuple[ChangeEvidence, ...]:
    """Return evidence for multiple candidate boxes in stable input order."""
    return tuple(compare_patch(reference, current, box, config) for box in boxes)
