"""Classical point-seeded bbox proposals (Step 1C logic) for the v2 bbox stage.

Ported unchanged from the Rapid v1 review service so candidate C in the new
review UI is the *same* classical proposal, not a new algorithm.  It never
discovers objects on its own: it only refines a bbox around a point a human (or
a model) already chose.
"""
from __future__ import annotations

from typing import Any, Sequence


def classic_cv_proposals(frame, x: float, y: float, *, half: int = 96) -> list[dict[str, Any]]:
    """Return up to three point-seeded proposals (adaptive / otsu / edge)."""
    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    x0 = int(max(0, min(width - 1, round(x - half))))
    x1 = int(max(1, min(width, round(x + half))))
    y0 = int(max(0, min(height - 1, round(y - half))))
    y1 = int(max(1, min(height, round(y + half))))
    window = frame[y0:y1, x0:x1]
    if window.size == 0:
        return []
    gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    masks = {
        "A-adaptive": cv2.morphologyEx(
            cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY_INV, 31, 8),
            cv2.MORPH_CLOSE, kernel, iterations=2),
        "B-otsu": cv2.morphologyEx(
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
            cv2.MORPH_CLOSE, kernel, iterations=2),
        "C-edge": cv2.morphologyEx(cv2.Canny(gray, 60, 160), cv2.MORPH_CLOSE, kernel,
                                   iterations=3),
    }
    point = (int(round(x)) - x0, int(round(y)) - y0)
    proposals: list[dict[str, Any]] = []
    for label, mask in masks.items():
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        best = None
        for index in range(1, count):
            bx, by, bw, bh, area = stats[index]
            if area < 24 or area > window.shape[0] * window.shape[1] * 0.9:
                continue
            inside = (bx <= point[0] < bx + bw and by <= point[1] < by + bh)
            distance = 0 if inside else min(
                abs(point[0] - bx), abs(point[0] - (bx + bw)),
                abs(point[1] - by), abs(point[1] - (by + bh)))
            score = (0 if inside else 1, distance, -area)
            if best is None or score < best[0]:
                best = (score, (bx, by, bw, bh, area), inside)
        if best is None:
            continue
        bx, by, bw, bh, area = best[1]
        proposals.append({
            "label": label,
            "bbox_xyxy": [float(bx + x0), float(by + y0),
                          float(bx + x0 + bw), float(by + y0 + bh)],
            "area": int(area),
            "contains_point": bool(best[2]),
            "candidate_half": half,
        })
    return proposals


def dedupe_proposals(options: Sequence[dict[str, Any]], *, iou_threshold: float = 0.85) -> list[dict[str, Any]]:
    """Drop near-identical proposals so A/B/C are never three copies of one box."""
    from rtsp_annotator.ground_litter_historical import bbox_iou

    keep: list[dict[str, Any]] = []
    for option in sorted(options, key=lambda item: (not item.get("contains_point"),
                                                    -(item.get("area") or 0))):
        box = option.get("bbox_xyxy")
        if not box:
            continue
        if any(bbox_iou(box, other["bbox_xyxy"]) >= iou_threshold for other in keep):
            continue
        keep.append(dict(option))
    return keep
