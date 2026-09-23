"""Point-seeded bbox proposals for Step 1C-2M supplemental Required targets.

The human already said *where* the missed litter is; the machine only has to turn that
click into a small set of plausible boxes.  Three deliberately different methods are
offered, all seeded by the human point and all computed on the **already reviewed
640x640 tile PNG** (no source decode, no crop movement, no resize):

  A  local contrast connected component (Otsu on the detail image, component nearest
     the point) — reuses ``ClassicProposalEngine``, the code path proven in Step 1C-1;
  B  adaptive-threshold / smaller-scale region around the point, i.e. the same idea
     under a different local threshold so a low-contrast object still gets an option;
  C  point-centred empirical size prior taken from the verified bboxes of the accepted
     positive pool (median width/height and aspect) — this only helps *locate*; it never
     decides existence or truth.

Proposals are advisory data only.  Nothing here is ever written as a label, no
confidence is exposed, and the reviewer must pick a candidate explicitly.

cv2 is imported lazily so the geometry and state logic stays testable without an image
stack.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .ground_litter_localization_proposal import (
    ClassicProposalEngine,
    ProposalEngineError,
)

#: Methods that may appear in a proposal list, in preference order for lettering.
METHOD_POINT_COMPONENT = "point_local_contrast_component"
METHOD_POINT_ADAPTIVE = "point_adaptive_threshold_region"
METHOD_POINT_SIZE_PRIOR = "point_centred_size_prior"
METHOD_POINT_PADDED_WINDOW = "point_centred_padded_window"

MAX_PROPOSALS = 3
PROPOSAL_LETTERS = ("A", "B", "C")


def _clip(box: Sequence[float], size: int) -> list[int]:
    x1, y1, x2, y2 = (int(round(float(v))) for v in box)
    x1 = max(0, min(size - 1, x1))
    y1 = max(0, min(size - 1, y1))
    x2 = max(x1 + 1, min(size, x2))
    y2 = max(y1 + 1, min(size, y2))
    return [x1, y1, x2, y2]


def box_contains_point(box: Sequence[float], point: Sequence[float], *,
                       margin: float = 1.0) -> bool:
    return (box[0] - margin <= point[0] <= box[2] + margin
            and box[1] - margin <= point[1] <= box[3] + margin)


def box_touches_border(box: Sequence[float], size: int, *, tolerance: int = 1) -> str | None:
    """Return which sides of the tile the box touches or crosses (None when inside)."""
    sides = []
    if box[0] <= tolerance:
        sides.append("left")
    if box[1] <= tolerance:
        sides.append("top")
    if box[2] >= size - tolerance:
        sides.append("right")
    if box[3] >= size - tolerance:
        sides.append("bottom")
    return ",".join(sides) if sides else None


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    ix2, iy2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (float(a[2]) - float(a[0])) * (float(a[3]) - float(a[1]))
    area_b = (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def size_prior_from_labels(labels: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Median width/height of the verified one-class labels (tile pixels)."""
    widths: list[float] = []
    heights: list[float] = []
    for label in labels:
        box = label.get("tile_xyxy")
        if not box:
            continue
        width = float(box[2]) - float(box[0])
        height = float(box[3]) - float(box[1])
        if width > 0 and height > 0:
            widths.append(width)
            heights.append(height)

    def median(values: list[float], fallback: float) -> float:
        if not values:
            return fallback
        ordered = sorted(values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return float(ordered[middle])
        return (ordered[middle - 1] + ordered[middle]) / 2.0

    return {
        "width": round(median(widths, 48.0), 3),
        "height": round(median(heights, 40.0), 3),
        "sample_count": float(len(widths)),
    }


class PointProposalEngine:
    """Deterministic A/B/C proposal generator seeded by one human click."""

    def __init__(self, *, max_candidates: int = MAX_PROPOSALS,
                 adaptive_scale: float = 0.9) -> None:
        self.max_candidates = int(max_candidates)
        self.adaptive_scale = float(adaptive_scale)
        self._classic = ClassicProposalEngine(max_candidates=MAX_PROPOSALS)

    # -- individual methods -------------------------------------------------- #

    def _component_candidates(self, tile_path: Path, point: Sequence[float],
                              size: int, revision: int) -> list[dict[str, Any]]:
        try:
            rows = self._classic.propose(frame_path=tile_path, seed_bbox=None,
                                         point=[float(point[0]), float(point[1])],
                                         frame_width=size, frame_height=size,
                                         revision=int(revision))
        except ProposalEngineError:
            return []
        out = []
        for row in rows:
            method = str(row.get("method"))
            if method.startswith("point_local_contrast_component") or \
                    method == "local_contrast_component":
                method = METHOD_POINT_COMPONENT
            elif method == "point_centred_padded_window":
                method = METHOD_POINT_PADDED_WINDOW
            out.append({"bbox": _clip(row["bbox"], size), "method": method})
        return out

    def _adaptive_candidate(self, tile_path: Path, point: Sequence[float],
                            size: int) -> list[dict[str, Any]]:
        import cv2
        import numpy as np

        image = cv2.imread(str(tile_path))
        if image is None:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        block = max(11, int(size * 0.10) | 1)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY_INV, block, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
        px, py = int(round(point[0])), int(round(point[1]))
        px = max(0, min(size - 1, px))
        py = max(0, min(size - 1, py))
        label_id = int(labels[py, px])
        chosen = None
        if label_id > 0:
            chosen = label_id
        else:
            # nothing under the cursor: take the nearest component of a sane size
            best = None
            for index in range(1, count):
                area = int(stats[index, cv2.CC_STAT_AREA])
                if area < 6:
                    continue
                cx = stats[index, cv2.CC_STAT_LEFT] + stats[index, cv2.CC_STAT_WIDTH] / 2
                cy = stats[index, cv2.CC_STAT_TOP] + stats[index, cv2.CC_STAT_HEIGHT] / 2
                distance = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                if best is None or distance < best[0]:
                    best = (distance, index)
            if best is not None and best[0] <= max(24.0, 0.06 * size):
                chosen = best[1]
        if chosen is None:
            return []
        x = int(stats[chosen, cv2.CC_STAT_LEFT])
        y = int(stats[chosen, cv2.CC_STAT_TOP])
        w = int(stats[chosen, cv2.CC_STAT_WIDTH])
        h = int(stats[chosen, cv2.CC_STAT_HEIGHT])
        if w < 2 or h < 2:
            return []
        return [{"bbox": _clip([x - 2, y - 2, x + w + 2, y + h + 2], size),
                 "method": METHOD_POINT_ADAPTIVE}]

    def _size_prior_candidate(self, point: Sequence[float], size: int,
                              prior: Mapping[str, float]) -> list[dict[str, Any]]:
        width = float(prior.get("width") or 48.0) * self.adaptive_scale
        height = float(prior.get("height") or 40.0) * self.adaptive_scale
        width = max(6.0, min(float(size), width))
        height = max(6.0, min(float(size), height))
        cx, cy = float(point[0]), float(point[1])
        return [{"bbox": _clip([cx - width / 2.0, cy - height / 2.0,
                                cx + width / 2.0, cy + height / 2.0], size),
                 "method": METHOD_POINT_SIZE_PRIOR}]

    # -- public -------------------------------------------------------------- #

    def propose(self, *, tile_path: Path | str, point_tile_xy: Sequence[float],
                size: int, size_prior: Mapping[str, float] | None = None,
                revision: int = 1) -> dict[str, Any]:
        """Return at most ``max_candidates`` proposals, lettered A/B/C."""
        path = Path(tile_path)
        # cheap validation first: a bad request must not depend on the file system
        point = [float(point_tile_xy[0]), float(point_tile_xy[1])]
        if not (0.0 <= point[0] < size and 0.0 <= point[1] < size):
            return {"ok": False, "error_code": "POINT_OUTSIDE_TILE",
                    "message": f"point {point} outside the {size}x{size} tile",
                    "candidates": []}
        if not path.is_file():
            return {"ok": False, "error_code": "TILE_IMAGE_MISSING",
                    "message": f"tile image not found: {path}", "candidates": []}
        prior = dict(size_prior or {})
        try:
            component_rows = self._component_candidates(path, point, size, revision)
            adaptive_rows = self._adaptive_candidate(path, point, size)
            prior_rows = self._size_prior_candidate(point, size, prior)
        except ImportError as exc:                      # pragma: no cover
            return {"ok": False, "error_code": "PROPOSAL_BACKEND_MISSING",
                    "message": f"cv2 is required for proposal generation: {exc}",
                    "candidates": []}
        except Exception as exc:                        # pragma: no cover
            return {"ok": False, "error_code": "PROPOSAL_FAILED",
                    "message": f"{type(exc).__name__}: {exc}", "candidates": []}

        ordered: list[dict[str, Any]] = []
        for group in (component_rows, adaptive_rows, prior_rows):
            for row in group:
                ordered.append({"bbox": list(row["bbox"]), "method": str(row["method"])})
        # One candidate per method first, in A/B/C preference order, then the extras.
        first_of_method: list[dict[str, Any]] = []
        seen_methods: set[str] = set()
        extras: list[dict[str, Any]] = []
        for row in ordered:
            if row["method"] in seen_methods:
                extras.append(row)
            else:
                seen_methods.add(row["method"])
                first_of_method.append(row)
        ordered = first_of_method + extras

        kept: list[dict[str, Any]] = []
        for row in ordered:
            if any(iou(row["bbox"], old["bbox"]) >= 0.92 for old in kept):
                continue
            kept.append(row)
            if len(kept) >= self.max_candidates:
                break
        if not kept:
            return {"ok": False, "error_code": "NO_PROPOSAL_FOUND",
                    "message": ("no region could be proposed around this point; "
                                "repoint closer to the target centre"),
                    "candidates": []}

        for index, row in enumerate(kept):
            row["letter"] = PROPOSAL_LETTERS[index]
            row["contains_point"] = box_contains_point(row["bbox"], point)
            row["touches_border"] = box_touches_border(row["bbox"], size)
            row["point_tile"] = list(point)
            row["revision"] = int(revision)
        # a proposal that does not contain the click is a worse explanation than one that
        # does, so it is demoted (but never silently dropped: the reviewer decides)
        kept.sort(key=lambda row: (not row["contains_point"],
                                   PROPOSAL_LETTERS.index(row["letter"])))
        for index, row in enumerate(kept):
            row["letter"] = PROPOSAL_LETTERS[index]
        return {"ok": True, "candidates": kept, "point_tile": list(point),
                "revision": int(revision), "size_prior": prior}


def load_default_engine() -> PointProposalEngine:
    return PointProposalEngine()
