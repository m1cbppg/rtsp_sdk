"""Classic-CV bbox proposals for Step 1C-1 localization repair.

Deliberately **not** a detector and **not** a learned segmenter.  The task forbids
letting a future evaluation model decide truth existence, and the truth here is
already frozen as REQUIRED_LITTER — all that is needed is a *localization helper* that
proposes plausible boxes around a target whose approximate position is known.

What it uses (deterministic, explainable, no weights):
  * local high-frequency contrast (|gray - blur|) inside a search window,
  * Otsu thresholding + morphology + connected components,
  * component ranking by distance to the known anchor and area plausibility,
  * a seed-rescale fallback so an oversized/undersized box always has a corrected
    alternative even when segmentation finds nothing.

Every candidate carries only a letter (A/B/C) and a method name; there is no score,
no model identity and no confidence, so the UI cannot leak one.

Imports cv2/numpy lazily: the unit-test venv has neither.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


class ProposalEngineError(RuntimeError):
    """Proposal generation could not run."""


class ClassicProposalEngine:
    """Local-contrast + component proposals, with a seed-rescale fallback."""

    def __init__(self, *, max_candidates: int = 3, search_expand: float = 1.6,
                 # floor is a fraction of the FRAME, never of the seed: an oversized
                 # seed must not raise the bar so high that small litter is filtered out.
                 min_component_fraction: float = 1.2e-5,
                 max_component_fraction: float = 0.60) -> None:
        self.max_candidates = int(max_candidates)
        self.search_expand = float(search_expand)
        self.min_component_fraction = float(min_component_fraction)
        self.max_component_fraction = float(max_component_fraction)

    # -- geometry ----------------------------------------------------------- #

    @staticmethod
    def _clip(box: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = (int(round(float(v))) for v in box)
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width, x2))
        y2 = max(y1 + 1, min(height, y2))
        return x1, y1, x2, y2

    def _search_roi(self, seed: Sequence[float] | None, point: Sequence[float] | None,
                    width: int, height: int) -> tuple[int, int, int, int]:
        if seed is not None:
            x1, y1, x2, y2 = (float(v) for v in seed)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            # A wrong/short box must still be able to reach the real object nearby, so
            # the search window has a generous floor rather than hugging the seed.
            half_w = max(96.0, (x2 - x1) * self.search_expand)
            half_h = max(96.0, (y2 - y1) * self.search_expand)
            return self._clip((cx - half_w, cy - half_h, cx + half_w, cy + half_h),
                              width, height)
        if point is not None:
            px, py = float(point[0]), float(point[1])
            half = max(64.0, 0.04 * width)
            return self._clip((px - half, py - half, px + half, py + half), width, height)
        raise ProposalEngineError("need either a seed bbox or a point")

    # -- core --------------------------------------------------------------- #

    def _components(self, detail, roi, min_area: float, max_area: float):
        import cv2
        import numpy as np

        x1, y1, x2, y2 = roi
        patch = detail[y1:y2, x1:x2]
        if patch.size == 0:
            return []
        patch = cv2.GaussianBlur(patch, (5, 5), 0)
        _, binary = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
        rows = []
        for index in range(1, count):
            bx, by, bw, bh, area = (int(stats[index, cv2.CC_STAT_LEFT]),
                                    int(stats[index, cv2.CC_STAT_TOP]),
                                    int(stats[index, cv2.CC_STAT_WIDTH]),
                                    int(stats[index, cv2.CC_STAT_HEIGHT]),
                                    int(stats[index, cv2.CC_STAT_AREA]))
            if bw < 2 or bh < 2 or area < min_area or area > max_area:
                continue
            rows.append({
                "bbox": [x1 + bx, y1 + by, x1 + bx + bw, y1 + by + bh],
                "area": area,
                "centroid": [float(centroids[index][0]) + x1,
                             float(centroids[index][1]) + y1],
            })
        return rows

    def propose(self, *, frame_path: Path, seed_bbox: Sequence[float] | None,
                point: Sequence[float] | None, frame_width: int, frame_height: int,
                revision: int) -> list[dict[str, Any]]:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            raise ProposalEngineError("cv2 is required for proposal generation") from exc

        image = cv2.imread(str(frame_path))
        if image is None:
            raise ProposalEngineError("verification frame could not be read")
        height, width = image.shape[:2]
        if (int(width), int(height)) != (int(frame_width), int(frame_height)):
            # never map a proposal from a differently-sized image
            raise ProposalEngineError(
                f"frame is {width}x{height} but source is {frame_width}x{frame_height}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (0, 0), 3)
        detail = cv2.absdiff(gray, blurred)

        roi = self._search_roi(seed_bbox, point, width, height)
        roi_area = float((roi[2] - roi[0]) * (roi[3] - roi[1]))
        if seed_bbox is not None:
            seed_area = max(1.0, (float(seed_bbox[2]) - float(seed_bbox[0]))
                            * (float(seed_bbox[3]) - float(seed_bbox[1])))
        else:
            seed_area = max(1.0, 0.00006 * width * height)
        frame_area = float(width * height)
        min_area = max(6.0, self.min_component_fraction * frame_area)
        max_area = max(min_area * 6.0, self.max_component_fraction * roi_area)
        components = self._components(detail, roi, min_area, max_area)

        if seed_bbox is not None:
            anchor = [(float(seed_bbox[0]) + float(seed_bbox[2])) / 2.0,
                      (float(seed_bbox[1]) + float(seed_bbox[3])) / 2.0]
        else:
            anchor = [float(point[0]), float(point[1])]

        def component_rank(row):
            dx = row["centroid"][0] - anchor[0]
            dy = row["centroid"][1] - anchor[1]
            distance = (dx * dx + dy * dy) ** 0.5
            ratio = max(row["area"] / seed_area, seed_area / max(1.0, row["area"]))
            return (distance / max(1.0, 0.25 * (roi[2] - roi[0])), min(ratio, 8.0))

        components.sort(key=component_rank)

        candidates: list[dict[str, Any]] = []
        for row in components[:2]:
            candidates.append({"bbox": list(row["bbox"]),
                               "method": "local_contrast_component",
                               "revision": revision})

        # Seed rescale: always offer a corrected version of the existing box so an
        # obviously oversized/undersized/offset box has a plainer alternative.
        if seed_bbox is not None:
            if components:
                target_area = float(components[0]["area"])
                scale = (target_area / seed_area) ** 0.5
                scale = max(0.35, min(2.5, scale))
                method = ("seed_rescaled_to_component_area" if abs(scale - 1.0) > 0.12
                          else "seed_unchanged_shape")
            else:
                # Nothing segmented nearby.  Offer several concentric rescalings rather
                # than one blind shrink, so an oversized/undersized seed still has real
                # alternatives for the reviewer to judge.
                x1, y1, x2, y2 = (float(v) for v in seed_bbox)
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                for scale in (0.45, 0.65, 0.85):
                    half_w = max(2.0, (x2 - x1) * scale / 2.0)
                    half_h = max(2.0, (y2 - y1) * scale / 2.0)
                    candidates.append({"bbox": list(self._clip(
                        (cx - half_w, cy - half_h, cx + half_w, cy + half_h),
                        width, height)),
                        "method": f"seed_rescaled_no_component_{int(scale * 100)}",
                        "revision": revision})
                return self._dedupe(candidates)
            x1, y1, x2, y2 = (float(v) for v in seed_bbox)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            half_w = max(2.0, (x2 - x1) * scale / 2.0)
            half_h = max(2.0, (y2 - y1) * scale / 2.0)
            candidates.append({"bbox": list(self._clip(
                (cx - half_w, cy - half_h, cx + half_w, cy + half_h), width, height)),
                "method": method, "revision": revision})
        elif components:
            row = components[0]
            pad = max(4.0, 0.35 * max(row["bbox"][2] - row["bbox"][0],
                                      row["bbox"][3] - row["bbox"][1]))
            cx = (row["bbox"][0] + row["bbox"][2]) / 2.0
            cy = (row["bbox"][1] + row["bbox"][3]) / 2.0
            half = max(row["bbox"][2] - row["bbox"][0], row["bbox"][3] - row["bbox"][1]) / 2.0 + pad
            candidates.append({"bbox": list(self._clip((cx - half, cy - half,
                                                       cx + half, cy + half),
                                                      width, height)),
                               "method": "point_centred_padded_window",
                               "revision": revision})

        return self._dedupe(candidates)

    @staticmethod
    def _dedupe(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        for row in candidates:
            duplicate = False
            for old in kept:
                a, b = row["bbox"], old["bbox"]
                ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
                iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
                inter = ix * iy
                area_a = (a[2] - a[0]) * (a[3] - a[1])
                area_b = (b[2] - b[0]) * (b[3] - b[1])
                union = area_a + area_b - inter
                if union > 0 and inter / union >= 0.92:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(row)
        return kept
