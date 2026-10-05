"""Production-scale tiled inference for Rapid Dataset v2 (spec §5/§6).

The exact contract used by the four-camera exploratory probe and required by the
v2 spec:

* source-native 2560x1440 is tiled into 640x640 windows at 512-pixel stride
  (128-pixel overlap) with full right/bottom coverage — never resized;
* only ROI-intersecting tiles are inferred, and a predicted box centre must lie
  inside the ROI;
* per-class NMS at IoU 0.50 across tiles;
* a loose per-model confidence floor (0.01); the two model confidences are
  never compared against each other.

cv2/numpy are imported lazily so this module can be imported on a machine with
no working cv2 for the pure tile-plan tests.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

from rtsp_annotator.ground_litter_historical import (
    CANDIDATE_CONF_FLOOR, CROSS_MODEL_MATCH_IOU, NMS_IOU, STRIDE, TILE,
    bbox_center, bbox_iou, center_distance,
)


def tile_starts(length: int, *, tile: int = TILE, stride: int = STRIDE) -> list[int]:
    """Starts along one axis with full trailing coverage, no duplicates."""
    if length < tile:
        raise ValueError(f"source axis {length} is smaller than the {tile} tile")
    starts = list(range(0, length - tile + 1, stride))
    if not starts or starts[-1] != length - tile:
        starts.append(length - tile)
    return sorted(set(starts))


def tile_plan(width: int, height: int, *, tile: int = TILE,
              stride: int = STRIDE) -> list[tuple[int, int]]:
    return [(x, y) for y in tile_starts(height, tile=tile, stride=stride)
            for x in tile_starts(width, tile=tile, stride=stride)]


def nms_boxes(boxes: Sequence[dict[str, Any]], *, iou_threshold: float = NMS_IOU) -> list[dict[str, Any]]:
    """Per-class greedy NMS; identical to the exploratory probe's rule."""
    keep: list[dict[str, Any]] = []
    for row in sorted(boxes, key=lambda item: float(item["confidence"]), reverse=True):
        if all(int(row["class_id"]) != int(old["class_id"])
               or bbox_iou(row["xyxy"], old["xyxy"]) <= iou_threshold for old in keep):
            keep.append(dict(row))
    return keep


def merge_cross_model_candidates(
    rows: Iterable[dict[str, Any]], *, iou_threshold: float = CROSS_MODEL_MATCH_IOU,
) -> list[dict[str, Any]]:
    """Fuse per-model boxes into model-agnostic observations with source support.

    Boxes from different models are matched one-to-one by IoU or, for tiny
    targets, by centre distance.  The observation keeps the box of the more
    confident source and records every source's confidence, so downstream
    priority logic can distinguish ``turhancan_only`` from ``both`` without
    ever comparing the two models' confidences.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["source"]), []).append(dict(row))
    sources = sorted(grouped)
    observations: list[dict[str, Any]] = []

    claimed: set[int] = set()
    primary = sources[0] if sources else None
    if primary is None:
        return observations
    for base in sorted(grouped[primary], key=lambda item: -float(item["confidence"])):
        support = {primary: base}
        for other in sources[1:]:
            best_index, best_score = None, None
            for index, candidate in enumerate(grouped[other]):
                if index in claimed:
                    continue
                iou = bbox_iou(base["xyxy"], candidate["xyxy"])
                short = min(abs(base["xyxy"][2] - base["xyxy"][0]),
                            abs(base["xyxy"][3] - base["xyxy"][1]))
                tolerance = max(8.0, 0.5 * short)
                distance = center_distance(base["xyxy"], candidate["xyxy"])
                if iou < iou_threshold and distance > tolerance:
                    continue
                score = (1.0 - iou, distance)
                if best_score is None or score < best_score:
                    best_index, best_score = index, score
            if best_index is not None:
                support[other] = grouped[other][best_index]
                claimed.add(best_index)
        winner = max(support.values(), key=lambda item: float(item["confidence"]))
        observations.append({
            "bbox_xyxy": [float(v) for v in winner["xyxy"]],
            "bbox_by_source": {name: [float(v) for v in box["xyxy"]] for name, box in support.items()},
            "confidence_by_source": {name: float(box["confidence"]) for name, box in support.items()},
            "class_name_by_source": {name: str(box.get("class_name", "")) for name, box in support.items()},
            "class_id_by_source": {name: int(box.get("class_id", -1)) for name, box in support.items()},
            "tile_xy_by_source": {name: list(box.get("tile_xy", [])) for name, box in support.items()},
        })
    # any unmatched box from a secondary source becomes its own observation
    for other in sources[1:]:
        for index, candidate in enumerate(grouped[other]):
            if index in claimed:
                continue
            observations.append({
                "bbox_xyxy": [float(v) for v in candidate["xyxy"]],
                "bbox_by_source": {other: [float(v) for v in candidate["xyxy"]]},
                "confidence_by_source": {other: float(candidate["confidence"])},
                "class_name_by_source": {other: str(candidate.get("class_name", ""))},
                "class_id_by_source": {other: int(candidate.get("class_id", -1))},
                "tile_xy_by_source": {other: list(candidate.get("tile_xy", []))},
            })
    observations.sort(key=lambda item: (-max(float(v) for v in item["confidence_by_source"].values()),
                                        item["bbox_xyxy"][1], item["bbox_xyxy"][0]))
    return observations


def _cv2():
    import cv2  # noqa: PLC0415
    return cv2


def _np():
    import numpy as np  # noqa: PLC0415
    return np


def roi_polygon_pixels(roi: Sequence[Sequence[float]], width: int, height: int):
    np = _np()
    return np.array([(round(float(x) * width), round(float(y) * height)) for x, y in roi],
                    dtype=np.int32)


def roi_mask(roi: Sequence[Sequence[float]], width: int, height: int):
    cv2 = _cv2()
    np = _np()
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [roi_polygon_pixels(roi, width, height)], 255)
    return mask


def point_inside_roi(roi: Sequence[Sequence[float]], x: float, y: float,
                     width: int, height: int) -> bool:
    cv2 = _cv2()
    return cv2.pointPolygonTest(roi_polygon_pixels(roi, width, height),
                                (float(x), float(y)), False) >= 0


def box_truncated_by_tile(box: Sequence[float], tile_origin: tuple[int, int],
                          *, tile: int = TILE, margin: float = 1.0) -> bool:
    """True when a predicted box touches the tile edge it was cut out of."""
    x, y = tile_origin
    return (abs(float(box[0]) - x) <= margin or abs(float(box[1]) - y) <= margin
            or abs(float(box[2]) - (x + tile)) <= margin
            or abs(float(box[3]) - (y + tile)) <= margin)


def predict_frame(model, frame, mask, roi, *, tile: int = TILE, stride: int = STRIDE,
                  conf: float = CANDIDATE_CONF_FLOOR, batch: int = 4,
                  device: str = "cpu", rois: Sequence[Sequence[float]] | None = None,
                  width: int | None = None, height: int | None = None) -> dict[str, Any]:
    """Run one model over the ROI-intersecting tiles of one source-native frame."""
    np = _np()
    height = int(frame.shape[0]) if height is None else int(height)
    width = int(frame.shape[1]) if width is None else int(width)
    plan = [(x, y) for x in tile_starts(width, tile=tile, stride=stride)
            for y in tile_starts(height, tile=tile, stride=stride)
            if np.any(mask[y:y + tile, x:x + tile])]
    names = getattr(model, "names", {})
    raw: list[dict[str, Any]] = []
    for base in range(0, len(plan), max(1, batch)):
        part = plan[base:base + max(1, batch)]
        results = model.predict([frame[y:y + tile, x:x + tile] for x, y in part],
                                imgsz=tile, conf=conf, iou=0.7, device=device, verbose=False)
        for (x, y), result in zip(part, results):
            for box in result.boxes:
                local = box.xyxy[0].cpu().tolist()
                coords = [round(local[0] + x, 2), round(local[1] + y, 2),
                          round(local[2] + x, 2), round(local[3] + y, 2)]
                cx, cy = bbox_center(coords)
                if not (0 <= cx < width and 0 <= cy < height):
                    continue
                if not mask[int(round(cy)), int(round(cx))]:
                    continue
                class_id = int(box.cls[0])
                raw.append({
                    "xyxy": coords,
                    "confidence": round(float(box.conf[0]), 5),
                    "class_id": class_id,
                    "class_name": str(names.get(class_id, class_id) if isinstance(names, dict)
                                      else names[class_id]),
                    "tile_xy": [int(x), int(y)],
                    "tile_truncated": box_truncated_by_tile(coords, (x, y), tile=tile),
                })
    return {"tile_count": len(plan), "raw_count": len(raw), "boxes": nms_boxes(raw)}
