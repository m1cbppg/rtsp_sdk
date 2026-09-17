#!/usr/bin/env python3
"""Compare the repository's garbage detectors on local PS/RTSP recordings.

This is a screening tool, not an accuracy benchmark: precision/recall require
manually labelled ground truth.  It samples frames, applies an optional ROI,
decodes the pinned DeepStream YOLO output ABI, and writes JSON plus annotated
sample images.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort


DEFAULT_ROI = [[0.08, 0.08], [0.94, 0.08], [0.98, 0.96], [0.08, 0.96]]


def letterbox(image: np.ndarray, size: int = 640) -> tuple[np.ndarray, float, int, int]:
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, scale, left, top


def inside(point: tuple[float, float], polygon: list[list[float]]) -> bool:
    return cv2.pointPolygonTest(
        np.asarray(polygon, dtype=np.float32), point, False
    ) >= 0


def decode(
    output: np.ndarray,
    labels: list[str],
    confidence: float,
    iou: float,
    roi: list[list[float]],
    scale: float,
    left: int,
    top: int,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    rows = np.asarray(output).reshape(-1, 6)
    candidates: list[dict[str, Any]] = []
    boxes: list[list[int]] = []
    scores: list[float] = []
    for x1, y1, x2, y2, score, class_id in rows:
        score = float(score)
        class_id = int(round(float(class_id)))
        if score < confidence or not 0 <= class_id < len(labels):
            continue
        # The exported models use x1,y1,x2,y2 in the 640 letterboxed image.
        x1 = max(0.0, min(float(width), (float(x1) - left) / scale))
        y1 = max(0.0, min(float(height), (float(y1) - top) / scale))
        x2 = max(0.0, min(float(width), (float(x2) - left) / scale))
        y2 = max(0.0, min(float(height), (float(y2) - top) / scale))
        if x2 <= x1 or y2 <= y1:
            continue
        nx, ny = ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height)
        if not inside((nx, ny), roi):
            continue
        boxes.append([round(x1), round(y1), round(x2 - x1), round(y2 - y1)])
        scores.append(score)
        candidates.append(
            {"label": labels[class_id], "confidence": round(score, 4),
             "box": [round(x1), round(y1), round(x2), round(y2)]}
        )
    keep = cv2.dnn.NMSBoxes(boxes, scores, confidence, iou)
    indices = {int(i) for i in np.asarray(keep).reshape(-1)} if len(keep) else set()
    return [item for i, item in enumerate(candidates) if i in indices]


def global_nms(detections: list[dict[str, Any]], iou: float = 0.45) -> list[dict[str, Any]]:
    """Suppress duplicate detections where overlapping tiles see one object."""
    if not detections:
        return []
    boxes = [[x["box"][0], x["box"][1], x["box"][2] - x["box"][0],
              x["box"][3] - x["box"][1]] for x in detections]
    scores = [float(x["confidence"]) for x in detections]
    keep = cv2.dnn.NMSBoxes(boxes, scores, 0.0, iou)
    return [detections[int(i)] for i in np.asarray(keep).reshape(-1)] if len(keep) else []


def tiles_for_frame(frame: np.ndarray, enabled: bool) -> list[tuple[np.ndarray, int, int]]:
    if not enabled:
        return [(frame, 0, 0)]
    h, w = frame.shape[:2]
    # 2x2 overlapping crops preserve roughly twice the linear detail of a
    # single 640px full-frame resize while retaining enough context.
    crop_w, crop_h = w // 2 + w // 8, h // 2 + h // 8
    xs = [0, w - crop_w]
    ys = [0, h - crop_h]
    return [(frame[y : y + crop_h, x : x + crop_w], x, y) for y in ys for x in xs]


def run_model(model_path: Path, labels_path: Path, videos: list[Path], output_dir: Path,
              sample_seconds: float, confidence: float, roi: list[list[float]],
              tiled: bool = False) -> dict[str, Any]:
    labels = [line.strip() for line in labels_path.read_text().splitlines() if line.strip()]
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    model_result: dict[str, Any] = {"model": str(model_path), "videos": {}}
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if frame_count else 0.0
        stride = max(1, round(fps * sample_seconds))
        frames: list[dict[str, Any]] = []
        counts: dict[str, int] = {}
        confidences: dict[str, list[float]] = {}
        index = 0
        saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % stride:
                index += 1
                continue
            detections = []
            for crop, offset_x, offset_y in tiles_for_frame(frame, tiled):
                image, scale, left, top = letterbox(crop)
                blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (640, 640), swapRB=True)
                output = session.run(None, {input_name: blob})[0]
                # Tile-local filtering is deliberately disabled; the final
                # center is checked against the global ROI after remapping.
                tile_detections = decode(
                    output, labels, confidence, 0.45,
                    [[0, 0], [1, 0], [1, 1], [0, 1]],
                    scale, left, top, crop.shape[1], crop.shape[0]
                )
                for item in tile_detections:
                    x1, y1, x2, y2 = item["box"]
                    item["box"] = [x1 + offset_x, y1 + offset_y,
                                    x2 + offset_x, y2 + offset_y]
                    cx = (item["box"][0] + item["box"][2]) / 2 / frame.shape[1]
                    cy = (item["box"][1] + item["box"][3]) / 2 / frame.shape[0]
                    if inside((cx, cy), roi):
                        detections.append(item)
            if tiled:
                detections = global_nms(detections)
            for item in detections:
                counts[item["label"]] = counts.get(item["label"], 0) + 1
                confidences.setdefault(item["label"], []).append(item["confidence"])
            frames.append({"time_s": round(index / fps, 2), "detections": detections})
            if detections and saved < 12:
                annotated = frame.copy()
                points = np.asarray([[round(x * frame.shape[1]), round(y * frame.shape[0])] for x, y in roi])
                cv2.polylines(annotated, [points], True, (0, 255, 255), 3)
                for item in detections:
                    x1, y1, x2, y2 = item["box"]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    cv2.putText(annotated, f'{item["label"]} {item["confidence"]:.2f}',
                                (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 200, 0), 2, cv2.LINE_AA)
                image_path = output_dir / f"{video.stem}_{saved:02d}.jpg"
                cv2.imwrite(str(image_path), annotated)
                saved += 1
            index += 1
        cap.release()
        model_result["videos"][video.name] = {
            "duration_s": round(duration, 2), "sampled_frames": len(frames),
            "frames_with_detections": sum(bool(x["detections"]) for x in frames),
            "detection_count": sum(counts.values()), "counts_by_label": counts,
            "mean_confidence_by_label": {
                k: round(float(np.mean(v)), 4) for k, v in confidences.items()
            }, "frames": frames,
        }
    return model_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--videos", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sample-seconds", type=float, default=2.0)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--tiled", action="store_true", help="2x2 overlapping high-detail crops")
    parser.add_argument("--roi", type=Path, help="JSON list of normalized [x,y] points")
    args = parser.parse_args()
    roi = json.loads(args.roi.read_text()) if args.roi else DEFAULT_ROI
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args.model, args.labels, args.videos, args.output_dir,
                       args.sample_seconds, args.confidence, roi, args.tiled)
    result["roi"] = roi
    result["confidence_threshold"] = args.confidence
    result["tiled"] = args.tiled
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"model": str(args.model), "report": str(args.report),
                      "videos": {k: {x: v for x, v in value.items() if x != "frames"}
                                 for k, value in result["videos"].items()}},
               ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
