#!/usr/bin/env python3
"""Offline 2 FPS replay of frozen candidate evidence through recovery policies."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rtsp_annotator.ground_litter_candidate_recovery import (  # noqa: E402
    CandidateEvidence, DeferredRecoveryTracker, RecoveryOptions,
    box_overlap_fraction, same_location,
)
import train_ground_litter_candidate_filter as candidate_filter  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float | None:
    return None if not values else float(np.percentile(np.asarray(values), q))


@dataclass(frozen=True, slots=True)
class VideoFrame:
    frame: np.ndarray
    time_seconds: float
    index: int


def iter_video_frames(path: Path):
    """Decode PS sequentially and rebase its arbitrary PTS to file-relative time."""
    import av
    with av.open(str(path), timeout=(10.0, 60.0)) as container:
        stream = next((item for item in container.streams if item.type == "video"), None)
        if stream is None:
            raise RuntimeError("recording has no video stream")
        stream.thread_type = "FRAME"
        stream.codec_context.thread_count = 0
        base = None
        for index, decoded in enumerate(container.decode(video=0)):
            timestamp = decoded.time
            if timestamp is None and decoded.pts is not None and stream.time_base:
                timestamp = float(decoded.pts * stream.time_base)
            value = float(timestamp) if timestamp is not None else None
            if value is not None and base is None:
                base = value
            relative = 0.0 if value is None or base is None else value - base
            yield VideoFrame(decoded.to_ndarray(format="bgr24"), relative, index)


def iou(left, right) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    b = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    return inter / max(1, a + b - inter)


def roi_mask(shape, polygon, dilation_fraction=0.0):
    h, w = shape
    points = np.asarray([[round(x * w), round(y * h)] for x, y in polygon], np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [points], 255)
    radius = round(min(h, w) * max(0.0, dilation_fraction))
    if radius:
        mask = cv2.dilate(
            mask, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2,
            ),
        )
    return mask


def detect_actors(model, frame, confidence):
    started = time.perf_counter()
    result = model.predict(
        frame, imgsz=1280, conf=confidence, classes=[0, 1, 2, 3, 5, 7],
        verbose=False, device=0,
    )[0]
    boxes = [tuple(map(float, row)) for row in result.boxes.xyxy.cpu().tolist()]
    return boxes, time.perf_counter() - started


def filter_actor_overlap(proposals, actors, threshold):
    kept, rejected = [], []
    for proposal in proposals:
        overlap = max((
            box_overlap_fraction(tuple(proposal["bbox"]), actor) for actor in actors
        ), default=0.0)
        if overlap >= threshold:
            rejected.append({**proposal, "actor_overlap": overlap})
        else:
            kept.append(proposal)
    return kept, rejected


def tiles_for_frame(frame, mask, size=768, overlap=.25):
    h, w = frame.shape[:2]
    step = max(1, round(size * (1 - overlap)))
    xs = list(range(0, max(1, w - size + 1), step)) + [max(0, w - size)]
    ys = list(range(0, max(1, h - size + 1), step)) + [max(0, h - size)]
    result = []
    for y in sorted(set(ys)):
        for x in sorted(set(xs)):
            patch_mask = mask[y:y + size, x:x + size]
            if patch_mask.size and np.count_nonzero(patch_mask) > patch_mask.size * .08:
                result.append((x, y, frame[y:y + size, x:x + size]))
    return result


def semantic_candidates(model, frame, mask, conf: float, limit: int):
    h, w = frame.shape[:2]
    started = time.perf_counter()
    full = model.predict(frame, conf=conf, imgsz=1280, verbose=False, device=0)[0]
    proposals = []
    for box, score, cls in zip(
        full.boxes.xyxy.cpu().numpy(), full.boxes.conf.cpu().numpy(),
        full.boxes.cls.cpu().numpy(),
    ):
        x1, y1, x2, y2 = map(float, box)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        if 0 <= cx < w and 0 <= cy < h and mask[cy, cx]:
            proposals.append({
                "bbox": [round(x1), round(y1), round(x2), round(y2)],
                "semantic_score": float(score), "class_id": int(cls),
                "class_name": model.names[int(cls)], "source": "semantic_full",
            })
    tiles = tiles_for_frame(frame, mask)
    for offset in range(0, len(tiles), 8):
        batch = tiles[offset:offset + 8]
        results = model.predict(
            [row[2] for row in batch], conf=conf, imgsz=768,
            verbose=False, device=0, batch=8,
        )
        for (ox, oy, _), result in zip(batch, results):
            for box, score, cls in zip(
                result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
            ):
                x1, y1, x2, y2 = box
                bbox = [round(x1 + ox), round(y1 + oy),
                        round(x2 + ox), round(y2 + oy)]
                cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
                if 0 <= cx < w and 0 <= cy < h and mask[cy, cx]:
                    proposals.append({
                        "bbox": bbox, "semantic_score": float(score),
                        "class_id": int(cls), "class_name": model.names[int(cls)],
                        "source": "semantic_tile",
                    })
    kept = []
    for proposal in sorted(proposals, key=lambda row: row["semantic_score"], reverse=True):
        if all(iou(proposal["bbox"], old["bbox"]) < .5 for old in kept):
            kept.append(proposal)
    return kept[:limit], len(kept), time.perf_counter() - started


def crop(frame, box, scale=1.8):
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(32.0, (x2 - x1) * scale, (y2 - y1) * scale)
    return frame[
        max(0, round(cy - side / 2)):min(h, round(cy + side / 2)),
        max(0, round(cx - side / 2)):min(w, round(cx + side / 2)),
    ]


def classify_candidates(model, frame, proposals, device, artifact, threshold, batch):
    import torch
    if not proposals:
        return [], 0.0
    patches = [crop(frame, row["bbox"]) for row in proposals]
    if any(patch.size == 0 for patch in patches):
        raise RuntimeError("empty classifier crop")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    vectors = []
    for offset in range(0, len(patches), batch):
        values = model.embed(
            source=patches[offset:offset + batch], imgsz=320, batch=batch,
            device=0, verbose=False,
        )
        vectors.extend(value.detach().float().cpu().numpy() for value in values)
    matrix = np.stack(vectors).astype(np.float32)
    scores = candidate_filter.camera_excluded_scores(
        matrix, [device] * len(proposals), artifact,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    decisions = []
    for rank, (proposal, score) in enumerate(zip(proposals, scores), 1):
        candidate_id = f"t{classify_candidates.tick_index:04d}r{rank:02d}"
        decisions.append({
            **proposal, "candidate_id": candidate_id,
            "classifier_score": float(score), "classifier_passed": bool(score >= threshold),
            "classifier_margin": float(score - threshold),
        })
    return decisions, elapsed


classify_candidates.tick_index = 0


def save_event_image(output: Path, frame, event_id: str, bbox, label: str):
    canvas = frame.copy()
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 4)
    cv2.putText(canvas, label, (max(0, x1), max(30, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 0, 255), 2, cv2.LINE_AA)
    height, width = canvas.shape[:2]
    if width > 1280:
        canvas = cv2.resize(canvas, (1280, round(height * 1280 / width)))
    path = output / "events" / f"{event_id}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return str(path.relative_to(output))


def merge_recovery_event(events, row, frame, output, options):
    """Merge repeated emissions at one location into one reviewable event card."""
    for existing in events:
        if existing["policy"] != row["policy"]:
            continue
        if same_location(tuple(existing["bbox"]), tuple(row["bbox"]), options):
            existing["last_file_offset_seconds"] = row["file_offset_seconds"]
            existing["occurrences"] += 1
            existing["near_anchor"] = existing["near_anchor"] or row["near_anchor"]
            existing["maximum_semantic_score"] = max(
                existing["maximum_semantic_score"], row["maximum_semantic_score"],
            )
            return existing
    event_id = f"{row['policy']}-{len(events):05d}"
    row["event_id"] = event_id
    row["first_file_offset_seconds"] = row["file_offset_seconds"]
    row["last_file_offset_seconds"] = row["file_offset_seconds"]
    row["occurrences"] = 1
    row["image"] = save_event_image(
        output, frame, event_id, row["bbox"],
        f"{row['policy']} {row['file_offset_seconds']:.2f}s",
    )
    events.append(row)
    return row


def parse_box(text: str | None):
    if not text:
        return None
    values = tuple(float(value) for value in text.split(","))
    if len(values) != 4:
        raise argparse.ArgumentTypeError("anchor bbox needs four comma-separated values")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--trash-model", required=True)
    parser.add_argument("--feature-model", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-offset", type=float, required=True)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--analysis-fps", type=float, default=2.0)
    parser.add_argument("--confidence", type=float, default=.025)
    parser.add_argument("--per-frame-limit", type=int, default=60)
    parser.add_argument("--classifier-batch", type=int, default=32)
    parser.add_argument("--roi-dilation", type=float, default=.08)
    parser.add_argument("--actor-filter", action="store_true")
    parser.add_argument("--actor-confidence", type=float, default=.30)
    parser.add_argument("--actor-overlap-threshold", type=float, default=.15)
    parser.add_argument("--anchor-bbox", default=None)
    parser.add_argument("--anchor-offset", type=float, default=None)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output already exists")
    if args.duration <= 0 or args.analysis_fps <= 0:
        raise SystemExit("duration and analysis FPS must be positive")
    args.output.mkdir(parents=True)
    identity = json.loads(args.identity.read_text())
    geometry = json.loads(args.geometry.read_text())
    device = identity["device_code"]
    if sha256(args.input) != identity["sha256"]:
        raise RuntimeError("recording SHA-256 mismatch")

    fit_path = args.artifact_dir / "FIT_REPORT.json"
    fit = json.loads(fit_path.read_text())
    report = fit["models"]["yolo26s"]
    artifact_path = args.artifact_dir / report["artifact"]
    if sha256(artifact_path) != report["artifact_sha256"]:
        raise RuntimeError("classifier artifact SHA-256 mismatch")
    loaded = np.load(artifact_path, allow_pickle=False)
    artifact = {key: loaded[key] for key in loaded.files}
    threshold = float(artifact["threshold"][0])
    if device not in set(artifact["held_out_devices"].astype(str)):
        raise RuntimeError("classifier artifact has no camera-excluded head for device")

    from ultralytics import YOLO
    trash_model = YOLO(args.trash_model)
    feature_model = YOLO(args.feature_model)
    policies = {
        "temporal_any": DeferredRecoveryTracker(RecoveryOptions(
            semantic_score_threshold=None,
        )),
        "temporal_high_semantic": DeferredRecoveryTracker(RecoveryOptions(
            semantic_score_threshold=.45,
        )),
    }
    anchor_bbox = parse_box(args.anchor_bbox)
    target_interval = 1.0 / args.analysis_fps
    next_target = args.start_offset
    end_offset = args.start_offset + args.duration
    ticks, events = [], []
    anchor_recovered_ticks = {name: 0 for name in policies}
    last_selected = -math.inf
    mask = None
    for decoded in iter_video_frames(args.input):
        position = decoded.time_seconds
        if position + 1e-6 < next_target:
            continue
        if position > end_offset + target_interval:
            break
        if position - last_selected < target_interval * .8:
            continue
        frame = decoded.frame
        if mask is None:
            mask = roi_mask(frame.shape[:2], geometry["roi"], args.roi_dilation)
        classify_candidates.tick_index = len(ticks)
        proposals, raw_count, semantic_seconds = semantic_candidates(
            trash_model, frame, mask, args.confidence, args.per_frame_limit,
        )
        actors, actor_seconds = ([], 0.0)
        actor_rejected = []
        if args.actor_filter:
            actors, actor_seconds = detect_actors(
                feature_model, frame, args.actor_confidence,
            )
            proposals, actor_rejected = filter_actor_overlap(
                proposals, actors, args.actor_overlap_threshold,
            )
        decisions, classifier_seconds = classify_candidates(
            feature_model, frame, proposals, device, artifact, threshold,
            args.classifier_batch,
        )
        evidence = [CandidateEvidence(
            row["candidate_id"], tuple(row["bbox"]), row["semantic_score"],
            row["classifier_score"], row["classifier_passed"],
        ) for row in decisions]
        policy_rows = {}
        state_started = time.perf_counter()
        for name, tracker in policies.items():
            observation = tracker.observe(position, evidence)
            policy_rows[name] = {
                "active_tracks": observation.active_tracks,
                "active_recovered_tracks": len(observation.active_recovered),
                "evicted_tracks": observation.evicted_tracks,
                "temporal_events": len(observation.temporal_recovered),
                "overlap_events": len(observation.overlap_recovered),
            }
            active_anchor = bool(
                anchor_bbox is not None
                and (args.anchor_offset is None
                     or abs(position - args.anchor_offset) <= 10.0)
                and any(same_location(
                    event.bbox, anchor_bbox, tracker.options,
                ) for event in observation.active_recovered)
            )
            policy_rows[name]["active_anchor_recovered"] = active_anchor
            if active_anchor:
                anchor_recovered_ticks[name] += 1
            # An overlapping passed proposal already reaches the ordinary state
            # machine.  Keep that as a diagnostic count; it is not an extra
            # recovered output and must not create a review card every tick.
            for event in observation.temporal_recovered:
                row = {
                    "policy": name, **asdict(event),
                    "file_offset_seconds": position,
                    "near_anchor": bool(
                        anchor_bbox is not None
                        and same_location(event.bbox, anchor_bbox, tracker.options)
                        and (args.anchor_offset is None
                             or abs(position - args.anchor_offset) <= 10.0)
                    ),
                }
                merge_recovery_event(events, row, frame, args.output, tracker.options)
        state_seconds = time.perf_counter() - state_started
        baseline_anchor = False
        if anchor_bbox is not None:
            baseline_anchor = any(
                row["classifier_passed"]
                and same_location(tuple(row["bbox"]), anchor_bbox,
                                  RecoveryOptions())
                for row in decisions
            )
        tick = {
            "tick": len(ticks), "file_offset_seconds": position,
            "source_frame_index": decoded.index, "raw_candidate_count": raw_count,
            "candidate_count": len(decisions),
            "actor_count": len(actors),
            "actor_rejected_count": len(actor_rejected),
            "passed_count": sum(row["classifier_passed"] for row in decisions),
            "filtered_count": sum(not row["classifier_passed"] for row in decisions),
            "baseline_anchor_passed": baseline_anchor,
            "semantic_seconds": semantic_seconds,
            "actor_seconds": actor_seconds,
            "classifier_seconds": classifier_seconds,
            "state_seconds": state_seconds,
            "policies": policy_rows,
            "decisions": decisions,
        }
        ticks.append(tick)
        last_selected = position
        while next_target <= position + 1e-6:
            next_target += target_interval
        print(json.dumps({
            "tick": len(ticks), "offset": round(position, 3),
            "candidates": len(decisions), "events": len(events),
            "semantic_s": round(semantic_seconds, 3),
            "classifier_s": round(classifier_seconds, 3),
        }), flush=True)

    if not ticks:
        raise RuntimeError("no frames sampled from requested window")
    classifier_times = [row["classifier_seconds"] for row in ticks]
    semantic_times = [row["semantic_seconds"] for row in ticks]
    result = {
        "schema": "ground_litter_deferred_recovery_replay_v1",
        "input": {
            "recording_identity": identity,
            "geometry_sha256": sha256(args.geometry),
            "fit_report_sha256": sha256(fit_path),
            "artifact_sha256": sha256(artifact_path),
            "trash_model_sha256": sha256(Path(args.trash_model)),
            "feature_model_sha256": sha256(Path(args.feature_model)),
            "start_offset": args.start_offset, "duration": args.duration,
            "analysis_fps": args.analysis_fps,
        },
        "configuration": {
            "confidence": args.confidence, "per_frame_limit": args.per_frame_limit,
            "roi_dilation": args.roi_dilation,
            "actor_filter": args.actor_filter,
            "actor_confidence": args.actor_confidence,
            "actor_overlap_threshold": args.actor_overlap_threshold,
            "classifier_threshold": threshold,
            "recovery": {name: asdict(tracker.options) for name, tracker in policies.items()},
        },
        "summary": {
            "ticks": len(ticks), "events": len(events),
            "near_anchor_events": sum(row["near_anchor"] for row in events),
            "baseline_anchor_ticks": sum(row["baseline_anchor_passed"] for row in ticks),
            "anchor_recovered_ticks": anchor_recovered_ticks,
            "maximum_active_tracks": {
                name: max(row["policies"][name]["active_tracks"] for row in ticks)
                for name in policies
            },
            "classifier_seconds_p95": percentile(classifier_times, 95),
            "classifier_seconds_p99": percentile(classifier_times, 99),
            "semantic_seconds_p50": percentile(semantic_times, 50),
            "semantic_seconds_p95": percentile(semantic_times, 95),
            "actor_rejected_total": sum(row["actor_rejected_count"] for row in ticks),
            "actor_seconds_p95": percentile(
                [row["actor_seconds"] for row in ticks], 95,
            ),
        },
        "events": events,
        "ticks": ticks,
    }
    (args.output / "REPLAY_RESULTS.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
    )
    gc.collect()
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
