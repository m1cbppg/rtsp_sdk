#!/usr/bin/env python3
"""Replay V3 candidates through an occlusion-aware event lifecycle.

The replay is offline and read-only with respect to the Clean Reference.  It
uses the frozen V3 candidates, recomputes per-frame protected residual support,
and runs the existing actor detector only while candidates or live events are
present.  The result answers whether the event-memory design survives a real
timeline rather than only representative review frames.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rtsp_annotator.ground_litter_detection import (  # noqa: E402
    UltralyticsGroundLitterDetector,
)
from rtsp_annotator.ground_litter_geometry import box_overlap_fraction  # noqa: E402
from run_ground_litter_clean_temporal_poc import (  # noqa: E402
    ANALYSIS_FPS,
    T01_REGION,
    iter_sampled_frames,
    read_raw_jsonl,
    t01_metrics,
    yolo_options,
)
from run_ground_litter_clean_temporal_v2 import (  # noqa: E402
    SUPPORT_LUMINANCE_THRESHOLD,
    SUPPORT_SIGNATURE_THRESHOLD,
    residual_maps,
)
from run_ground_litter_event_memory_v31 import (  # noqa: E402
    CONTEXT_OCCLUSION_EXTERNAL_AREA_PX,
    box_center,
    measure_context_support,
    same_fixed_view_anchor,
)
from run_ground_litter_prior_region_v3 import protected_normalize  # noqa: E402


EVENT_MATCH_DISTANCE_PX = 30.0
EVENT_MATCH_SIZE_RATIO = 5.0
EVENT_MERGE_DISTANCE_PX = 20.0
EVENT_MERGE_SIZE_RATIO = 10.0
CLEAR_CONFIRM_SECONDS = 5.0
CLEAN_MAX_SUPPORT_PIXELS = 2
ANCHOR_SUPPORT_PAD_PX = 4
CONFIRM_VISIBLE_SECONDS = 5.0
PENDING_EXPIRE_SECONDS = 15.0


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def max_actor_overlap(
    candidate: Iterable[float], actors: Iterable[Iterable[float]]
) -> float:
    return max(
        (box_overlap_fraction(candidate, actor) for actor in actors),
        default=0.0,
    )


def anchor_support_pixels(
    support: np.ndarray,
    valid: np.ndarray,
    box: Iterable[float],
    *,
    pad_px: int = ANCHOR_SUPPORT_PAD_PX,
) -> int:
    height, width = support.shape
    left, top, right, bottom = (int(round(value)) for value in box)
    left = max(0, left - pad_px)
    top = max(0, top - pad_px)
    right = min(width, right + pad_px)
    bottom = min(height, bottom + pad_px)
    if right <= left or bottom <= top:
        return 0
    return int(np.count_nonzero((support[top:bottom, left:right] > 0)
                                & (valid[top:bottom, left:right] > 0)))


@dataclass
class OnlineEvent:
    event_id: int
    first_seen: float
    anchor_box: list[float]
    state: str = "ANOMALY_PENDING"
    last_visible: float | None = None
    visible_timestamps: list[float] = field(default_factory=list)
    boxes: list[list[float]] = field(default_factory=list)
    state_history: list[dict[str, Any]] = field(default_factory=list)
    clear_observed_seconds: float = 0.0
    occluded_samples: int = 0
    uncertain_samples: int = 0
    confirmed_at: float | None = None
    closed_at: float | None = None
    closed_reason: str | None = None

    def transition(self, timestamp: float, state: str, reason: str) -> None:
        if self.state != state or not self.state_history:
            self.state_history.append({
                "timestamp": round(float(timestamp), 3),
                "state": state,
                "reason": reason,
            })
        self.state = state

    def observe(self, timestamp: float, box: Iterable[float], sample_period: float) -> None:
        values = [float(value) for value in box]
        self.boxes.append(values)
        self.boxes = self.boxes[-30:]
        self.anchor_box = np.median(np.asarray(self.boxes), axis=0).tolist()
        self.visible_timestamps.append(float(timestamp))
        self.last_visible = float(timestamp)
        self.clear_observed_seconds = 0.0
        if self.confirmed_at is None and len(set(self.visible_timestamps)) * sample_period >= CONFIRM_VISIBLE_SECONDS:
            self.confirmed_at = float(timestamp)
        self.transition(timestamp, "VISIBLE_ANOMALY", "candidate_visible")

    def absorb_same_frame_component(self, box: Iterable[float]) -> None:
        """Fold an additional nearby component into the current event."""
        self.boxes.append([float(value) for value in box])
        self.boxes = self.boxes[-30:]
        self.anchor_box = np.median(np.asarray(self.boxes), axis=0).tolist()

    def absorb_event(self, other: "OnlineEvent", timestamp: float) -> None:
        self.first_seen = min(self.first_seen, other.first_seen)
        self.boxes.extend(other.boxes)
        self.boxes = self.boxes[-60:]
        self.anchor_box = np.median(np.asarray(self.boxes), axis=0).tolist()
        self.visible_timestamps = sorted(set(
            self.visible_timestamps + other.visible_timestamps
        ))
        visible = [value for value in (self.last_visible, other.last_visible) if value is not None]
        self.last_visible = max(visible) if visible else None
        confirmed = [value for value in (self.confirmed_at, other.confirmed_at) if value is not None]
        self.confirmed_at = min(confirmed) if confirmed else None
        self.occluded_samples += other.occluded_samples
        self.uncertain_samples += other.uncertain_samples
        self.state_history.append({
            "timestamp": round(float(timestamp), 3),
            "state": self.state,
            "reason": f"merged_event_{other.event_id}",
        })

    def as_dict(self, sample_period: float) -> dict[str, Any]:
        visible_seconds = len(set(self.visible_timestamps)) * sample_period
        return {
            "event_id": self.event_id,
            "first_seen": round(self.first_seen, 3),
            "last_visible": None if self.last_visible is None else round(self.last_visible, 3),
            "anchor_box": [round(value, 2) for value in self.anchor_box],
            "state": self.state,
            "visible_hits": len(set(self.visible_timestamps)),
            "visible_duration_seconds": round(visible_seconds, 3),
            "confirmed_at": None if self.confirmed_at is None else round(self.confirmed_at, 3),
            "confirmation_delay_seconds": (
                None if self.confirmed_at is None
                else round(self.confirmed_at - self.first_seen, 3)
            ),
            "closed_at": None if self.closed_at is None else round(self.closed_at, 3),
            "closed_reason": self.closed_reason,
            "clear_observed_seconds": round(self.clear_observed_seconds, 3),
            "occluded_samples": self.occluded_samples,
            "uncertain_samples": self.uncertain_samples,
            "state_history": self.state_history,
            "visible_timestamps": [round(value, 3) for value in self.visible_timestamps],
        }


class OnlineEventMemory:
    def __init__(
        self,
        *,
        sample_fps: float = ANALYSIS_FPS,
        actor_overlap_threshold: float = 0.2,
    ) -> None:
        self.sample_period = 1.0 / sample_fps
        self.actor_overlap_threshold = actor_overlap_threshold
        self.events: list[OnlineEvent] = []
        self._next_id = 1

    @property
    def active(self) -> list[OnlineEvent]:
        return [event for event in self.events if event.closed_at is None]

    def update(
        self,
        *,
        timestamp: float,
        candidates: list[dict[str, Any]],
        support: np.ndarray,
        valid: np.ndarray,
        actors: list[list[float]],
        environment_state: str,
    ) -> dict[str, Any]:
        if environment_state == "ENVIRONMENT_CHANGE":
            for event in self.active:
                event.transition(timestamp, "ENVIRONMENT_CHANGE", "frame_unavailable")
            return {"visible": 0, "context_blocked": 0, "actor_blocked": 0}

        visible_candidates = []
        context_blocked = []
        actor_blocked = []
        for row in candidates:
            context = measure_context_support(support, valid, row["box"])
            if context["touching_external_support_px"] >= CONTEXT_OCCLUSION_EXTERNAL_AREA_PX:
                context_blocked.append(row)
                continue
            if max_actor_overlap(row["box"], actors) > self.actor_overlap_threshold:
                actor_blocked.append(row)
                continue
            visible_candidates.append(row)

        pairs = []
        active = self.active
        for event_index, event in enumerate(active):
            for candidate_index, candidate in enumerate(visible_candidates):
                if not same_fixed_view_anchor(
                    event.anchor_box,
                    candidate["box"],
                    center_distance_px=EVENT_MATCH_DISTANCE_PX,
                    size_ratio=EVENT_MATCH_SIZE_RATIO,
                ):
                    continue
                pairs.append((
                    math.dist(box_center(event.anchor_box), box_center(candidate["box"])),
                    event_index,
                    candidate_index,
                ))
        used_events: set[int] = set()
        used_candidates: set[int] = set()
        for _distance, event_index, candidate_index in sorted(pairs):
            if event_index in used_events or candidate_index in used_candidates:
                continue
            active[event_index].observe(
                timestamp, visible_candidates[candidate_index]["box"], self.sample_period
            )
            used_events.add(event_index)
            used_candidates.add(candidate_index)

        # A single object may split into multiple nearby connected components
        # in one frame.  Absorb additional components into an existing event
        # after its primary one-to-one match instead of creating duplicates.
        for candidate_index, candidate in enumerate(visible_candidates):
            if candidate_index in used_candidates:
                continue
            related = [
                (
                    math.dist(box_center(event.anchor_box), box_center(candidate["box"])),
                    event,
                )
                for event in active
                if same_fixed_view_anchor(
                    event.anchor_box,
                    candidate["box"],
                    center_distance_px=EVENT_MATCH_DISTANCE_PX,
                    size_ratio=EVENT_MATCH_SIZE_RATIO,
                )
            ]
            if not related:
                continue
            _distance, event = min(related, key=lambda pair: pair[0])
            event.absorb_same_frame_component(candidate["box"])
            used_candidates.add(candidate_index)

        for candidate_index, candidate in enumerate(visible_candidates):
            if candidate_index in used_candidates:
                continue
            event = OnlineEvent(
                event_id=self._next_id,
                first_seen=float(timestamp),
                anchor_box=[float(value) for value in candidate["box"]],
            )
            event.observe(timestamp, candidate["box"], self.sample_period)
            self.events.append(event)
            self._next_id += 1

        # Re-read active because new events were appended; only pre-existing,
        # unmatched events require a missing-observation transition.
        for event_index, event in enumerate(active):
            if event_index in used_events:
                continue
            context = measure_context_support(support, valid, event.anchor_box)
            context_occluded = (
                context["touching_external_support_px"]
                >= CONTEXT_OCCLUSION_EXTERNAL_AREA_PX
            )
            actor_occluded = (
                max_actor_overlap(event.anchor_box, actors)
                > self.actor_overlap_threshold
            )
            if context_occluded or actor_occluded:
                event.occluded_samples += 1
                event.clear_observed_seconds = 0.0
                event.transition(
                    timestamp,
                    "OCCLUDED",
                    "context_occluded" if context_occluded else "actor_occluded",
                )
                continue
            if (
                event.confirmed_at is None
                and event.last_visible is not None
                and timestamp - event.last_visible >= PENDING_EXPIRE_SECONDS
            ):
                event.closed_at = float(timestamp)
                event.closed_reason = "pending_timeout"
                event.transition(timestamp, "EXPIRED_PENDING", "pending_timeout")
                continue
            support_pixels = anchor_support_pixels(support, valid, event.anchor_box)
            if support_pixels <= CLEAN_MAX_SUPPORT_PIXELS:
                event.clear_observed_seconds += self.sample_period
                event.transition(timestamp, "CLEAN_PENDING", "clean_reference_match")
                if event.clear_observed_seconds >= CLEAR_CONFIRM_SECONDS:
                    event.closed_at = float(timestamp)
                    event.closed_reason = "clean_confirmed"
                    event.transition(timestamp, "CLEARED", "clean_confirmed")
                continue
            event.uncertain_samples += 1
            event.clear_observed_seconds = 0.0
            event.transition(timestamp, "ANOMALY_PENDING", "residual_without_candidate")

        self._merge_converged_events(timestamp)

        return {
            "visible": len(visible_candidates),
            "context_blocked": len(context_blocked),
            "actor_blocked": len(actor_blocked),
        }

    def _merge_converged_events(self, timestamp: float) -> None:
        active = sorted(self.active, key=lambda event: event.event_id)
        consumed: set[int] = set()
        for index, primary in enumerate(active):
            if primary.event_id in consumed:
                continue
            for secondary in active[index + 1:]:
                if secondary.event_id in consumed:
                    continue
                if not same_fixed_view_anchor(
                    primary.anchor_box,
                    secondary.anchor_box,
                    center_distance_px=EVENT_MERGE_DISTANCE_PX,
                    size_ratio=EVENT_MERGE_SIZE_RATIO,
                ):
                    continue
                primary.absorb_event(secondary, timestamp)
                if (
                    primary.confirmed_at is None
                    and len(set(primary.visible_timestamps)) * self.sample_period
                    >= CONFIRM_VISIBLE_SECONDS
                ):
                    primary.confirmed_at = float(timestamp)
                secondary.closed_at = float(timestamp)
                secondary.closed_reason = f"merged_into:{primary.event_id}"
                secondary.transition(timestamp, "MERGED", f"merged_into:{primary.event_id}")
                consumed.add(secondary.event_id)


class ActorCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows: dict[float, list[list[float]]] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                self.rows[float(row["timestamp"])] = row["boxes"]

    def get(self, timestamp: float) -> list[list[float]] | None:
        return self.rows.get(float(timestamp))

    def add(self, timestamp: float, boxes: list[list[float]]) -> None:
        timestamp = float(timestamp)
        if timestamp in self.rows:
            return
        cleaned = [[round(float(value), 3) for value in box] for box in boxes]
        self.rows[timestamp] = cleaned
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"timestamp": timestamp, "boxes": cleaned}) + "\n")


def replay_video(
    *,
    name: str,
    video: Path,
    start: float,
    v3_directory: Path,
    output: Path,
    reference: np.ndarray,
    reference_valid: np.ndarray,
    detector: UltralyticsGroundLitterDetector,
    detector_options: Any,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    raw_candidates = read_raw_jsonl(v3_directory / "v3_raw_candidates.jsonl")
    summary = load_json(v3_directory / "summary.json")
    matrix = np.asarray(summary["alignment"]["matrix"], np.float64)
    height, width = reference_valid.shape
    aligned_reference = cv2.warpPerspective(
        reference, matrix, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    aligned_valid = cv2.warpPerspective(
        reference_valid, matrix, (width, height), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    memory = OnlineEventMemory(
        sample_fps=ANALYSIS_FPS,
        actor_overlap_threshold=float(detector_options.actor_overlap_threshold),
    )
    actor_cache = ActorCache(output / "actor_boxes.jsonl")
    timeline = []
    started = time.perf_counter()
    for index, (timestamp, frame) in enumerate(iter_sampled_frames(
        video, sample_fps=ANALYSIS_FPS, start=start
    )):
        normalized, usable, environment = protected_normalize(
            aligned_reference, frame, aligned_valid
        )
        signature, luminance = residual_maps(normalized, frame)
        support = (
            (signature >= SUPPORT_SIGNATURE_THRESHOLD)
            & (luminance >= SUPPORT_LUMINANCE_THRESHOLD)
            & (usable > 0)
        ).astype(np.uint8)
        candidates = raw_candidates.get(float(timestamp), [])
        actors = actor_cache.get(timestamp)
        if actors is None:
            actors = (
                detector.actor_boxes(frame, detector_options)
                if candidates or memory.active else []
            )
            actor_cache.add(timestamp, actors)
        counts = memory.update(
            timestamp=timestamp,
            candidates=candidates,
            support=support,
            valid=aligned_valid,
            actors=actors,
            environment_state=environment["state"],
        )
        timeline.append({
            "timestamp": timestamp,
            "raw_candidates": len(candidates),
            "actor_boxes": len(actors),
            "active_events": len(memory.active),
            "environment_state": environment["state"],
            **counts,
        })
        if index and index % 20 == 0:
            print(
                f"[{name}] t={timestamp:.0f}s frames={index + 1} "
                f"events={len(memory.events)} active={len(memory.active)}",
                flush=True,
            )

    events = [event.as_dict(memory.sample_period) for event in memory.events]
    (output / "events.json").write_text(
        json.dumps(events, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "timeline.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in timeline),
        encoding="utf-8",
    )
    result = {
        "name": name,
        "sample_fps": ANALYSIS_FPS,
        "evaluation_start_seconds": start,
        "frames": len(timeline),
        "events_total": len(events),
        "events_confirmed_5s": sum(row["confirmed_at"] is not None for row in events),
        "events_active_at_end": sum(row["closed_at"] is None for row in events),
        "context_blocked_candidates": sum(row["context_blocked"] for row in timeline),
        "actor_blocked_candidates": sum(row["actor_blocked"] for row in timeline),
        "environment_states": dict(
            (state, sum(row["environment_state"] == state for row in timeline))
            for state in sorted({row["environment_state"] for row in timeline})
        ),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    if name == "positive":
        matching = [
            row for row in events
            if not str(row.get("closed_reason") or "").startswith("merged_into:")
            if T01_REGION[0] <= box_center(row["anchor_box"])[0] <= T01_REGION[2]
            and T01_REGION[1] <= box_center(row["anchor_box"])[1] <= T01_REGION[3]
        ]
        result["t01"] = {
            "matching_event_ids": [row["event_id"] for row in matching],
            "first_seen": min((row["first_seen"] for row in matching), default=None),
            "first_confirmed_5s": min(
                (row["confirmed_at"] for row in matching if row["confirmed_at"] is not None),
                default=None,
            ),
            "visible_duration_seconds": sum(
                row["visible_duration_seconds"] for row in matching
            ),
        }
    (output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--v1-output", type=Path, required=True)
    parser.add_argument("--v3-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--litter-model", type=Path, required=True)
    parser.add_argument("--actor-model", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--videos",
        nargs="+",
        choices=("negative", "positive", "clean_holdout"),
        default=("negative", "positive", "clean_holdout"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    arrays = np.load(args.v1_output / "reference_arrays.npz")
    reference = arrays["reference"]
    reference_valid = cv2.imread(
        str(args.v3_output / "reference_valid_mask_v3.png"), cv2.IMREAD_GRAYSCALE
    )
    if reference_valid is None:
        raise FileNotFoundError(args.v3_output / "reference_valid_mask_v3.png")
    options = yolo_options(args.litter_model, args.actor_model)
    detector = UltralyticsGroundLitterDetector(
        model_path=args.litter_model,
        actor_model_path=args.actor_model,
        device=args.device,
    )
    video_config = {
        "negative": ("正常负样本.mp4", "negative"),
        "positive": ("小垃圾正样本.mp4", "positive"),
        "clean_holdout": ("clean_reference.mp4", "clean_holdout"),
    }
    results = {}
    for name in args.videos:
        filename, directory = video_config[name]
        v3_summary = load_json(args.v3_output / directory / "summary.json")
        results[name] = replay_video(
            name=name,
            video=args.input_dir / filename,
            start=float(v3_summary["evaluation_start_seconds"]),
            v3_directory=args.v3_output / directory,
            output=args.output / directory,
            reference=reference,
            reference_valid=reference_valid,
            detector=detector,
            detector_options=options,
        )
    payload = {
        "kind": "ground_litter_online_event_replay_v31",
        "parameters": {
            "event_match_distance_px": EVENT_MATCH_DISTANCE_PX,
            "event_match_size_ratio": EVENT_MATCH_SIZE_RATIO,
            "event_merge_distance_px": EVENT_MERGE_DISTANCE_PX,
            "event_merge_size_ratio": EVENT_MERGE_SIZE_RATIO,
            "context_external_area_px": CONTEXT_OCCLUSION_EXTERNAL_AREA_PX,
            "clear_confirm_seconds": CLEAR_CONFIRM_SECONDS,
            "clean_max_support_pixels": CLEAN_MAX_SUPPORT_PIXELS,
            "confirm_visible_seconds": CONFIRM_VISIBLE_SECONDS,
            "pending_expire_seconds": PENDING_EXPIRE_SECONDS,
            "actor_model": args.actor_model.name,
            "actor_overlap_threshold": float(options.actor_overlap_threshold),
            "clean_reference_updated": False,
        },
        "videos": results,
    }
    (args.output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
