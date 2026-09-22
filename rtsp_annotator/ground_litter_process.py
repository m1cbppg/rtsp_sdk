"""Lossy side process that runs the ground-litter model off the main RTSP path.

The main DeepStream pipeline copies one native-resolution frame every
``1 / analysis_fps`` seconds into this process.  Everything expensive (tiled
inference, actor overlap checks, display smoothing) happens here, so a slow
litter model can never back-pressure decode, OSD or NVENC.
"""

from __future__ import annotations

import logging
import multiprocessing
import queue
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .event_engine import NormalizedRect

from .ground_litter_detection import (
    GroundLitterDetectionOptions,
    GroundLitterDisplayTracker,
    GroundLitterResultCache,
    GroundLitterSnapshot,
    UltralyticsGroundLitterDetector,
    as_bgr,
    build_ground_litter_tiles,
)
from .ground_litter_v32 import (
    CleanReferenceProfileV32,
    CleanReferenceV32Processor,
    analyze_prior_frame,
    propose_prior_candidates,
)
from .ground_litter_v33 import (
    PriorObservation,
    SemanticObservation,
    V33EventMemory,
    crop_semantic_observations,
    same_target,
)


LOGGER = logging.getLogger("rtsp_annotator.ground_litter_process")


@dataclass(frozen=True, slots=True)
class GroundLitterProcessConfig:
    model_path: Path
    device: str
    half: bool
    actor_model_path: Path | None
    options_by_pad: dict[int, GroundLitterDetectionOptions]
    profile_root: Path | None = None


def _put_latest(output_queue: Any, item: Any) -> None:
    """Keep only the newest snapshot; an old box is worse than no box."""
    try:
        output_queue.put_nowait(item)
        return
    except queue.Full:
        pass
    try:
        output_queue.get_nowait()
    except queue.Empty:
        pass
    try:
        output_queue.put_nowait(item)
    except queue.Full:
        pass


class _HybridPadState:
    """Per-pad dual-channel state for ``mode=hybrid_v33``.

    Holds the Clean Reference alignment cache, the prior channel's environment
    stability streak and the shared V3.3 event memory. Both channels write into
    that one memory so fusion happens at the event layer, never by unioning
    per-frame boxes.
    """

    def __init__(
        self,
        options: GroundLitterDetectionOptions,
        profile: CleanReferenceProfileV32,
    ) -> None:
        self.options = options
        self.profile = profile
        reference = profile.reference
        self.pixel_scale = reference.shape[1] / 2560.0
        self.memory = V33EventMemory(options, pixel_scale=self.pixel_scale)
        self.memory.set_reference_size(reference.shape[1], reference.shape[0])
        self.aligned: Any = None
        self.alignment: dict[str, Any] = {}
        self.last_full_scan_at: float | None = None
        self.normal_streak = 0
        self.masks: dict[str, np.ndarray] | None = None
        self.tiles: list[tuple[int, int, int, int]] | None = None
        self.tile_size: tuple[int, int] | None = None
        self.semantic_runs_full = 0
        self.semantic_runs_crop = 0
        self.result_version = 0
        self.prior_degraded = False
        self.semantic_degraded = False
        self.prior_error = ""
        self.semantic_error = ""


def _select_crop_priors(
    priors: list[PriorObservation],
    options: GroundLitterDetectionOptions,
    memory: V33EventMemory,
    pixel_scale: float,
) -> list[PriorObservation]:
    """Pick the upstream candidates worth a prior-guided crop this tick.

    Unconfirmed proposals rank ahead of ones already attached to a confirmed
    event, then by anomaly score: checking a brand-new target is more useful
    than re-checking a target the pipeline already tracks.
    """
    limit = int(options.prior_crop_maximum)
    if limit <= 0 or not priors:
        return []
    confirmed = [
        event.anchor_box for event in memory.active
        if event.confirmed_at is not None
    ]

    def rank(item: PriorObservation) -> tuple[int, float]:
        attached = any(
            same_target(anchor, item.box_xyxy, pixel_scale=pixel_scale)
            for anchor in confirmed
        )
        return (1 if attached else 0, -float(item.anomaly_score))

    return sorted(priors, key=rank)[:limit]


def _analyse_hybrid(
    *,
    state: _HybridPadState,
    detector: UltralyticsGroundLitterDetector | None,
    bgr: np.ndarray,
    timestamp: float,
    night: bool,
    actor_boxes: list[list[float]],
) -> GroundLitterSnapshot:
    """Run one dual-channel tick: prior always, semantic on its own cadence.

    Each channel is wrapped in its own try/except: a failure in one must leave
    the other able to confirm and display, and a model/Profile initialisation
    failure is a configuration error handled before the loop starts.
    """
    options = state.options
    reference = state.profile.reference
    profile_height, profile_width = reference.shape[:2]
    height, width = bgr.shape[:2]
    scale_x = profile_width / max(1, width)
    scale_y = profile_height / max(1, height)
    tick_started = time.perf_counter()

    # ------------------------------ prior channel --------------------------
    prior_started = time.perf_counter()
    prior_available = False
    environment_state = "UNAVAILABLE"
    support: np.ndarray | None = None
    valid: np.ndarray | None = None
    prior_observations: list[PriorObservation] = []
    prior_proposed = 0
    # Event anchors live in Profile space.  Keep a separately scaled actor list
    # for lifecycle/occlusion decisions while the semantic detector continues to
    # consume actors in native input-frame pixels.
    profile_actor_boxes = [
        [
            float(box[0]) * scale_x,
            float(box[1]) * scale_y,
            float(box[2]) * scale_x,
            float(box[3]) * scale_y,
        ]
        for box in actor_boxes
        if len(box) >= 4
    ]
    try:
        analysis, aligned = analyze_prior_frame(
            state.profile, options, bgr,
            aligned=state.aligned, actors=actor_boxes,
        )
        state.aligned = aligned
        if analysis.alignment:
            state.alignment = analysis.alignment
        profile_actor_boxes = analysis.actor_rows
        environment_state = analysis.environment_state
        # `support` is the residual support mask and comes from candidate
        # proposal, not from the frame analysis: without a fresh proposal there
        # is no support mask, so clean evidence stays paused for this tick.
        valid = analysis.valid
        if environment_state == "NORMAL":
            state.normal_streak += 1
        else:
            state.normal_streak = 0
        prior_available = (
            state.normal_streak >= int(options.normal_stability_samples)
        )
        if prior_available:
            candidates, support, proposed = propose_prior_candidates(
                options, analysis
            )
            prior_proposed = proposed
            for candidate in candidates:
                x1, y1, x2, y2 = candidate["box"]
                prior_observations.append(PriorObservation(
                    box_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                    anomaly_score=float(candidate.get("anomaly_score", 0.0)),
                    region_id=str(candidate.get("region_id", "")),
                    support_pixels=int(candidate.get("support_pixels", 0)),
                    observed_at=float(timestamp),
                ))
        state.prior_degraded = False
        state.prior_error = ""
    except Exception as exc:
        LOGGER.exception("hybrid_v33 prior 通道异常，semantic 通道继续")
        state.prior_degraded = True
        state.prior_error = f"{type(exc).__name__}: {exc}"
        prior_available = False
        support = None
        valid = None
        prior_observations = []
        prior_proposed = 0
    prior_ms = (time.perf_counter() - prior_started) * 1000.0

    # ---------------------------- semantic channel -------------------------
    semantic_observations: list[SemanticObservation] = []
    semantic_raw = 0
    semantic_retained = 0
    semantic_rejected_roi = 0
    semantic_rejected_actor = 0
    crop_raw = 0
    crop_unmatched = 0
    full_scan_ms = 0.0
    crop_ms = 0.0
    full_scan_ran = False
    semantic_ran = False
    scan_interval = float(options.semantic_scan_interval_seconds)
    due = (
        state.last_full_scan_at is None
        or (timestamp - state.last_full_scan_at) >= scan_interval
    )
    try:
        if detector is not None:
            if due:
                if state.tiles is None or state.tile_size != (width, height):
                    state.masks, state.tiles = build_ground_litter_tiles(
                        options, width, height
                    )
                    state.tile_size = (width, height)
                scan_started = time.perf_counter()
                candidates, stats = detector.tile_candidates_batch(
                    bgr, options, masks=state.masks, tiles=state.tiles,
                    night=night, actors=actor_boxes,
                )
                full_scan_ms = (time.perf_counter() - scan_started) * 1000.0
                semantic_raw = int(stats.get("raw_candidates", 0))
                semantic_retained = len(candidates)
                semantic_rejected_roi = int(stats.get("rejected_roi", 0))
                semantic_rejected_actor = int(stats.get("rejected_actor", 0))
                for candidate in candidates:
                    rect = candidate.rectangle
                    semantic_observations.append(SemanticObservation(
                        box_xyxy=(
                            rect.left * profile_width,
                            rect.top * profile_height,
                            (rect.left + rect.width) * profile_width,
                            (rect.top + rect.height) * profile_height,
                        ),
                        confidence=float(candidate.confidence),
                        class_name=candidate.class_name,
                        region_id=candidate.region_id,
                        source="full_roi",
                        observed_at=float(timestamp),
                    ))
                state.last_full_scan_at = timestamp
                state.semantic_runs_full += 1
                full_scan_ran = True
                semantic_ran = True
            elif int(options.prior_crop_maximum) > 0 and prior_observations:
                selected = _select_crop_priors(
                    prior_observations, options, state.memory, state.pixel_scale
                )
                if selected:
                    native_boxes = [
                        (item.box_xyxy[0] / scale_x, item.box_xyxy[1] / scale_y,
                         item.box_xyxy[2] / scale_x, item.box_xyxy[3] / scale_y)
                        for item in selected
                    ]
                    crop_started = time.perf_counter()
                    rows_per_crop, _rects, _batches = detector.crop_candidates_batch(
                        bgr, native_boxes, options, night=night,
                    )
                    crop_ms = (time.perf_counter() - crop_started) * 1000.0
                    # Crop detections come back in native pixels; map them into
                    # profile space before associating them with prior boxes.
                    mapped = [
                        [
                            {
                                "box": [
                                    float(row["box"][0]) * scale_x,
                                    float(row["box"][1]) * scale_y,
                                    float(row["box"][2]) * scale_x,
                                    float(row["box"][3]) * scale_y,
                                ],
                                "confidence": row["confidence"],
                                "class_id": row["class_id"],
                            }
                            for row in rows
                        ]
                        for rows in rows_per_crop
                    ]
                    matched, crop_raw, crop_unmatched = crop_semantic_observations(
                        mapped, selected, timestamp=float(timestamp),
                        class_names=detector.class_names,
                        profile_scale=state.pixel_scale,
                    )
                    semantic_observations.extend(matched)
                    state.semantic_runs_crop += 1
                    semantic_ran = True
        state.semantic_degraded = False
        state.semantic_error = ""
    except Exception as exc:
        LOGGER.exception("hybrid_v33 semantic 通道异常，prior 通道继续")
        state.semantic_degraded = True
        state.semantic_error = f"{type(exc).__name__}: {exc}"

    result = state.memory.update(
        timestamp=float(timestamp),
        semantic=semantic_observations if semantic_ran else None,
        prior=prior_observations,
        prior_available=prior_available,
        environment_state=environment_state,
        support=support,
        valid=valid,
        actors=profile_actor_boxes,
        semantic_raw=semantic_raw,
        semantic_retained=semantic_retained,
        prior_raw=prior_proposed,
        prior_retained=len(prior_observations),
        semantic_crop_raw=crop_raw,
        semantic_crop_unmatched=crop_unmatched,
        # Only a full ROI scan advances the semantic window; a crop tick is
        # auxiliary evidence and must not look like a scan miss.
        semantic_scan=full_scan_ran,
    )

    if state.prior_degraded and state.semantic_degraded:
        branch_state = "both_degraded"
    elif state.prior_degraded:
        branch_state = "prior_degraded"
    elif state.semantic_degraded:
        branch_state = "semantic_degraded"
    else:
        branch_state = "ok"
    total_ms = (time.perf_counter() - tick_started) * 1000.0
    state.result_version += 1
    return GroundLitterSnapshot(
        state=result.state,
        detections=result.detections,
        result_version=state.result_version,
        updated_at=float(timestamp),
        message=result.message,
        raw_candidates=result.prior_raw,
        rejected_roi=semantic_rejected_roi,
        rejected_actor=semantic_rejected_actor,
        tile_count=len(state.tiles or ()),
        active_events=result.active_events,
        confirmed_events=result.confirmed_events,
        cleared_events=result.cleared_events,
        environment_state=result.environment_state,
        semantic_raw_candidates=result.semantic_raw,
        semantic_retained_candidates=result.semantic_retained,
        prior_raw_candidates=result.prior_raw,
        prior_retained_candidates=result.prior_retained,
        semantic_only_active=result.semantic_only_active,
        semantic_only_confirmed=result.semantic_only_confirmed,
        prior_only_active=result.prior_only_active,
        prior_only_confirmed=result.prior_only_confirmed,
        fused_active=result.fused_active,
        fused_confirmed=result.fused_confirmed,
        cross_source_merges=result.cross_source_merges,
        prior_environment_state=result.environment_state,
        semantic_model_runs_full=state.semantic_runs_full,
        semantic_model_runs_crop=state.semantic_runs_crop,
        semantic_crop_raw_candidates=result.semantic_crop_raw,
        semantic_crop_unmatched_candidates=result.semantic_crop_unmatched,
        last_prior_ms=round(prior_ms, 2),
        last_full_scan_ms=round(full_scan_ms, 2),
        last_crop_batch_ms=round(crop_ms, 2),
        last_total_ms=round(total_ms, 2),
        branch_state=branch_state,
        branch_message="; ".join(
            part for part in (state.prior_error, state.semantic_error) if part
        ),
    )


def _run_ground_litter_process(
    config: GroundLitterProcessConfig,
    input_queue: Any,
    output_queue: Any,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(processName)s %(message)s",
    )
    try:
        needs_detector = any(
            options.mode in ("yolo", "hybrid_v33")
            or options.actor_model is not None
            for options in config.options_by_pad.values()
        )
        detector = (
            UltralyticsGroundLitterDetector(
                model_path=config.model_path,
                device=config.device,
                half=config.half,
                actor_model_path=config.actor_model_path,
            )
            if needs_detector
            else None
        )
        trackers = {
            pad_index: GroundLitterDisplayTracker(options)
            for pad_index, options in config.options_by_pad.items()
            if options.mode == "yolo"
        }
        clean_processors = {}
        for pad_index, options in config.options_by_pad.items():
            if options.mode != "clean_reference_v32":
                continue
            if config.profile_root is None or options.profile_id is None:
                raise ValueError("clean_reference_v32缺少profile_root或profile_id")
            profile = CleanReferenceProfileV32.load(
                config.profile_root, options.profile_id
            )
            clean_processors[pad_index] = CleanReferenceV32Processor(
                options, profile
            )
        hybrid_states: dict[int, _HybridPadState] = {}
        for pad_index, options in config.options_by_pad.items():
            if options.mode != "hybrid_v33":
                continue
            if config.profile_root is None or options.profile_id is None:
                raise ValueError("hybrid_v33缺少profile_root或profile_id")
            profile = CleanReferenceProfileV32.load(
                config.profile_root, options.profile_id
            )
            hybrid_states[pad_index] = _HybridPadState(options, profile)
        counters = {
            pad_index: {"analyzed_frames": 0, "dropped_frames": 0}
            for pad_index in config.options_by_pad
        }
        views: dict[int, tuple[Any, Any, tuple[int, int]]] = {}
    except Exception as exc:
        LOGGER.exception("零散垃圾独立推理进程初始化失败")
        for pad_index in config.options_by_pad:
            _put_latest(
                output_queue,
                (
                    pad_index,
                    GroundLitterSnapshot(
                        state="error",
                        message=(
                            "零散垃圾检测组件初始化失败: "
                            f"{type(exc).__name__}"
                        ),
                    ),
                ),
            )
        return
    for pad_index, options in config.options_by_pad.items():
        _put_latest(
            output_queue,
            (
                pad_index,
                GroundLitterSnapshot(
                    state="starting",
                    message=(
                        "零散垃圾识别启动中"
                        if not options.enabled
                        else (
                            f"零散垃圾识别启动中({options.mode},"
                            f"{len(options.zones)}区)"
                        )
                    ),
                ),
            ),
        )
    while True:
        try:
            item = input_queue.get()
        except (EOFError, OSError):  # pragma: no cover - parent died
            return
        if item is None:
            return
        # The enqueue wall-clock is optional so older 5-tuples still work.
        enqueued_at = float(item[5]) if len(item) > 5 else time.monotonic()
        pad_index, frame, timestamp, night, actors = item[:5]
        pad_index = int(pad_index)
        options = config.options_by_pad.get(pad_index)
        if options is None:
            continue
        frame_age_ms = max(0.0, (time.monotonic() - enqueued_at) * 1000.0)
        # Drop frames that waited longer than the allowed staleness instead of
        # analysing a stale picture and publishing it as current.
        max_frame_age_ms = max(
            2000.0 * (1.0 / max(float(options.analysis_fps), 0.1)), 4000.0
        )
        if frame_age_ms > max_frame_age_ms:
            counters[pad_index]["dropped_frames"] += 1
            continue
        try:
            bgr = as_bgr(frame)
            height, width = bgr.shape[:2]
            # The main chain sends normalized [left, top, width, height]; the
            # detector compares pixel boxes, so convert against this frame.
            actor_boxes = [
                [
                    float(box[0]) * width,
                    float(box[1]) * height,
                    (float(box[0]) + float(box[2])) * width,
                    (float(box[1]) + float(box[3])) * height,
                ]
                for box in actors
                if len(box) >= 4
            ]
            started = time.perf_counter()
            if detector is not None and detector.has_actor_model:
                actor_boxes.extend(detector.actor_boxes(bgr, options))
            if options.mode == "clean_reference_v32":
                snapshot = clean_processors[pad_index].update(
                    bgr,
                    timestamp=float(timestamp),
                    actors=actor_boxes,
                )
                stats = {
                    "raw_candidates": snapshot.raw_candidates,
                    "rejected_roi": 0,
                    "rejected_actor": 0,
                }
                tile_count = 0
            elif options.mode == "hybrid_v33":
                snapshot = _analyse_hybrid(
                    state=hybrid_states[pad_index],
                    detector=detector,
                    bgr=bgr,
                    timestamp=float(timestamp),
                    night=bool(night),
                    actor_boxes=actor_boxes,
                )
                stats = {
                    "raw_candidates": snapshot.prior_raw_candidates,
                    "rejected_roi": 0,
                    "rejected_actor": 0,
                }
                tile_count = snapshot.tile_count
            else:
                assert detector is not None
                view = views.get(pad_index)
                if view is None or view[2] != (width, height):
                    masks, tiles = build_ground_litter_tiles(
                        options,
                        width,
                        height,
                    )
                    trackers[pad_index].reset()
                    view = views[pad_index] = (masks, tiles, (width, height))
                    LOGGER.info(
                        "零散垃圾分块就绪: pad=%d 分辨率=%dx%d 分块=%d 区域=%d",
                        pad_index,
                        width,
                        height,
                        len(tiles),
                        len(options.zones),
                    )
                masks, tiles, _size = view
                candidates, stats = detector.candidates(
                    bgr,
                    options,
                    masks=masks,
                    tiles=tiles,
                    night=bool(night),
                    actors=actor_boxes,
                )
                snapshot = trackers[pad_index].update(
                    candidates,
                    timestamp=float(timestamp),
                    occluders=[
                        NormalizedRect(
                            a / width, b / height,
                            (c - a) / width, (d - b) / height,
                        )
                        for a, b, c, d in getattr(
                            detector, "last_actor_boxes", actor_boxes
                        )
                    ],
                )
                tile_count = len(tiles)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            counters[pad_index]["analyzed_frames"] += 1
            _put_latest(
                output_queue,
                (
                    pad_index,
                    replace(
                        snapshot,
                        last_inference_ms=round(elapsed_ms, 2),
                        analyzed_frames=counters[pad_index][
                            "analyzed_frames"
                        ],
                        raw_candidates=int(stats["raw_candidates"]),
                        rejected_roi=int(stats["rejected_roi"]),
                        rejected_actor=int(stats["rejected_actor"]),
                        tile_count=tile_count,
                        input_frame_age_ms=round(frame_age_ms, 2),
                        dropped_analysis_frames=int(
                            counters[pad_index]["dropped_frames"]
                        ),
                    ),
                ),
            )
        except Exception as exc:
            LOGGER.exception(
                "零散垃圾旁路分析失败，主RTSP继续运行: pad=%d",
                pad_index,
            )
            _put_latest(
                output_queue,
                (
                    pad_index,
                    GroundLitterSnapshot(
                        state="error",
                        message=(
                            "零散垃圾分析异常: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                    ),
                ),
            )


class GroundLitterProcessClient:
    """Frame submission and result collection for one DeepStream group."""

    def __init__(
        self,
        config: GroundLitterProcessConfig,
        cache: GroundLitterResultCache,
    ) -> None:
        self._cache = cache
        self._options_by_pad = dict(config.options_by_pad)
        self._next_due = {
            pad_index: 0.0 for pad_index in self._options_by_pad
        }
        context = multiprocessing.get_context("spawn")
        queue_size = max(len(self._options_by_pad) * 2, 2)
        self._input_queue = context.Queue(maxsize=queue_size)
        self._output_queue = context.Queue(maxsize=max(queue_size * 2, 4))
        self._process = context.Process(
            target=_run_ground_litter_process,
            args=(config, self._input_queue, self._output_queue),
            name="ground-litter-inference",
            daemon=True,
        )
        for pad_index in self._options_by_pad:
            cache.mark_state(
                pad_index,
                "starting",
                "零散垃圾识别启动中",
            )
        self._process.start()

    @property
    def enabled_pads(self) -> tuple[int, ...]:
        return tuple(
            pad_index
            for pad_index, options in self._options_by_pad.items()
            if options.enabled
        )

    def accepts(
        self,
        pad_index: int,
        *,
        timestamp: float | None = None,
    ) -> bool:
        """Rate limit and process health check before the device copy."""
        self.drain_results()
        options = self._options_by_pad.get(pad_index)
        if options is None or not options.enabled:
            return False
        if not self._process.is_alive():
            current = self._cache.snapshot(pad_index)
            if current.state != "error":
                self._cache.mark_state(
                    pad_index,
                    "error",
                    "零散垃圾独立推理进程已退出",
                )
            return False
        now = time.monotonic() if timestamp is None else float(timestamp)
        if now < self._next_due[pad_index]:
            return False
        self._next_due[pad_index] = now + 1.0 / options.analysis_fps
        return True

    def submit(
        self,
        pad_index: int,
        frame: np.ndarray,
        *,
        timestamp: float,
        night: bool = False,
        actors: Any = (),
    ) -> bool:
        """Queue one frame, keeping at most one pending frame per pad.

        The old implementation dropped the *new* frame when the queue was full,
        which is the opposite of its own comment: the side process would then
        analyse an ever-staler backlog. Here the newest frame always wins for its
        own pad, while another pad's only pending frame is never evicted, because
        with several pads on one group that would starve a stream.
        """
        if pad_index not in self._options_by_pad:
            return False
        item = (
            pad_index,
            frame,
            float(timestamp),
            bool(night),
            list(actors),
            time.monotonic(),
        )
        # Normalise on every submit, not only when the queue happens to be full:
        # a pad must hold at most one pending frame, otherwise a slower consumer
        # still builds a backlog of stale frames. Draining also lets us keep the
        # newest frame for *this* pad while leaving every other pad's only
        # pending frame untouched.
        pending: dict[int, Any] = {}
        sentinels = 0
        try:
            while True:
                other = self._input_queue.get_nowait()
                if other is None:
                    sentinels += 1  # shutdown sentinel: never drop it
                    continue
                if int(other[0]) == int(pad_index):
                    continue
                pending.setdefault(int(other[0]), other)
        except queue.Empty:
            pass
        pending[int(pad_index)] = item
        inserted = True
        for value in pending.values():
            try:
                self._input_queue.put_nowait(value)
            except queue.Full:
                inserted = False
                break
        for _ in range(sentinels):
            try:
                self._input_queue.put_nowait(None)
            except queue.Full:
                break
        return inserted

    def drain_results(self) -> None:
        while True:
            try:
                pad_index, snapshot = self._output_queue.get_nowait()
            except queue.Empty:
                return
            self._cache.store_snapshot(int(pad_index), snapshot)

    def shutdown(self, timeout: float = 5.0) -> None:
        self.drain_results()
        if self._process.is_alive():
            try:
                self._input_queue.put_nowait(None)
            except queue.Full:
                try:
                    self._input_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._input_queue.put_nowait(None)
                except queue.Full:
                    pass
            self._process.join(timeout=max(timeout, 0.0))
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)
        self._input_queue.close()
        self._output_queue.close()
