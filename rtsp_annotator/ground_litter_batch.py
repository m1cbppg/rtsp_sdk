"""Bounded offline replay runner for the independent ground-litter pilot."""
from __future__ import annotations

from dataclasses import dataclass
import html
import json
from pathlib import Path
import time
from typing import Iterator

import cv2

from .ground_litter_geometry import polygon_points
from .ground_litter_inventory import GroundLitterInventory
from .ground_litter_journal import PilotJournal, output_bytes
from .ground_litter_review import ReviewCollector
from .ground_litter_runtime import (
    CameraAnalysis,
    EvidenceWindow,
    LitterModel,
    pixels,
    validate_profiles,
)


def parse_inputs(values: list[str]) -> dict[str, Path]:
    """Parse repeated DEVICE=VIDEO arguments with strict device validation."""
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--input must use DEVICE=VIDEO")
        device, path = value.split("=", 1)
        if not device.isdigit() or len(device) != 20 or not path:
            raise ValueError("invalid --input device or video path")
        if device in result:
            raise ValueError("duplicate --input device")
        result[device] = Path(path)
    if not result:
        raise ValueError("at least one --input is required")
    return result


@dataclass(frozen=True, slots=True)
class ReplayFrame:
    index: int
    timestamp: float
    frame: object


@dataclass(frozen=True, slots=True)
class BatchOptions:
    sample_fps: float = 0.5
    max_frames: int = 10000
    max_duration_seconds: float | None = None
    jpeg_quality: int = 88
    max_candidate_frames: int = 120

    def validate(self) -> None:
        if not 0 < self.sample_fps <= 10:
            raise ValueError("sample_fps must be in (0, 10]")
        if self.max_frames < 1:
            raise ValueError("max_frames must be positive")
        if self.max_duration_seconds is not None and self.max_duration_seconds <= 0:
            raise ValueError("max_duration_seconds must be positive")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        if self.max_candidate_frames < 1:
            raise ValueError("max_candidate_frames must be positive")


def iter_video_frames(source: str | Path, options: BatchOptions) -> Iterator[ReplayFrame]:
    """Yield deterministic, finite samples from a local video."""
    options.validate()
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(path)
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise RuntimeError(f"could not open replay video: {path.name}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    if not 1 <= fps <= 240:
        fps = 25.0
    stride = max(1, round(fps / options.sample_fps))
    index = 0
    yielded = 0
    try:
        while yielded < options.max_frames:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            timestamp = index / fps
            if options.max_duration_seconds is not None and timestamp > options.max_duration_seconds:
                break
            if index % stride == 0:
                yield ReplayFrame(index, timestamp, frame)
                yielded += 1
            index += 1
    finally:
        capture.release()


def _write_jpeg(path: Path, frame, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, content = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"could not encode evidence image: {path.name}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content.tobytes())
    temporary.replace(path)


def _annotate(frame, camera: dict, proposals, confirmed, item_ids):
    annotation = frame.copy()
    height, width = annotation.shape[:2]
    for zone in camera["zones"]:
        cv2.polylines(annotation, [polygon_points(zone["polygon"], width, height)],
                      True, (0, 220, 220), 3)
        for exclusion in zone.get("exclude_zones", []):
            cv2.polylines(annotation, [polygon_points(exclusion, width, height)],
                          True, (0, 0, 255), 2)
    for exclusion in camera.get("overlay_exclude_zones", []):
        cv2.polylines(annotation, [polygon_points(exclusion, width, height)],
                      True, (0, 0, 255), 2)
    for proposal in proposals:
        x, y, right, bottom = proposal["box"]
        color = (0, 220, 220) if proposal.get("source") != "change_only" else (255, 180, 0)
        cv2.rectangle(annotation, (x, y), (right, bottom), color, 2)
        label = str(proposal.get("label", "candidate"))
        confidence = proposal.get("confidence")
        if confidence is not None:
            label += f" {float(confidence):.2f}"
        cv2.putText(annotation, label, (x, max(24, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2, cv2.LINE_AA)
    for item, identity in zip(confirmed, item_ids):
        x, y, right, bottom = pixels(item.rectangle, width, height)
        cv2.rectangle(annotation, (x, y), (right, bottom), (0, 0, 255), 3)
        cv2.putText(annotation, "confirmed " + identity[:10], (x, max(28, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2, cv2.LINE_AA)
    return annotation


def _review_page(output: Path, events: list[dict], candidates: list[dict] | None = None) -> None:
    """Write a local-only review page with downloadable labels."""
    candidates = candidates or []
    review_items = list(events) + [
        {"item_id": item["review_id"], "kind": "candidate",
         "source_time_seconds": item["source_time_seconds"]}
        for item in candidates
    ]
    payload = json.dumps(review_items, ensure_ascii=False).replace("</", "<\\/")
    cards = []
    candidates = candidates or []
    for event in events:
        identity = html.escape(event["item_id"])
        image = html.escape(event.get("confirmed_image", ""))
        cards.append(
            f'<article class="card"><h2>{identity[:12]}</h2>'
            f'<p>摄像头 {html.escape(event["camera_id"])}，区域 {html.escape(event["region_id"])}，'
            f'首次确认 {event["first_seen_seconds"]:.1f}s</p>'
            f'<a href="{image}"><img src="{image}" loading="lazy"></a>'
            f'<label><input type="radio" name="{identity}" value="true_litter">真实垃圾</label>'
            f'<label><input type="radio" name="{identity}" value="false_positive">误报</label>'
            f'<label><input type="radio" name="{identity}" value="uncertain">不确定</label>'
            f'<textarea data-note="{identity}" placeholder="备注（可选）"></textarea></article>'
        )
    for candidate in candidates:
        review_id = html.escape(candidate["review_id"])
        image = html.escape(candidate["image"])
        labels = ", ".join(html.escape(str(x.get("label", "unknown")))
                       for x in candidate.get("candidates", []))
        kind = '均匀抽样（检查漏检）' if candidate.get('kind') == 'audit' else '候选帧'
        crops = ''.join(f'<a href="{html.escape(path)}"><img class="crop" src="{html.escape(path)}" loading="lazy"></a>'
                        for path in candidate.get('crop_images', []))
        cards.append(
            f'<article class="card"><h2>{kind} {review_id}</h2>'
            f'<p>源时间 {float(candidate["source_time_seconds"]):.1f}s；模型类别：{labels}</p>'
            f'<a href="{image}"><img src="{image}" loading="lazy"></a>'
            f'<div>{crops}</div>'
            f'<label><input type="radio" name="{review_id}" value="true_litter">疑似真实垃圾</label>'
            f'<label><input type="radio" name="{review_id}" value="false_positive">明显误报</label>'
            f'<label><input type="radio" name="{review_id}" value="uncertain" checked>不确定</label>'
            f'<textarea data-note="{review_id}" placeholder="备注（可选）"></textarea></article>'
        )
    document = (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>零散垃圾人工复核</title>'
        '<style>body{font:15px system-ui;margin:20px;background:#eef3f5;color:#20333b}'
        'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:14px}'
        '.card{background:white;border-radius:8px;padding:12px;box-shadow:0 1px 4px #0002}'
        'img{width:100%;max-height:360px;object-fit:contain;background:#111}'
        'img.crop{width:auto;max-width:100%;height:150px;image-rendering:pixelated;margin:3px}'
        'label{display:block;margin:7px 0}textarea{width:100%;min-height:45px;box-sizing:border-box}'
        'button{padding:9px 14px;margin:12px 4px 18px 0}</style>'
        '<h1>零散垃圾候选人工复核</h1><p>结果仅用于评估，不会发送通知。'
        '局部图来自原图裁剪。均匀抽样与模型是否检出无关，用于查漏；未标注不算无垃圾。</p>'
        '<button id="download">下载 labels.json</button><button id="copy">复制 JSON</button>'
        f'<main>{"".join(cards) or "<p>没有持续确认记录。</p>"}</main>'
        f'<script type="application/json" id="events">{payload}</script><script>'
        'const reviewItems=JSON.parse(document.getElementById("events").textContent);'
        'function labels(){return {labels:reviewItems.map(e=>{const n=e.item_id;'
        'const c=[...document.querySelectorAll("input:checked")].find(x=>x.name===n);'
        'const note=[...document.querySelectorAll("textarea")].find(x=>x.dataset.note===n);'
        'return {item_id:n,label:c?c.value:"uncertain",note:note?note.value:""};})};}'
        'function save(){const b=new Blob([JSON.stringify(labels(),null,2)+"\\n"],'
        '{type:"application/json"});const a=document.createElement("a");'
        'a.href=URL.createObjectURL(b);a.download="labels.json";a.click();'
        'setTimeout(()=>URL.revokeObjectURL(a.href),1000);}'
        'document.getElementById("download").onclick=save;'
        'document.getElementById("copy").onclick=async()=>{await navigator.clipboard.writeText('
        'JSON.stringify(labels(),null,2));alert("已复制");};</script></html>'
    )
    (output / "review.html").write_text(document, encoding="utf-8")


def run_video(*, camera: dict, source: str | Path, output: str | Path,
              model_config: dict, mode: str, model: LitterModel,
              options: BatchOptions, allow_draft: bool = False) -> dict:
    """Run one camera replay and return a non-accuracy summary."""
    options.validate()
    if mode not in {"day", "night"}:
        raise ValueError("mode must be day or night")
    validate_profiles({"schema_version": 2, "model": model_config,
                       "inventory": {"clear_seconds": 60,
                                     "max_observation_gap_seconds": 3},
                       "cameras": [camera]}, allow_draft=allow_draft)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "evidence").mkdir(exist_ok=True)
    journal = PilotJournal(destination, f"batch-{camera['device_code'][-4:]}-{int(time.time())}")
    inventory = GroundLitterInventory(destination / "items.sqlite3", clear_seconds=60,
                                      max_observation_gap=3)
    # Offline replay measures detector/ROI behavior separately from realtime
    # freshness. The live shadow runner still enforces its normal age limit;
    # here slow CPU inference is logged rather than filtering the model result.
    analysis = CameraAnalysis(camera, model_config, mode)
    review = ReviewCollector(destination,limit=options.max_candidate_frames,quality=options.jpeg_quality)
    events: dict[str, dict] = {}
    candidate_review: list[dict] = []
    candidate_frames = 0
    observations = 0
    rejected = 0
    frame_count = 0
    started = time.monotonic()
    try:
        journal.write("run_started", camera_id=camera["device_code"], mode=mode,
                      source="local_replay", sample_fps=options.sample_fps)
        for sample in iter_video_frames(source, options):
            frame_count += 1
            acquired = time.monotonic()
            try:
                proposals, actors, confirmed, result = analysis.consume(
                    model, inventory, sample.frame, sample.timestamp, 1,
                    captured_at=acquired, enforce_freshness=False)
            except Exception as exc:
                rejected += 1
                analysis.evidence = EvidenceWindow(camera, mode)
                journal.write("rejected", camera_id=camera["device_code"],
                              reason=str(exc) if isinstance(exc,ValueError) else type(exc).__name__,
                              diagnostics=dict(analysis.last_diagnostics),source_frame_index=sample.index,
                              source_time_seconds=sample.timestamp)
                review.consider(sample.frame,sample.frame,[],sample.timestamp,sample.index,{})
                continue
            observations += 1
            if proposals:
                candidate_frames += 1
            annotation = _annotate(sample.frame, camera, proposals, confirmed,
                                   result.item_ids)
            entry = {
                "camera_id": camera["device_code"],
                "source_frame_index": sample.index,
                "source_time_seconds": round(sample.timestamp, 3),
                "inference_seconds": round(time.monotonic() - acquired, 3),
                "candidates": proposals,
                "visible_item_ids": result.item_ids,
                "created_item_ids": result.created_ids,
                "cleared_item_ids": result.cleared_ids,
                "diagnostics": dict(analysis.last_diagnostics),
            }
            journal.write("observation", **entry)
            for identity in result.created_ids:
                row = inventory.get(identity)
                event = {
                    "item_id": identity,
                    "camera_id": camera["device_code"],
                    "view_id": camera["view_id"],
                    "region_id": row["region_id"],
                    "first_seen_seconds": round(sample.timestamp, 3),
                    "last_seen_seconds": round(sample.timestamp, 3),
                    "confirmed_image": f"evidence/{identity}/confirmed.jpg",
                }
                events[identity] = event
                _write_jpeg(destination / event["confirmed_image"], annotation,
                            options.jpeg_quality)
            for identity in result.item_ids:
                if identity in events:
                    events[identity]["last_seen_seconds"] = round(sample.timestamp, 3)
                    if not (destination / "evidence" / identity / "latest.jpg").exists():
                        _write_jpeg(destination / "evidence" / identity / "latest.jpg",
                                    annotation, options.jpeg_quality)
            review.consider(sample.frame,annotation,proposals,sample.timestamp,sample.index,analysis.last_diagnostics)
        journal.write("run_stopped", frame_count=frame_count,
                      observations=observations, rejected=rejected)
    finally:
        journal.close()
        inventory.close()
    elapsed = time.monotonic() - started
    candidate_review=review.entries()
    review.save()
    summary = {
        "status": "offline_shadow_batch_not_accuracy",
        "camera_id": camera["device_code"],
        "source": Path(source).name,
        "mode": mode,
        "sample_fps": options.sample_fps,
        "sampled_frames": frame_count,
        "observations": observations,
        "rejected_frames": rejected,
        "candidate_frames": candidate_frames,
        "confirmed_records": len(events),
        "elapsed_seconds": round(elapsed, 3),
        "output_bytes": output_bytes(destination),
        "events": list(events.values()),
        "candidate_review_count": len(candidate_review),
        "review_sampling": {"candidate_pool":review.seen['candidate'],"audit_pool":review.seen['audit'],
                            "method":"bounded_reservoir_plus_periodic_independent_frames"},
        "limitations": [
            "没有人工真值，候选/记录数量不能解释为准确率",
            "本地文件的源时间用于时序确认，不证明平台回放接口的历史时间参数有效",
            "本批次不发送通知，不接入生产主链",
        ],
    }
    (destination / "events.json").write_text(
        json.dumps(list(events.values()), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    (destination / "candidate_review.json").write_text(
        json.dumps(candidate_review, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _review_page(destination, list(events.values()), candidate_review)
    return summary
