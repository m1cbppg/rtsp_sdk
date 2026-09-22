"""Deterministic three-scenario proof for Ground Litter V3.3 dual-channel recall.

The spec requires the three user-visible outcomes to be demonstrated with a
per-timestamp event trace, not just a final screenshot:

1. a bottle/bag the model recognises (semantic-only) confirms without any prior;
2. a small paper the model misses (prior-only) confirms without any semantic;
3. the same target seen by both produces exactly one fused event.

Traces cover the whole lifecycle: startup suppression, pending, confirmation,
display, clearing and closure. The module under test is pure logic, so this
needs no model, GPU or camera.

Usage:
  .venv/bin/python scripts/run_ground_litter_v33_scenarios.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = "output/ground_litter_v33_scenarios_20260918"

TARGET = (100.0, 100.0, 140.0, 140.0)
SHAPE = (400, 400)
SCAN_PERIOD = 4.0


def options(startup_suppress_seconds: float = 15.0):
    from rtsp_annotator.ground_litter_detection import (
        GroundLitterDetectionOptions,
    )

    return GroundLitterDetectionOptions(
        mode="hybrid_v33",
        enabled=False,
        analysis_fps=0.5,
        startup_suppress_seconds=startup_suppress_seconds,
        profile_id="camera_01_v32_afternoon_1080p",
        model="litter/turhancan_yolov8m_seg_trash.pt",
        semantic_scan_interval_seconds=SCAN_PERIOD,
        semantic_confirm_hits=2,
        semantic_hit_window=3,
        semantic_confirm_span_seconds=4.0,
        semantic_clear_seconds=8.0,
        semantic_clear_min_misses=2,
        prior_confirm_hits=4,
        prior_hit_window=6,
        prior_confirm_span_seconds=6.0,
        fused_confirm_hits=2,
        fused_hit_window=4,
        fused_confirm_span_seconds=2.0,
        clear_confirm_seconds=6.0,
        pending_expire_seconds=20.0,
        maximum_closed_events=1000,
    )


def clean_arrays():
    return np.zeros(SHAPE, np.uint8), np.ones(SHAPE, np.uint8)


def dirty_arrays():
    support = np.zeros(SHAPE, np.uint8)
    valid = np.ones(SHAPE, np.uint8)
    x1, y1, x2, y2 = (int(v) for v in TARGET)
    support[y1:y2, x1:x2] = 255
    return support, valid


def semantic_observation(timestamp: float, confidence: float = 0.62):
    from rtsp_annotator.ground_litter_v33 import SemanticObservation

    return SemanticObservation(
        box_xyxy=TARGET, confidence=confidence, class_name="Plastic",
        region_id="walkway_v33_acceptance", source="full_roi",
        observed_at=timestamp,
    )


def prior_observation(timestamp: float, score: float = 1.4):
    from rtsp_annotator.ground_litter_v33 import PriorObservation

    return PriorObservation(
        box_xyxy=TARGET, anomaly_score=score,
        region_id="walkway_v33_acceptance", support_pixels=420,
        observed_at=timestamp,
    )


def build_ticks(scenario: str) -> list[dict[str, Any]]:
    """Build one tick spec per analysis period for a scenario."""
    ticks: list[dict[str, Any]] = []
    for step in range(24):
        timestamp = float(step * 2)
        is_scan = abs(timestamp % SCAN_PERIOD) < 1e-9
        semantic: list[Any] | None = [] if is_scan else None
        prior: list[Any] = []
        support, valid = clean_arrays()
        if scenario == "semantic_only":
            if is_scan and timestamp in (16.0, 20.0):
                semantic = [semantic_observation(timestamp)]
            if timestamp in (24.0, 28.0, 32.0) and is_scan:
                semantic = []          # the model stops seeing it
        elif scenario == "prior_only":
            if timestamp in (16.0, 18.0, 20.0, 22.0):
                prior = [prior_observation(timestamp)]
                support, valid = dirty_arrays()
        elif scenario == "fused":
            if is_scan and timestamp in (16.0, 20.0):
                semantic = [semantic_observation(timestamp)]
            if timestamp in (16.0, 18.0, 20.0, 22.0):
                prior = [prior_observation(timestamp)]
                support, valid = dirty_arrays()
        if timestamp > 22.0:
            support, valid = clean_arrays()
        ticks.append({
            "timestamp": timestamp,
            "semantic": semantic,
            "prior": prior,
            "prior_available": True,
            "environment_state": "NORMAL",
            "support": support,
            "valid": valid,
            "semantic_scan": bool(is_scan),
        })
    return ticks


def run_scenario(scenario: str) -> dict[str, Any]:
    from rtsp_annotator.ground_litter_v33 import V33EventMemory

    memory = V33EventMemory(options(), pixel_scale=1.0)
    trace: list[dict[str, Any]] = []
    for tick in build_ticks(scenario):
        result = memory.update(**tick)
        trace.append({
            "timestamp": tick["timestamp"],
            "state": result.state,
            "displayed": [
                {
                    "event_id": detection.object_id,
                    "source": detection.source,
                    "confidence": round(float(detection.confidence), 3),
                    "rectangle": [
                        round(float(detection.rectangle.left), 4),
                        round(float(detection.rectangle.top), 4),
                        round(float(detection.rectangle.width), 4),
                        round(float(detection.rectangle.height), 4),
                    ],
                }
                for detection in result.detections
            ],
            "active_events": result.active_events,
            "confirmed_events": result.confirmed_events,
            "cleared_events": result.cleared_events,
            "semantic_only_active": result.semantic_only_active,
            "prior_only_active": result.prior_only_active,
            "fused_active": result.fused_active,
            "message": result.message,
        })
    events = [
        {
            "event_id": event.event_id,
            "evidence_kind": event.evidence_kind,
            "state": event.state,
            "confirmed_at": event.confirmed_at,
            "closed_at": event.closed_at,
            "closed_reason": event.closed_reason,
            "first_seen_at": event.first_seen_at,
            "last_seen_at": event.last_seen_at,
            "semantic_hits": event.semantic_hits,
            "prior_hits": event.prior_hits,
            "merges": event.merges,
            "state_history": event.state_history,
        }
        for event in memory.events
    ]
    return {"scenario": scenario, "trace": trace, "events": events}


def expected(scenario: str) -> dict[str, Any]:
    if scenario == "semantic_only":
        return {"source": "hybrid_v33_semantic", "events": 1,
                "kind": "semantic_only", "closed_reason": "semantic_absent_confirmed"}
    if scenario == "prior_only":
        return {"source": "hybrid_v33_prior", "events": 1,
                "kind": "prior_only", "closed_reason": "clean_confirmed"}
    return {"source": "hybrid_v33_fused", "events": 1,
            "kind": "semantic_and_prior", "closed_reason": "clean_confirmed"}


def verify(scenario: str, run: dict[str, Any]) -> dict[str, Any]:
    want = expected(scenario)
    displayed = [
        entry for entry in run["trace"] if entry["displayed"]
    ]
    sources = sorted({
        item["source"] for entry in displayed for item in entry["displayed"]
    })
    event_ids = sorted({
        item["event_id"] for entry in displayed for item in entry["displayed"]
    })
    warmup_display = [
        entry["timestamp"] for entry in run["trace"]
        if entry["state"] == "warming_up" and entry["displayed"]
    ]
    pre_confirm_display = [
        entry["timestamp"] for entry in run["trace"]
        if entry["confirmed_events"] == 0 and entry["displayed"]
    ]
    open_events = [e for e in run["events"] if e["closed_at"] is None]
    outcomes = {
        "displayed_at_all": bool(displayed),
        "sources_match": sources == [want["source"]],
        "single_event": len([e for e in run["events"]]) == 1,
        "single_event_id_displayed": len(event_ids) <= 1,
        "evidence_kind": (
            run["events"][0]["evidence_kind"] == want["kind"]
            if run["events"] else False
        ),
        "closed_reason": (
            run["events"][0]["closed_reason"] == want["closed_reason"]
            if run["events"] else False
        ),
        "no_warmup_display": not warmup_display,
        "no_single_frame_display": not pre_confirm_display,
        "event_closed": not open_events,
    }
    return {
        "expected": want,
        "observed_sources": sources,
        "observed_display_ticks": [entry["timestamp"] for entry in displayed],
        "checks": outcomes,
        "passed": all(outcomes.values()),
    }


def render_html(runs: list[dict[str, Any]], verdicts: dict[str, Any]) -> str:
    parts = [
        "<!doctype html><html lang='zh'><head><meta charset='utf-8'>",
        "<title>V3.3 双通道三场景事件 trace</title>",
        "<style>body{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:24px}",
        "table{border-collapse:collapse;margin:12px 0}td,th{border:1px solid #ccc;",
        "padding:4px 8px;font-size:13px}th{background:#f3f3f3}",
        ".s{color:#0a7}.p{color:#07a}.f{color:#a0a}.none{color:#999}",
        "code{background:#f6f6f6;padding:1px 4px}</style></head><body>",
        "<h1>V3.3 双通道召回 — 三场景事件 trace</h1>",
    ]
    for run in runs:
        scenario = run["scenario"]
        verdict = verdicts[scenario]
        parts.append(f"<h2>{scenario} — {'通过' if verdict['passed'] else '失败'}</h2>")
        parts.append(
            f"<p>期望来源 <code>{verdict['expected']['source']}</code>，"
            f"实际 <code>{verdict['observed_sources']}</code>；"
            f"显示时间戳 {verdict['observed_display_ticks']}</p>"
        )
        parts.append(
            "<table><tr><th>t(s)</th><th>state</th><th>显示</th>"
            "<th>active</th><th>confirmed</th><th>cleared</th>"
            "<th>S-only</th><th>P-only</th><th>fused</th></tr>"
        )
        for entry in run["trace"]:
            cells = []
            for item in entry["displayed"]:
                cls = {"hybrid_v33_semantic": "s", "hybrid_v33_prior": "p",
                       "hybrid_v33_fused": "f"}.get(item["source"], "none")
                cells.append(
                    f"<span class='{cls}'>#{item['event_id']} "
                    f"{item['source'].replace('hybrid_v33_', '')} "
                    f"{item['confidence']:.2f}</span>"
                )
            parts.append(
                f"<tr><td>{entry['timestamp']:.0f}</td><td>{entry['state']}</td>"
                f"<td>{' '.join(cells) or '-'}</td>"
                f"<td>{entry['active_events']}</td>"
                f"<td>{entry['confirmed_events']}</td>"
                f"<td>{entry['cleared_events']}</td>"
                f"<td>{entry['semantic_only_active']}</td>"
                f"<td>{entry['prior_only_active']}</td>"
                f"<td>{entry['fused_active']}</td></tr>"
            )
        parts.append("</table>")
        for event in run["events"]:
            parts.append(
                f"<p>事件 #{event['event_id']}：kind=<code>{event['evidence_kind']}</code>"
                f"，state=<code>{event['state']}</code>，"
                f"confirmed_at={event['confirmed_at']}，"
                f"closed_at={event['closed_at']}，"
                f"closed_reason=<code>{event['closed_reason']}</code>，"
                f"semantic_hits={event['semantic_hits']}，prior_hits={event['prior_hits']}</p>"
            )
            parts.append("<ol>")
            for item in event["state_history"]:
                parts.append(
                    f"<li>t={item.get('timestamp')} → {item.get('state')}"
                    f" ({item.get('reason', '')})</li>"
                )
            parts.append("</ol>")
    parts.append("</body></html>")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    scenarios = ("semantic_only", "prior_only", "fused")
    runs = [run_scenario(name) for name in scenarios]
    verdicts = {run["scenario"]: verify(run["scenario"], run) for run in runs}

    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "trace.json").write_text(
        json.dumps({"runs": runs, "verdicts": verdicts}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "trace.html").write_text(
        render_html(runs, verdicts), encoding="utf-8"
    )

    lines = ["# V3.3 双通道 — 三场景事件 trace 报告", ""]
    lines.append("| 场景 | 期望来源 | 实际来源 | 事件数 | evidence_kind | 关闭原因 | 判定 |")
    lines.append("|---|---|---|---|---|---|---|")
    for name in scenarios:
        verdict = verdicts[name]
        event = runs[scenarios.index(name)]["events"]
        lines.append(
            f"| {name} | `{verdict['expected']['source']}` | "
            f"`{','.join(verdict['observed_sources'])}` | {len(event)} | "
            f"`{event[0]['evidence_kind'] if event else '-'}` | "
            f"`{event[0]['closed_reason'] if event else '-'}` | "
            f"{'通过' if verdict['passed'] else '**失败**'} |"
        )
    lines.append("")
    lines.append("显示时间戳（确认后才允许上屏）：")
    for name in scenarios:
        lines.append(
            f"- {name}: {verdicts[name]['observed_display_ticks']}"
        )
    lines.append("")
    lines.append("逐时间戳明细与状态机历史见 `trace.json`，可读视图见 `trace.html`。")
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for name in scenarios:
        verdict = verdicts[name]
        print(f"{name:<14} passed={verdict['passed']} "
              f"sources={verdict['observed_sources']} "
              f"checks={verdict['checks']}")
    if not all(verdicts[name]["passed"] for name in scenarios):
        raise SystemExit("three-scenario proof FAILED")
    print(f"\nwrote {out_dir}/REPORT.md, trace.json, trace.html")


if __name__ == "__main__":
    main()
