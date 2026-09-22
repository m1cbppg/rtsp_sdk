"""Human-reviewed episode evaluation, independent of inference and delivery.

Candidate counts are descriptive only. Missing source/time/truth evidence blocks
all quality metrics. A report can never authorize production or notifications.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path


METRICS = ("record_precision", "visible_episode_recall", "false_records_per_camera_hour",
           "duplicate_records", "merged_episodes", "erroneous_clears",
           "clear_recall", "confirmation_delay_p95_seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_logs(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    """Explicit snapshots (including rotations/restarts); never open SQLite."""
    rows, fingerprints = [], []
    for path in paths:
        fingerprints.append({"sha256": sha256_file(path), "bytes": path.stat().st_size})
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError()
                except ValueError:
                    raise ValueError(f"invalid JSON log at line {number}") from None
                rows.append(row)
    return rows, fingerprints


def _number(value, name: str, *, minimum: float | None = 0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} is below minimum")
    return float(value)


def _text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value


def _time(value) -> datetime:
    result = datetime.fromisoformat(_text(value, "anchor time"))
    if result.tzinfo is None:
        raise ValueError("anchor times require timezone")
    return result


def _ratio(n: int, d: int) -> dict:
    if not d:
        return {"value": None, "numerator": n, "denominator": d, "wilson_95": None,
                "reason": "无法计算：分母为零"}
    p, z = n / d, 1.959963984540054
    center = (p + z*z/(2*d)) / (1 + z*z/d)
    radius = z * math.sqrt(p*(1-p)/d + z*z/(4*d*d)) / (1 + z*z/d)
    return {"value": p, "numerator": n, "denominator": d,
            "wilson_95": [max(0, center-radius), min(1, center+radius)]}


def make_truth_template(camera: dict, mode: str) -> dict:
    """No inferred labels, dates, verified flags, or invented placements."""
    return {
        "schema_version": 1, "camera_id": camera["device_code"],
        "view_id": camera["view_id"], "mode": mode, "split": "validation",
        "duration_seconds": None, "confirm_seconds": camera[mode]["confirm_seconds"],
        "clear_seconds": 60, "reviewer": None, "reviewed_at": None,
        "coverage_complete": False, "records_review_complete": False,
        "source": {"media_sha256": None, "camera_verified": False,
                   "camera_evidence": None, "anchors": []},
        "log_fingerprints": [], "timeline": [], "initial_item_ids": [],
        "episodes": [], "record_reviews": [],
    }


def prepare_review(truth: dict, rows: list[dict], fingerprints: list[dict]) -> dict:
    """Import identities/provenance only; never infer physical objects or labels."""
    import copy
    draft = copy.deepcopy(truth)
    draft.update(reviewer=None, reviewed_at=None, records_review_complete=False,
                 log_fingerprints=fingerprints, timeline=[], record_reviews=[])
    identities, segments = set(), {}
    for row in rows:
        if row.get("event") != "observation" or row.get("camera_id") != truth["camera_id"]:
            continue
        for key in ("created_item_ids", "visible_item_ids", "cleared_item_ids"):
            identities.update(row.get(key, []))
        key = (row.get("run_id"), row.get("generation"))
        segments[key] = {"run_id": key[0], "generation": key[1], "field": "source_time_seconds",
                         "offset_seconds": 0, "verified": False}
    draft["timeline"] = list(segments.values())
    draft["record_reviews"] = [{"item_id": identity, "label": "uncertain", "episode_ids": [],
                                 "evidence": None} for identity in sorted(identities)]
    return draft


def _validate_truth(truth: dict) -> list[str]:
    """Pending fields block metrics; malformed supplied annotations are errors."""
    if truth.get("schema_version") != 1:
        raise ValueError("unsupported truth schema")
    camera = _text(truth.get("camera_id"), "camera_id")
    if len(camera) != 20 or not camera.isdigit():
        raise ValueError("camera_id must contain 20 digits")
    _text(truth.get("view_id"), "view_id")
    if truth.get("mode") not in {"day", "night"}:
        raise ValueError("mode must be day or night")
    if truth.get("split") not in {"calibration", "validation"}:
        raise ValueError("split must be calibration or validation")
    for key in ("confirm_seconds", "clear_seconds"):
        if _number(truth.get(key), key) <= 0:
            raise ValueError(f"{key} must be positive")
    blockers = []
    duration = truth.get("duration_seconds")
    if duration is None:
        blockers.append("尚未填写录像时长")
    elif _number(duration, "duration_seconds") <= 0:
        raise ValueError("duration_seconds must be positive")
    if not truth.get("reviewer") or not truth.get("reviewed_at"):
        blockers.append("缺少人工复核人或复核时间")
    else:
        _text(truth["reviewer"], "reviewer")
        _time(truth["reviewed_at"])
    for key, reason in (("coverage_complete", "未穷尽标注整个验收时段的地面垃圾"),
                        ("records_review_complete", "持续记录尚未全部人工复核")):
        if truth.get(key) is not True:
            blockers.append(reason)
    source = truth.get("source", {})
    digest = source.get("media_sha256")
    if not digest:
        blockers.append("缺少原始录像 SHA-256")
    elif len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("invalid media_sha256")
    if source.get("camera_verified") is not True or not source.get("camera_evidence"):
        blockers.append("画面机位尚未经人工核对")
    anchors = source.get("anchors", [])
    if len(anchors) < 2:
        blockers.append("至少需要两处画面时间/现场动作核验锚点")
    previous = None
    for anchor in anchors:
        at = _number(anchor.get("media_seconds"), "anchor media_seconds")
        expected, observed = _time(anchor.get("expected_time")), _time(anchor.get("observed_time"))
        _text(anchor.get("evidence"), "anchor evidence")
        if abs((expected-observed).total_seconds()) > 2:
            blockers.append("画面时间与预期相差超过 2 秒")
        if duration is not None and at > duration:
            raise ValueError("anchor outside clip")
        if previous:
            prev_at, prev_time = previous
            if at <= prev_at or abs((expected-prev_time).total_seconds()-(at-prev_at)) > 2:
                blockers.append("时间锚点不连续或与媒体时间轴不一致")
        previous = at, expected
    if not truth.get("timeline"):
        blockers.append("缺少日志到录像时间轴的人工核验映射")
    if not truth.get("log_fingerprints"):
        blockers.append("缺少日志快照指纹")
    episodes = truth.get("episodes", [])
    ids = set()
    for episode in episodes:
        identity = _text(episode.get("episode_id"), "episode_id")
        if identity in ids:
            raise ValueError("duplicate episode_id")
        ids.add(identity)
        if episode.get("category") not in {"paper", "plastic_bag", "bottle", "other_litter"}:
            raise ValueError("invalid ground litter category")
        _text(episode.get("region_id"), "region_id")
        box = episode.get("box")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("box must be normalized xyxy")
        x, y, right, bottom = [_number(v, "box") for v in box]
        if not 0 <= x < right <= 1 or not 0 <= y < bottom <= 1:
            raise ValueError("box must be normalized xyxy")
        if type(episode.get("eligible")) is not bool:
            raise ValueError("episode eligibility needs a human decision")
        if not episode["eligible"]:
            _text(episode.get("exclusion_reason"), "exclusion_reason")
        intervals = episode.get("intervals", [])
        if not intervals:
            raise ValueError("episode requires visibility intervals")
        last_end = -1.0
        for interval in intervals:
            start = _number(interval.get("start"), "interval start")
            end = _number(interval.get("end"), "interval end")
            if start >= end or start < last_end or (duration is not None and end > duration):
                raise ValueError("overlapping or out-of-range intervals")
            if interval.get("state") not in {"visible", "occluded", "unknown", "clean"}:
                raise ValueError("invalid visibility state")
            last_end = end
    reviewed = set()
    for review in truth.get("record_reviews", []):
        identity = _text(review.get("item_id"), "item_id")
        if identity in reviewed:
            raise ValueError("duplicate item review")
        reviewed.add(identity)
        label = review.get("label")
        if label not in {"true_litter", "false_positive", "uncertain"}:
            raise ValueError("invalid record review label")
        links = review.get("episode_ids", [])
        if len(links) != len(set(links)) or any(link not in ids for link in links):
            raise ValueError("invalid episode link")
        if (label == "true_litter") != bool(links):
            raise ValueError("only true_litter reviews must link physical episodes")
        if label != "uncertain":
            _text(review.get("evidence"), "record review evidence")
    return list(dict.fromkeys(blockers))


def evaluate(truth: dict, rows: list[dict], fingerprints: list[dict],
             *, media_sha256: str | None = None) -> dict:
    blockers = _validate_truth(truth)
    observations = [row for row in rows if row.get("event") == "observation"
                    and row.get("camera_id") == truth["camera_id"]]
    report = {
        "schema_version": 1, "camera_id": truth["camera_id"], "mode": truth["mode"],
        "split": truth["split"], "status": "无法计算", "blockers": blockers,
        "descriptive": {"observations": len(observations),
                        "candidate_boxes": sum(len(r.get("candidates", [])) for r in observations)},
        "metrics": {key: None for key in METRICS},
        "accuracy": None, "false_positive_rate": None,
        "limitations": ["候选框数量不是识别率；未计算逐框精度/召回率。",
                        "没有定义真负样本，整体准确率及 FP/(FP+TN) 无法计算。",
                        "置信区间仅描述样本不确定性；同场景相关样本不能代表生产总体。"],
        "production_go": False, "notifications_go": False,
    }
    if not observations:
        blockers.append("没有该机位的有效观测日志")
    if media_sha256 is None or media_sha256 != truth.get("source", {}).get("media_sha256"):
        blockers.append("未提供原始录像或录像指纹不匹配")
    if fingerprints != truth.get("log_fingerprints"):
        blockers.append("日志快照与人工复核绑定的指纹不一致")
    if blockers:
        return report

    mappings = {}
    for segment in truth["timeline"]:
        key = (_text(segment.get("run_id"), "run_id"), segment.get("generation"))
        if key in mappings:
            raise ValueError("duplicate timeline segment")
        if segment.get("field") not in {"source_time_seconds", "capture_monotonic"}:
            raise ValueError("unsupported timeline field")
        _number(segment.get("offset_seconds"), "offset_seconds", minimum=None)
        if segment.get("verified") is not True:
            blockers.append("时间轴映射尚未经人工核验")
        mappings[key] = segment
    timed, previous = [], {}
    for row in observations:
        key = (row.get("run_id"), row.get("generation"))
        mapping = mappings.get(key)
        if mapping is None:
            blockers.append("日志存在未映射的 run_id/generation")
            continue
        at = _number(row.get(mapping["field"]), "log timestamp") + mapping["offset_seconds"]
        if not 0 <= at < truth["duration_seconds"]:
            blockers.append("日志时间超出已标注录像范围")
        if key in previous and at <= previous[key]:
            blockers.append("日志时间倒退、重复或快照顺序错误")
        if row.get("mode", truth["mode"]) != truth["mode"]:
            blockers.append("日志昼夜模式与真值不一致")
        previous[key] = at
        timed.append((at, row))
    if blockers:
        report["blockers"] = list(dict.fromkeys(blockers))
        return report
    timed.sort(key=lambda pair: pair[0])
    seen, created, clears = defaultdict(list), defaultdict(list), []
    for at, row in timed:
        for key in ("visible_item_ids", "created_item_ids", "cleared_item_ids"):
            values = row.get(key)
            if not isinstance(values, list) or len(values) != len(set(values)):
                raise ValueError("item IDs must be unique lists")
            for identity in values:
                _text(identity, "log item_id")
        for identity in set(row["visible_item_ids"] + row["created_item_ids"]):
            seen[identity].append(at)
        for identity in row["created_item_ids"]:
            created[identity].append(at)
        clears.extend((identity, at) for identity in row["cleared_item_ids"])
    all_ids = set(seen) | {identity for identity, _ in clears}
    reviews = {r["item_id"]: r for r in truth["record_reviews"]}
    initial = set(truth.get("initial_item_ids", []))
    if all_ids != set(reviews):
        blockers.append("记录复核必须恰好覆盖日志中全部 item_id（含遗留记录）")
    if all_ids - set(created) - initial:
        blockers.append("存在没有创建证据或初始库存声明的 item_id")
    if initial & set(created) or any(len(times) != 1 for times in created.values()):
        blockers.append("同一 item_id 重复创建或与初始库存冲突，日志需复核")
    if any(r["label"] == "uncertain" for r in reviews.values()):
        blockers.append("仍有 uncertain 记录，不能当真或当假计分")
    if blockers:
        return report

    episodes = {e["episode_id"]: e for e in truth["episodes"]}
    links, delays = defaultdict(set), []
    reviewed_links = defaultdict(set)
    for identity, review in reviews.items():
        for eid in review["episode_ids"]:
            reviewed_links[eid].add(identity)
    eligible = set()
    for eid, episode in episodes.items():
        visible = [i for i in episode["intervals"] if i["state"] == "visible"]
        if episode["eligible"] and any(i["end"]-i["start"] >= truth["confirm_seconds"] for i in visible):
            eligible.add(eid)
        for identity, review in reviews.items():
            if eid not in review["episode_ids"]:
                continue
            hits = [at for at in seen[identity] if any(i["start"] <= at < i["end"] for i in visible)]
            if hits:
                links[eid].add(identity)
        if links[eid] and eid in eligible:
            first = min(at for identity in links[eid] for at in seen[identity]
                        if any(i["start"] <= at < i["end"] for i in visible))
            delays.append(first - min(i["start"] for i in visible))
    matched = eligible & {eid for eid, ids in links.items() if ids}
    false_records = sum(reviews[identity]["label"] == "false_positive" for identity in created)
    true_records = len(created) - false_records
    errors, assessed_clears, correct_clears = [], 0, set()
    clear_eligible = {eid for eid in matched if any(i["state"] == "clean" and
                      i["end"]-i["start"] >= truth["clear_seconds"] for i in episodes[eid]["intervals"])}
    for identity, at in clears:
        episode_ids = reviews[identity]["episode_ids"]
        if not episode_ids:
            continue  # False records do not establish litter cleanup performance.
        assessed_clears += 1
        past = [eid for eid in episode_ids if episodes[eid]["intervals"][0]["start"] <= at]
        eid = max(past, key=lambda e: episodes[e]["intervals"][0]["start"]) if past else None
        clean = eid is not None and any(
            i["state"] == "clean" and i["start"] + truth["clear_seconds"] <= at < i["end"]
            for i in episodes[eid]["intervals"])
        if clean:
            correct_clears.add(eid)
        else:
            errors.append({"item_id": identity, "episode_id": eid, "at": at})
    metrics = {
        "record_precision": _ratio(true_records, len(created)),
        "visible_episode_recall": _ratio(len(matched), len(eligible)),
        "false_records_per_camera_hour": false_records / (truth["duration_seconds"] / 3600),
        "duplicate_records": sum(max(0, len(ids)-1) for ids in reviewed_links.values()),
        "merged_episodes": sum(max(0, len(r["episode_ids"])-1) for r in reviews.values()),
        "erroneous_clears": {"count": len(errors), "assessed_links": assessed_clears, "details": errors},
        "clear_recall": _ratio(len(correct_clears & clear_eligible), len(clear_eligible)),
        "confirmation_delay_p95_seconds": sorted(delays)[math.ceil(.95*len(delays))-1] if delays else None,
    }
    report.update(status="已计算样本指标，尚非生产验收", metrics=metrics,
                  counts={"new_records": len(created), "initial_records": len(initial),
                          "episodes": len(episodes), "eligible_episodes": len(eligible),
                          "excluded_or_insufficient_visibility": len(episodes)-len(eligible),
                          "false_records": false_records},
                  missed_episode_ids=sorted(eligible-matched))
    report["limitations"].append("零次错误清理且 assessed_links=0 表示未测到清理，不表示清理通过。")
    return report


def render_report(report: dict) -> str:
    lines = ["# 零散垃圾真值验收", "", f"机位：{report['camera_id']}；{report['mode']}；{report['split']}",
             "", f"状态：**{report['status']}**", "",
             "生产接入：No-Go；通知：No-Go。本工具只计算样本指标，不自动批准上线。", ""]
    if report["blockers"]:
        lines.extend(["待补齐：", "", *[f"- {b}" for b in report["blockers"]], ""])
    lines.extend(["```json", json.dumps({"descriptive": report["descriptive"], "metrics": report["metrics"]},
                                      ensure_ascii=False, indent=2), "```", ""])
    lines.extend(f"- {line}" for line in report["limitations"])
    return "\n".join(lines) + "\n"
