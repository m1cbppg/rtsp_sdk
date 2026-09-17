"""Build a conservative report from a local demo collection or trace JSONL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for path in sorted(args.input.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    session = args.input / "session.json"
    metadata = json.loads(session.read_text(encoding="utf-8")) if session.is_file() else {}
    actions = [r for r in records if r.get("event", "").startswith("camera_control.")]
    zoom_out = [r for r in actions if int(r.get("parameters", {}).get("zoom_delta", 0)) < 0]
    homes = [r for r in actions if r.get("parameters", {}).get("reason") in {"task_finalization", "automatic_home"}]
    report = {
        "scope": "offline trace accounting; no claim of image IoU, OCR accuracy, or device effect",
        "stream_id": metadata.get("stream_id"),
        "trace_records": len(records),
        "camera_actions": len(actions),
        "negative_zoom_actions": len(zoom_out),
        "automatic_home_actions": len(homes),
        "observation_discard_events": sum(1 for r in records if "discard" in str(r.get("event", ""))),
        "manual_review": "Required for target identity, frame IoU, clipping and vessel-number readability.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("# Demo continuous tracking report\n\n" + "\n".join(f"- **{k}**: {v}" for k, v in report.items()) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
