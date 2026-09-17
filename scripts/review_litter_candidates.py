#!/usr/bin/env python3
"""Compare localized matches on selected objects, not a precision/recall benchmark."""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--references", type=Path, required=True)
    ap.add_argument("--reports", type=Path, nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    references = json.loads(args.references.read_text())
    results = {"reference_metadata": references, "models": {}}
    for path in args.reports:
        report = json.loads(path.read_text())
        if "summary" not in report:
            raise ValueError(f"Incomplete inference report: {path}")
        frames = {f["image"]: f["detections"] for f in report["frames"]}
        if any(r["image"] not in frames for r in references["references"]):
            raise ValueError(f"Missing reference frame in {path}")
        runs = []
        for conf in [.15, .25, .4]:
            for minimum_iou in [.3, .5]:
                counts = defaultdict(lambda: {"matches": 0, "appearances": 0})
                for ref in references["references"]:
                    # At most one count for a reference, regardless of duplicates.
                    matched = any(p["confidence"] >= conf and iou(ref["box"], p["box"]) >= minimum_iou
                                  for p in frames[ref["image"]])
                    counts[ref["id"]]["matches"] += int(matched)
                    counts[ref["id"]]["appearances"] += 1
                runs.append({"confidence": conf, "minimum_iou": minimum_iou, "counts": dict(counts),
                             "matches": sum(c["matches"] for c in counts.values()),
                             "appearances": sum(c["appearances"] for c in counts.values())})
        results["models"][path.parent.name] = {"report": str(path), "sha256": report["sha256"],
                                              "mode": report["mode"], "checks": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    for name, model in results["models"].items():
        print(name, [(c["confidence"], c["minimum_iou"], c["matches"], c["appearances"])
                     for c in model["checks"]])


if __name__ == "__main__":
    main()
