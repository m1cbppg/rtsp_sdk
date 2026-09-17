"""Collapse frame-level review labels into spatially persistent litter items."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _center(box):
    x, y, r, b = box
    return ((x + r) / 2.0, (y + b) / 2.0)


def deduplicate(labels, rows, *, radius_px=80.0):
    """Return item clusters and audit metadata; labels remain frame-level evidence.

    A cluster joins boxes from the same camera/view whose centers are within
    ``radius_px``. This is deliberately conservative and does not claim
    cross-camera identity or infer that a missing frame means cleanup.
    """
    by_id = {str(item.get("item_id")): item for item in labels}
    observations = []
    for row in rows:
        label = by_id.get(str(row.get("sample_id")))
        if not label:
            continue
        for index, candidate in enumerate(row.get("candidates", [])):
            box = candidate.get("box")
            if not isinstance(box, list) or len(box) != 4:
                continue
            cx, cy = _center(box)
            observations.append({"frame_id": str(row["sample_id"]), "candidate_index": index,
                                "label": label.get("label", "uncertain"), "note": label.get("note", ""),
                                "box": box, "center_px": [cx, cy],
                                "source_time_seconds": row.get("source_time_seconds")})
    clusters = []
    radius_sq = float(radius_px) ** 2
    for obs in observations:
        matches = []
        for index, cluster in enumerate(clusters):
            # Compare to every prior observation, not a drifting centroid.
            if any((obs["center_px"][0] - old["center_px"][0]) ** 2
                   + (obs["center_px"][1] - old["center_px"][1]) ** 2 <= radius_sq
                   for old in cluster):
                matches.append(index)
        if not matches:
            clusters.append([obs])
        else:
            first = matches[0]
            clusters[first].append(obs)
            for index in reversed(matches[1:]):
                clusters[first].extend(clusters.pop(index))
    clusters.sort(key=lambda group: (min(o["center_px"][1] for o in group),
                                     min(o["center_px"][0] for o in group)))
    items = []
    for number, group in enumerate(clusters, 1):
        items.append({"item_id": f"item-{number:03d}",
                      "label_set": sorted({o["label"] for o in group}),
                      "frame_count": len({o["frame_id"] for o in group}),
                      "observation_count": len(group),
                      "frame_ids": sorted({o["frame_id"] for o in group}),
                      "first_frame": min(o["frame_id"] for o in group),
                      "last_frame": max(o["frame_id"] for o in group),
                      "center_range_px": {
                          "x_min": min(o["center_px"][0] for o in group),
                          "x_max": max(o["center_px"][0] for o in group),
                          "y_min": min(o["center_px"][1] for o in group),
                          "y_max": max(o["center_px"][1] for o in group)},
                      "observations": group})
    return {"items": items, "input_frame_labels": len(labels),
            "input_candidate_observations": len(observations),
            "deduplicated_item_count": len(items), "radius_px": radius_px,
            "accuracy": None, "cleanup": None,
            "note": "Frame labels were supplied by the reviewer. Spatial clustering removes repeated views; it does not prove object identity during unseen intervals."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True,
                        help="Directory containing inference_part*/results.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius-px", type=float, default=80)
    args = parser.parse_args()
    if not math.isfinite(args.radius_px) or not 1 <= args.radius_px <= 500:
        parser.error("radius-px must be in [1,500]")
    labels = json.loads(args.labels.read_text(encoding="utf-8")).get("labels", [])
    rows = []
    for path in sorted(args.inference.glob("inference_part*/results.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    result = deduplicate(labels, rows, radius_px=args.radius_px)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("input_frame_labels", "input_candidate_observations", "deduplicated_item_count", "radius_px")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
