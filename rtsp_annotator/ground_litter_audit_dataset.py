"""High-recall review-corpus helpers for fixed-camera litter audits.

This module deliberately separates *proposal* from *classification*.  A proposal may
be wrong; its job is to put a small, reviewable patch in front of a human without
letting any one detector define the dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class Proposal:
    frame_id: str
    bbox: tuple[int, int, int, int]
    source: str
    score: float


def select_files_evenly(files: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Select files nearest evenly spaced wall-clock targets, deterministically."""
    rows = sorted(files, key=lambda row: (row["record_start"], row["file_id"]))
    if count <= 0 or not rows:
        return []
    if len(rows) <= count:
        return rows
    stamps = [datetime.strptime(row["record_start"], "%Y-%m-%d %H:%M:%S").timestamp() for row in rows]
    lo, hi = stamps[0], stamps[-1]
    chosen: set[int] = set()
    for slot in range(count):
        target = lo + (slot + 0.5) * (hi - lo) / count
        index = min((i for i in range(len(rows)) if i not in chosen), key=lambda i: abs(stamps[i] - target))
        chosen.add(index)
    return [rows[i] for i in sorted(chosen)]


def polygon_mask(shape: tuple[int, int], points: Sequence[Sequence[float]]) -> np.ndarray:
    height, width = shape
    poly = np.asarray([[round(x * width), round(y * height)] for x, y in points], np.int32)
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return mask


def expanded_mask(core: np.ndarray, radius_fraction: float = 0.08) -> np.ndarray:
    radius = max(3, round(min(core.shape) * radius_fraction))
    size = radius * 2 + 1
    return cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))


def box_iou(a: Sequence[int], b: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = max(1.0, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return inter / union


def temporal_proposals(frame_id: str, before: np.ndarray, current: np.ndarray, after: np.ndarray,
                       mask: np.ndarray, limit: int = 4) -> list[Proposal]:
    """Find local change blobs. Motion false positives are expected review material."""
    def gray(image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    a = cv2.absdiff(gray(before), gray(current))
    b = cv2.absdiff(gray(current), gray(after))
    change = cv2.max(a, b)
    change[mask == 0] = 0
    _, binary = cv2.threshold(change, 22, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binary = cv2.dilate(binary, np.ones((7, 7), np.uint8), iterations=1)
    rows: list[Proposal] = []
    height, width = current.shape[:2]
    for contour in cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < 36 or area > width * height * 0.06 or max(w, h) > min(width, height) * 0.35:
            continue
        pad = max(6, round(max(w, h) * 0.35))
        box = (max(0, x-pad), max(0, y-pad), min(width, x+w+pad), min(height, y+h+pad))
        score = float(change[y:y+h, x:x+w].mean()) * math.sqrt(area)
        rows.append(Proposal(frame_id, box, "temporal", score))
    return sorted(rows, key=lambda item: item.score, reverse=True)[:limit]


def texture_proposals(frame_id: str, image: np.ndarray, mask: np.ndarray,
                      sizes: Sequence[int] = (48, 72, 112, 168), limit: int = 4) -> list[Proposal]:
    """Rank small windows by edge/contrast energy, independent of object categories."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.GaussianBlur(cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_32F)), (5, 5), 0)
    h, w = gray.shape
    rows: list[Proposal] = []
    for size in sizes:
        step = max(16, size // 2)
        for y in range(0, max(1, h-size+1), step):
            for x in range(0, max(1, w-size+1), step):
                region_mask = mask[y:y+size, x:x+size]
                if region_mask.size == 0 or np.count_nonzero(region_mask) < region_mask.size * 0.55:
                    continue
                tile = gray[y:y+size, x:x+size]
                energy = float(lap[y:y+size, x:x+size][region_mask > 0].mean())
                contrast = float(tile[region_mask > 0].std())
                rows.append(Proposal(frame_id, (x, y, x+size, y+size), "texture", energy + 0.2 * contrast))
    picked: list[Proposal] = []
    for row in sorted(rows, key=lambda item: item.score, reverse=True):
        if all(box_iou(row.bbox, old.bbox) < 0.25 for old in picked):
            picked.append(row)
            if len(picked) >= limit:
                break
    return picked


def random_grid_proposals(frame_id: str, image: np.ndarray, mask: np.ndarray,
                          count: int = 3, seed: int = 20260921) -> list[Proposal]:
    """Coverage sample that can expose misses from every automatic proposal source."""
    h, w = image.shape[:2]
    rng = random.Random(f"{seed}:{frame_id}")
    rows: list[Proposal] = []
    sizes = (64, 96, 144, 208)
    attempts = 0
    while len(rows) < count and attempts < 500:
        attempts += 1
        size = rng.choice(sizes)
        x = rng.randrange(0, max(1, w-size))
        y = rng.randrange(0, max(1, h-size))
        patch = mask[y:y+size, x:x+size]
        if patch.size and np.count_nonzero(patch) >= patch.size * 0.65:
            box = (x, y, min(w, x+size), min(h, y+size))
            if all(box_iou(box, old.bbox) < 0.2 for old in rows):
                rows.append(Proposal(frame_id, box, "random_grid", rng.random()))
    return rows


def merge_proposals(proposals: Iterable[Proposal], per_source: dict[str, int], total: int) -> list[Proposal]:
    """Quota then cross-source dedupe. No high-volume source may crowd out coverage."""
    # Callers may pass a generator.  Materialise once because the quota pass
    # and the unused-capacity pass must inspect the same proposal population.
    proposal_rows = list(proposals)
    def source_priority(source: str) -> int:
        if source.startswith("semantic_tile"):
            return 0
        if source.startswith("semantic_full"):
            return 1
        return {"temporal": 2, "texture": 3, "random_grid": 4}.get(source, 99)
    grouped: dict[str, list[Proposal]] = {}
    for proposal in proposal_rows:
        grouped.setdefault(proposal.source, []).append(proposal)
    kept: list[Proposal] = []
    for source in sorted(grouped, key=source_priority):
        count = 0
        for row in sorted(grouped[source], key=lambda item: item.score, reverse=True):
            if count >= per_source.get(source, total):
                break
            same_frame = [old for old in kept if old.frame_id == row.frame_id]
            # Random coverage remains independent; all other sources are spatially deduplicated.
            if source != "random_grid" and any(box_iou(row.bbox, old.bbox) >= 0.55 for old in same_frame):
                continue
            kept.append(row)
            count += 1
            if len(kept) >= total:
                return kept
    # A score band may contain fewer proposals than its budget. Preserve the
    # already selected random coverage, then fill unused capacity from all
    # remaining proposals instead of silently shrinking the review batch.
    for row in sorted(proposal_rows, key=lambda item: (source_priority(item.source), -item.score)):
        if row in kept:
            continue
        same_frame = [old for old in kept if old.frame_id == row.frame_id]
        if row.source != "random_grid" and any(box_iou(row.bbox, old.bbox) >= 0.55 for old in same_frame):
            continue
        kept.append(row)
        if len(kept) >= total:
            break
    return kept


def dedupe_persistent_proposals(proposals: Iterable[Proposal]) -> list[Proposal]:
    """Conservatively collapse near-identical fixed-camera proposals across frames."""
    def family(source: str) -> str:
        return "semantic" if source.startswith("semantic_") else source
    kept: list[Proposal] = []
    for row in sorted(proposals, key=lambda item: item.score, reverse=True):
        if row.source == "random_grid":
            kept.append(row)
            continue
        x1,y1,x2,y2=row.bbox
        cx,cy=(x1+x2)/2,(y1+y2)/2
        diagonal=math.hypot(x2-x1,y2-y1)
        duplicate=False
        for old in kept:
            if old.source == "random_grid" or family(old.source) != family(row.source):
                continue
            X1,Y1,X2,Y2=old.bbox
            Cx,Cy=(X1+X2)/2,(Y1+Y2)/2
            old_diagonal=math.hypot(X2-X1,Y2-Y1)
            ratio=diagonal/max(1.0,old_diagonal)
            if box_iou(row.bbox,old.bbox)>=.45 or (
                .65<=ratio<=1.55
                and math.hypot(cx-Cx,cy-Cy)<=.18*max(diagonal,old_diagonal)
            ):
                duplicate=True
                break
        if not duplicate:
            kept.append(row)
    return kept


def dataset_fingerprint(items: Sequence[dict[str, Any]]) -> str:
    stable = [{"review_id": row["review_id"], "bbox": row["bbox"], "source": row["source"]} for row in items]
    return hashlib.sha256(json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
