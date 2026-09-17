"""Persistent item identity experiment, independent of notifications/video IO.

An absent detector box is NOT evidence of removal. Only item-specific,
unoccluded, healthy observations explicitly marked clear can close an item.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
import json
import sqlite3
from uuid import uuid4

from .event_engine import NormalizedRect


@dataclass(frozen=True)
class ConfirmedLitter:
    rectangle: NormalizedRect
    region_id: str
    merchant_id: str | None = None


@dataclass(frozen=True)
class InventoryResult:
    item_ids: tuple[str, ...]
    created_ids: tuple[str, ...]
    cleared_ids: tuple[str, ...]


def _box(value: NormalizedRect) -> list[float]:
    values = [value.left, value.top, value.width, value.height]
    if (not all(isfinite(v) for v in values) or min(values[:2]) < 0
            or min(values[2:]) <= 0 or value.left + value.width > 1.000001
            or value.top + value.height > 1.000001):
        raise ValueError("Invalid normalized box")
    return values


def _same_location(left: NormalizedRect, right: NormalizedRect) -> bool:
    # A fraction of the OBJECT size, not 5–15% of an entire street image.
    # Fixed anchor prevents a chain of nearby litter from dragging identity.
    if max(left.width / right.width, right.width / left.width,
           left.height / right.height, right.height / left.height) > 3:
        return False
    x, y = left.center
    u, v = right.center
    distance = ((x - u) / min(left.width, right.width)) ** 2
    distance += ((y - v) / min(left.height, right.height)) ** 2
    return distance <= 0.5 ** 2


class GroundLitterInventory:
    """Single-owner SQLite ledger; new IDs are created exactly once per match.

    Call in the analysis/persistence worker, never on the main video callback.
    Timestamps use one acquisition clock per camera/view. Recovery preserves
    IDs but discards partial clear timers, so downtime cannot clear garbage.
    """

    def __init__(self, path: str | Path = ":memory:", *, clear_seconds: float = 60,
                 max_observation_gap: float = 3, max_active_items: int = 1000):
        if (not isfinite(clear_seconds) or not isfinite(max_observation_gap)
                or clear_seconds <= 0 or max_observation_gap <= 0
                or max_active_items < 1):
            raise ValueError("Invalid inventory limits")
        self.clear_seconds = clear_seconds
        self.max_observation_gap = max_observation_gap
        self.max_active_items = max_active_items
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS litter_items (
                item_id TEXT PRIMARY KEY, camera_id TEXT NOT NULL,
                view_id TEXT NOT NULL, region_id TEXT NOT NULL, merchant_id TEXT,
                anchor TEXT NOT NULL, first_seen REAL NOT NULL,
                last_seen REAL NOT NULL, state TEXT NOT NULL,
                clear_started REAL, clear_last REAL, cleared_at REAL
            );
            CREATE INDEX IF NOT EXISTS litter_scope
            ON litter_items(camera_id, view_id, state);
            UPDATE litter_items SET clear_started=NULL, clear_last=NULL, state='unobserved'
            WHERE state != 'cleared';
        """)
        self._last_timestamp: dict[tuple[str, str], float] = {}

    def close(self) -> None:
        self.db.close()

    def mark_unobserved(self, camera_id: str, view_id: str) -> None:
        """A source health event invalidates evidence, not the frame clock.

        Using wall time here can race with a frame already in the mailbox and
        make that otherwise fresh frame appear to go backwards in time.
        """
        with self.db:
            self.db.execute("UPDATE litter_items SET state='unobserved',"
                            "clear_started=NULL,clear_last=NULL WHERE camera_id=? "
                            "AND view_id=? AND state!='cleared'", (camera_id,view_id))

    def observe(self, camera_id: str, view_id: str, timestamp: float,
                items: tuple[ConfirmedLitter, ...] = (), *,
                clear_item_ids: tuple[str, ...] = (),
                scene_observable: bool = True) -> InventoryResult:
        """Only confirmed semantic observations may create IDs.

        clear_item_ids must come from localized clear-ground evidence, not
        from absence of model detections. Occlusion, bad frames and outages
        must set scene_observable=False or omit those specific clear IDs.
        """
        if not camera_id or not view_id or not isfinite(timestamp) or timestamp < 0:
            raise ValueError("Invalid observation identity/time")
        scope = (camera_id, view_id)
        previous = self._last_timestamp.get(scope)
        if previous is not None and timestamp <= previous:
            raise ValueError("Each observation must have a fresh increasing timestamp")
        if len(items) > self.max_active_items:
            raise ValueError("Too many detections in one observation")
        for item in items:
            _box(item.rectangle)
            if not item.region_id:
                raise ValueError("region_id required")
        ids: list[str] = []
        created: list[str] = []
        cleared: list[str] = []
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            rows = [dict(row) for row in self.db.execute(
                "SELECT * FROM litter_items WHERE camera_id=? AND view_id=? "
                "AND state!='cleared' ORDER BY first_seen,item_id", scope)]
            used: set[str] = set()
            for item in items if scene_observable else ():
                # One-to-one assignment: two simultaneously visible nearby
                # items cannot consume the same identity.
                matches = [row for row in rows if row["item_id"] not in used
                           and _same_location(item.rectangle, NormalizedRect(*json.loads(row["anchor"])))]
                if matches:
                    row = min(matches, key=lambda r: sum(
                        (a - b) ** 2 for a, b in zip(item.rectangle.center,
                            NormalizedRect(*json.loads(r["anchor"])).center)))
                    identity = row["item_id"]
                    # Ownership is locked at first confirmation. Boundary
                    # jitter / merchant remapping does not create a new item.
                    self.db.execute("UPDATE litter_items SET last_seen=?,state='present',"
                                    "clear_started=NULL,clear_last=NULL WHERE item_id=?",
                                    (timestamp, identity))
                else:
                    if len(rows) >= self.max_active_items:
                        raise RuntimeError("Inventory capacity reached; do not evict active IDs")
                    identity = uuid4().hex
                    row = dict(item_id=identity, camera_id=camera_id, view_id=view_id,
                               region_id=item.region_id, merchant_id=item.merchant_id,
                               anchor=json.dumps(_box(item.rectangle)), first_seen=timestamp,
                               last_seen=timestamp, state="present", clear_started=None,
                               clear_last=None, cleared_at=None)
                    self.db.execute("INSERT INTO litter_items VALUES "
                                    "(:item_id,:camera_id,:view_id,:region_id,:merchant_id,"
                                    ":anchor,:first_seen,:last_seen,:state,:clear_started,"
                                    ":clear_last,:cleared_at)", row)
                    rows.append(row)
                    created.append(identity)
                used.add(identity)
                ids.append(identity)
            clear_set = set(clear_item_ids) if scene_observable else set()
            for row in rows:
                identity = row["item_id"]
                if identity in used:
                    continue
                if identity not in clear_set:
                    self.db.execute("UPDATE litter_items SET state='unobserved',"
                                    "clear_started=NULL,clear_last=NULL WHERE item_id=?", (identity,))
                    continue
                last = row["clear_last"]
                start = row["clear_started"]
                if last is None or timestamp - last > self.max_observation_gap:
                    start = timestamp
                if timestamp - start >= self.clear_seconds:
                    self.db.execute("UPDATE litter_items SET state='cleared',cleared_at=? "
                                    "WHERE item_id=?", (timestamp, identity))
                    cleared.append(identity)
                else:
                    self.db.execute("UPDATE litter_items SET state='verifying_clear',"
                                    "clear_started=?,clear_last=? WHERE item_id=?", (start, timestamp, identity))
        self._last_timestamp[scope] = timestamp
        return InventoryResult(tuple(ids), tuple(created), tuple(cleared))

    def get(self, item_id: str) -> dict:
        row = self.db.execute("SELECT * FROM litter_items WHERE item_id=?", (item_id,)).fetchone()
        if row is None:
            raise KeyError(item_id)
        return dict(row)

    def active(self, camera_id: str, view_id: str) -> list[dict]:
        return [dict(row) for row in self.db.execute(
            "SELECT * FROM litter_items WHERE camera_id=? AND view_id=? "
            "AND state!='cleared' ORDER BY first_seen,item_id", (camera_id,view_id))]
