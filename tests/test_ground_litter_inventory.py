from pathlib import Path
import tempfile
import unittest

from rtsp_annotator.event_engine import NormalizedRect
from rtsp_annotator.ground_litter_inventory import ConfirmedLitter, GroundLitterInventory


ITEM = ConfirmedLitter(NormalizedRect(.40, .60, .01, .02), "region-A", "merchant-A")


class GroundLitterInventoryTests(unittest.TestCase):
    def ledger(self, **options):
        value = GroundLitterInventory(**options)
        self.addCleanup(value.close)
        return value

    def test_present_for_hour_still_one_record(self):
        ledger = self.ledger()
        first = ledger.observe("camera", "fixed", 0, (ITEM,)).created_ids
        for t in (30, 300, 600, 3600):
            result = ledger.observe("camera", "fixed", t, (ITEM,))
            self.assertEqual(result.item_ids, first)
            self.assertFalse(result.created_ids)

    def test_outage_or_model_miss_does_not_mean_cleared(self):
        ledger = self.ledger()
        first = ledger.observe("camera", "fixed", 0, (ITEM,)).created_ids
        ledger.observe("camera", "fixed", 60)
        ledger.observe("camera", "fixed", 3600, scene_observable=False, clear_item_ids=first)
        result = ledger.observe("camera", "fixed", 3601, (ITEM,))
        self.assertEqual(result.item_ids, first)
        self.assertFalse(result.created_ids)

    def test_restart_preserves_item_and_discards_partial_clear(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inventory.sqlite3"
            ledger = GroundLitterInventory(path, clear_seconds=5)
            first = ledger.observe("camera", "fixed", 0, (ITEM,)).created_ids
            for t in range(1, 5):
                ledger.observe("camera", "fixed", t, clear_item_ids=first)
            ledger.close()
            ledger = GroundLitterInventory(path, clear_seconds=5)
            try:
                self.assertFalse(ledger.observe("camera", "fixed", 100, clear_item_ids=first).cleared_ids)
                result = ledger.observe("camera", "fixed", 101, (ITEM,))
                self.assertEqual(result.item_ids, first)
                self.assertFalse(result.created_ids)
            finally:
                ledger.close()

    def test_healthy_clear_evidence_then_new_item_at_same_place(self):
        ledger = self.ledger(clear_seconds=5)
        first = ledger.observe("camera", "fixed", 0, (ITEM,)).created_ids
        for t in range(1, 6):
            self.assertFalse(ledger.observe("camera", "fixed", t, clear_item_ids=first).cleared_ids)
        self.assertEqual(ledger.observe("camera", "fixed", 6, clear_item_ids=first).cleared_ids, first)
        second = ledger.observe("camera", "fixed", 7, (ITEM,)).created_ids
        self.assertEqual(len(second), 1)
        self.assertNotEqual(second, first)

    def test_occlusion_and_long_gap_reset_clear_evidence(self):
        ledger = self.ledger(clear_seconds=5)
        first = ledger.observe("camera", "fixed", 0, (ITEM,)).created_ids
        for t in (1, 2, 3):
            ledger.observe("camera", "fixed", t, clear_item_ids=first)
        ledger.observe("camera", "fixed", 4, clear_item_ids=first, scene_observable=False)
        self.assertFalse(ledger.observe("camera", "fixed", 6, clear_item_ids=first).cleared_ids)
        self.assertFalse(ledger.observe("camera", "fixed", 100, clear_item_ids=first).cleared_ids)

    def test_two_nearby_objects_are_not_merged(self):
        ledger = self.ledger()
        near = ConfirmedLitter(NormalizedRect(.413, .60, .01, .02), "region-A", "merchant-A")
        result = ledger.observe("camera", "fixed", 0, (ITEM, near))
        self.assertEqual(len(set(result.created_ids)), 2)
        reversed_result = ledger.observe("camera", "fixed", 1, (near, ITEM))
        self.assertEqual(reversed_result.item_ids, tuple(reversed(result.item_ids)))

    def test_region_boundary_jitter_does_not_change_owner(self):
        ledger = self.ledger()
        first = ledger.observe("camera", "fixed", 0, (ITEM,)).created_ids
        jitter = ConfirmedLitter(NormalizedRect(.402, .60, .01, .02), "region-B", "merchant-B")
        result = ledger.observe("camera", "fixed", 1, (jitter,))
        self.assertEqual(result.item_ids, first)
        self.assertEqual(ledger.get(first[0])["merchant_id"], "merchant-A")

    def test_camera_scopes_are_independent(self):
        ledger = self.ledger()
        a = ledger.observe("camera-A", "fixed", 0, (ITEM,)).created_ids
        b = ledger.observe("camera-B", "fixed", 0, (ITEM,)).created_ids
        self.assertNotEqual(a, b)

    def test_duplicate_timestamp_cannot_accumulate_evidence(self):
        ledger = self.ledger()
        ledger.observe("camera", "fixed", 0, (ITEM,))
        with self.assertRaises(ValueError):
            ledger.observe("camera", "fixed", 0, (ITEM,))

    def test_source_health_event_clears_timers_without_advancing_frame_clock(self):
        ledger = self.ledger(clear_seconds=5)
        ids=ledger.observe('camera','fixed',10,(ITEM,)).item_ids
        ledger.observe('camera','fixed',11,clear_item_ids=ids)
        ledger.mark_unobserved('camera','fixed')
        row=ledger.get(ids[0])
        self.assertEqual(row['state'],'unobserved')
        self.assertIsNone(row['clear_started'])
        self.assertEqual(ledger.observe('camera','fixed',11.1,(ITEM,)).item_ids,ids)

    def test_capacity_error_does_not_forget_existing_items(self):
        ledger = self.ledger(max_active_items=1)
        first = ledger.observe("camera", "fixed", 0, (ITEM,)).created_ids
        far = ConfirmedLitter(NormalizedRect(.8, .8, .01, .02), "region-A")
        with self.assertRaises(RuntimeError):
            ledger.observe("camera", "fixed", 1, (far,))
        self.assertEqual(ledger.observe("camera", "fixed", 2, (ITEM,)).item_ids, first)
