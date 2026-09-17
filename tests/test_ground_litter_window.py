import unittest

from rtsp_annotator.event_engine import NormalizedRect
from rtsp_annotator.ground_litter_inventory import ConfirmedLitter
from rtsp_annotator.ground_litter_runtime import EvidenceWindow


ITEM=ConfirmedLitter(NormalizedRect(.4,.6,.01,.02),'ground')
CAMERA={'analysis_fps':1,'night':{'confirm_seconds':4},'evidence':{
    'hit_window':5,'minimum_hits':3,'minimum_hit_fraction':.6,'actor_clear_seconds':1}}


class WindowTests(unittest.TestCase):
    def test_short_unknown_preserves_track_but_not_elapsed_evidence(self):
        w=EvidenceWindow(CAMERA,'night')
        for t in (0,1,2):self.assertFalse(w.observe(t,[ITEM]))
        track=w.tracks[0]
        w.unknown(3)
        self.assertFalse(w.observe(4,[ITEM]))
        self.assertIs(w.tracks[0],track)
        self.assertEqual(track.observed_seconds,2)
        self.assertFalse(w.observe(5,[ITEM]))
        self.assertTrue(w.observe(6,[ITEM]))

    def test_long_unknown_cannot_bridge_confirmation(self):
        w=EvidenceWindow(CAMERA,'night')
        for t in (0,1,2):w.observe(t,[ITEM])
        w.unknown(10)
        self.assertEqual(w.tracks,[])
        self.assertFalse(w.observe(11,[ITEM]))
        self.assertEqual(w.diagnostics[0]['observed_seconds'],0)

    def test_alternating_valid_and_unknown_cannot_confirm_by_wall_clock(self):
        w=EvidenceWindow(CAMERA,'night')
        for t in range(30):
            if t%2:w.unknown(t)
            else:self.assertFalse(w.observe(t,[ITEM]))
        self.assertEqual(w.tracks[0].observed_seconds,0)

    def test_repeated_unknown_does_not_count_as_new_frame(self):
        w=EvidenceWindow(CAMERA,'night');w.observe(1,[ITEM]);w.unknown(2)
        with self.assertRaises(ValueError):w.unknown(2)
        with self.assertRaises(ValueError):w.observe(2,[ITEM])

    def test_occlusion_at_another_position_does_not_pause_item(self):
        w=EvidenceWindow(CAMERA,'night')
        other=NormalizedRect(.8,.8,.1,.1)
        for t in range(4):self.assertFalse(w.observe(t,[ITEM],[other]))
        self.assertTrue(w.observe(4,[ITEM],[other]))


if __name__=='__main__':unittest.main()
