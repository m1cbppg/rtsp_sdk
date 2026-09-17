from pathlib import Path
import tempfile
import unittest

import numpy as np

from rtsp_annotator.ground_litter_review import ReviewCollector
from rtsp_annotator.ground_litter_runtime import EvidenceWindow
from rtsp_annotator.ground_litter_inventory import ConfirmedLitter
from rtsp_annotator.event_engine import NormalizedRect


class ReviewTests(unittest.TestCase):
    def test_samples_include_misses_and_late_candidates_with_bounded_files(self):
        with tempfile.TemporaryDirectory() as directory:
            collector=ReviewCollector(directory,limit=4,audit_interval=10)
            frame=np.full((80,80,3),120,np.uint8)
            for i in range(200):
                proposals=[{'box':[10,10,30,30],'label':'Paper'}] if i%2 else []
                collector.consider(frame,frame,proposals,i,i,{})
            collector.save()
            entries=collector.entries()
            self.assertEqual(len(entries),8)
            self.assertTrue(all(not x['candidates'] for x in entries if x['kind']=='audit'))
            self.assertTrue(any(x['source_time_seconds']>100 for x in entries if x['kind']=='candidate'))
            self.assertLessEqual(len(list(Path(directory).rglob('*.jpg'))),12)
            for row in entries:
                self.assertTrue((Path(directory)/row['image']).is_file())

    def test_confirmation_diagnostics_explain_duration_occlusion_and_gaps(self):
        camera={'analysis_fps':1,'day':{'confirm_seconds':4},
                'evidence':{'hit_window':5,'minimum_hits':3,'minimum_hit_fraction':.6,'actor_clear_seconds':1}}
        window=EvidenceWindow(camera,'day')
        rect=NormalizedRect(.4,.6,.01,.02)
        item=ConfirmedLitter(rect,'ground')
        self.assertFalse(window.observe(0,[item]))
        self.assertIn('confirm_duration',window.diagnostics[0]['reasons'])
        self.assertFalse(window.observe(1,[item],[rect]))
        self.assertIn('actor_occluded',window.diagnostics[0]['reasons'])
        for t in (2,3): self.assertFalse(window.observe(t,[item]))
        for t in (4,5): self.assertFalse(window.observe(t,[item]))
        self.assertTrue(window.observe(6,[item]))
        self.assertEqual(window.diagnostics[0]['reasons'],[])
        self.assertFalse(window.observe(10,[item]))
        self.assertEqual(window.reset_reason,'observation_gap')
        self.assertEqual(window.diagnostics[0]['span_seconds'],0)


if __name__=='__main__':
    unittest.main()
