import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from rtsp_annotator.ground_litter_diagnostics import timed_stage
from rtsp_annotator.ground_litter_runtime import CameraAnalysis, LitterModel
from scripts.analyze_ground_litter_run import analyze


def tensor(rows):
    return SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: rows))


def result(rows):
    return SimpleNamespace(boxes=SimpleNamespace(data=tensor(rows),
                           xyxy=tensor([r[:4] for r in rows])), names={0:'Paper'})


class DiagnosticTests(unittest.TestCase):
    def test_failure_retains_stage_timing(self):
        stats={}
        with self.assertRaises(RuntimeError):
            with timed_stage(stats,'decode'):
                raise RuntimeError('failed')
        self.assertEqual(stats['stage_calls']['decode'],1)
        self.assertGreaterEqual(stats['stage_seconds']['decode'],0)

    def test_candidate_trace_separates_duplicate_roi_and_actor_rejections(self):
        model=LitterModel.__new__(LitterModel)
        model.device='cpu';model.actor_imgsz=1280;model.diagnostic_candidates=True
        # One retained item, one duplicate, one outside ROI, one inside actor.
        model.model=Mock(predict=Mock(return_value=[result([
            [30,30,70,70,.9,0],[31,31,71,71,.8,0],
            [400,400,430,430,.8,0],[120,120,160,160,.8,0]])]))
        model.actor=Mock(predict=Mock(return_value=[result([[115,115,165,165,.9,0]])]))
        mask=np.zeros((640,640),np.uint8);mask[:300,:300]=1
        camera={'night':{'minimum_confidence':.2},'zones':[{
            'region_id':'ground','merchant_id':None,'minimum_short_side_px':12,
            'minimum_box_area_px':160}]}
        kept,_=model.analyze(np.zeros((640,640,3),np.uint8),camera,
                             [(0,0,640,640)],{'ground':mask},'night')
        self.assertEqual([d['box'] for d in kept],[[30,30,70,70]])
        self.assertEqual(model.last_stats['raw_before_nms'],4)
        self.assertEqual(len(model.last_stats['nms_suppressed']),1)
        self.assertEqual({d['reason'] for d in model.last_stats['rejected_candidates']},
                         {'outside_roi','actor_overlap'})
        self.assertEqual(model.last_stats['stage_calls']['actor_local'],1)
        self.assertIn('model_total',model.last_stats['stage_seconds'])

    def test_alignment_failure_never_runs_model_or_reuses_old_model_stats(self):
        analysis=CameraAnalysis.__new__(CameraAnalysis)
        frame=np.zeros((640,640,3),np.uint8)
        analysis.reference=frame;analysis.generation=1
        analysis.alignment=SimpleNamespace(check=Mock(side_effect=ValueError('view_alignment_unknown')),
                                           diagnostics={'matches':0})
        model=Mock(last_stats={'raw_before_nms':999})
        with self.assertRaisesRegex(ValueError,'view_alignment_unknown'):
            analysis.consume(model,Mock(),frame,1,1,enforce_freshness=False)
        model.analyze.assert_not_called()
        self.assertNotIn('raw_before_nms',analysis.last_diagnostics)
        self.assertIn('alignment',analysis.last_diagnostics['stage_seconds'])
        self.assertIn('analysis_total',analysis.last_diagnostics['stage_seconds'])

    def test_rejected_slow_attempts_are_included_in_latency_report(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            rows=[{'event':'observation','camera_id':'camera','inference_seconds':.3,
                   'result_age_seconds':.5,'diagnostics':{'stage_seconds':{'model_total':.2}}},
                  {'event':'rejected','camera_id':'camera','reason':'analysis_result_stale',
                   'inference_seconds':4,'result_age_seconds':5,
                   'diagnostics':{'stage_seconds':{'model_total':3}}},
                  {'event':'rejected','camera_id':'camera','reason':'legacy_missing_timing'}]
            (p/'observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            r=analyze(p)['cameras']['camera']
            self.assertEqual(r['observations'],1)
            self.assertEqual(r['result_age_seconds_p95'],.5)
            self.assertEqual(r['all_attempt_result_age_seconds_p95'],5)
            self.assertEqual(r['all_attempts_with_result_age'],2)
            self.assertEqual(r['stage_seconds']['model_total']['count'],2)


if __name__=='__main__':
    unittest.main()
