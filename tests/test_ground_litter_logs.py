import json
from pathlib import Path
import tempfile
import unittest
from datetime import datetime, timezone

from scripts.analyze_ground_litter_run import analyze, markdown
from rtsp_annotator.ground_litter_journal import PilotJournal
from rtsp_annotator.ground_litter_runner import analysis_mode


class GroundLitterLogTests(unittest.TestCase):
    def test_journal_lifecycle_and_rejections_are_not_detections(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = PilotJournal(directory, 'test-run')
            journal.write('run_started')
            journal.write('source_state', camera_id='camera', state='live')
            journal.write('rejected', camera_id='camera', reason='view_alignment_unknown')
            journal.write('storage', bytes=123, limit_bytes=1000)
            journal.write('run_stopped')
            journal.close()
            report = analyze(Path(directory))
            self.assertEqual(report['cameras']['camera']['observations'], 0)
            self.assertEqual(report['created_id_count'], 0)
            self.assertEqual(report['rejection_counts'], {'view_alignment_unknown': 1})
            self.assertEqual(report['source_state_counts'], {'live': 1})
            self.assertEqual(report['storage_bytes_max'], 123)
            self.assertIn('view_alignment_unknown', markdown(report))

    def test_numeric_rotations_preserve_order_and_only_count_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for suffix, identity in [('.10', 'old'), ('.2', 'middle'), ('', 'new')]:
                row = {'event': 'observation', 'camera_id': 'camera',
                       'created_item_ids': [identity], 'candidates': [{'label': 'Paper'}],
                       'inference_seconds': .2, 'result_age_seconds': .3}
                (path / ('observations.jsonl' + suffix)).write_text(json.dumps(row)+'\n')
            (path / 'observations.jsonl.backup').write_text('not a numbered rotation')
            report = analyze(path)
            camera = report['cameras']['camera']
            self.assertEqual(camera['created_item_ids'], ['old', 'middle', 'new'])
            self.assertEqual(camera['observations'], 3)
            self.assertEqual(camera['candidates'], 3)
            self.assertEqual(camera['inference_seconds_p95'], .2)
            self.assertEqual(report['review_counts'], {})

    def test_invalid_log_fails_without_echoing_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/'observations.jsonl').write_text('{"private": "secret')
            with self.assertRaisesRegex(ValueError, 'observations.jsonl:1') as error:
                analyze(path)
            self.assertNotIn('secret', str(error.exception))

    def test_camera_timezone_and_day_boundaries(self):
        for hour, expected in [(22, 'night'), (23, 'day'), (2, 'day'), (10, 'day'), (11, 'night')]:
            now = datetime(2026, 9, 11, hour, tzinfo=timezone.utc)
            self.assertEqual(analysis_mode('auto', 'Asia/Shanghai', now), expected)
        now = datetime(2026, 9, 11, 2, tzinfo=timezone.utc)
        self.assertEqual(analysis_mode('auto', 'UTC', now), 'night')
        self.assertEqual(analysis_mode('night', 'Asia/Shanghai', now), 'night')
