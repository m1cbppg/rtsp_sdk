import unittest
import json
import threading
from types import SimpleNamespace
from unittest.mock import patch, Mock

from rtsp_annotator.ground_litter_source import read_camera, SourceTimeline, inspect_rtsp_url_expiry


class GroundLitterSourceTests(unittest.TestCase):
    def test_explicit_url_never_falls_back_to_live_resolver(self):
        stop = threading.Event()
        mailbox = SimpleNamespace(decoder=SimpleNamespace(value=0), state=SimpleNamespace(value=0), count=Mock())
        def decode(*args):
            stop.set()
            return False
        with patch.dict('sys.modules', {'av': None}), patch(
                'rtsp_annotator.ground_litter_source._read_camera_opencv', side_effect=decode) as reader, patch(
                'rtsp_annotator.ground_litter_source.resolve_rtsp') as resolver:
            read_camera({'device_code': 'camera'}, mailbox, stop, source_url='rtsp://example.test/replay')
        resolver.assert_not_called()
        self.assertEqual(reader.call_args.args[3], 'rtsp://example.test/replay')

    def test_source_pts_does_not_fabricate_progress_from_repeated_frames(self):
        clock=SourceTimeline()
        self.assertEqual(clock.observe(100),(0,'source_pts'))
        self.assertEqual(clock.observe(100.5),(.5,'source_pts'))
        for pts in [100.5,99]:self.assertEqual(clock.observe(pts),(None,'nonmonotonic_pts'))
        for pts in [None,float('nan'),float('inf')]:self.assertEqual(clock.observe(pts),(None,'missing_pts'))
        self.assertEqual(clock.observe(101),(1,'source_pts'))

    def test_rtsp_url_expiry_diagnostic_is_sanitized(self):
        url='rtsp://user:secret@example.test:8554/live?a=secret&TimeStamp=1700000000000'
        result=inspect_rtsp_url_expiry(url, now=1700000010)
        self.assertEqual(result['status'],'unknown')
        self.assertFalse(result['expiry_verified'])
        self.assertEqual(result['age_seconds'],10)
        self.assertNotIn('secret', repr(result))
        self.assertEqual(inspect_rtsp_url_expiry('rtsp://camera/live', now=1)['status'],'unknown')

    def test_timestamp_hints_never_prove_expiry_or_availability(self):
        for key in ['TimeStamp','timestamp','expires','Expires']:
            for epoch in [1700000000,1700000000000,1700000020]:
                result=inspect_rtsp_url_expiry(f'rtsp://camera/live?{key}={epoch}',now=1700000010)
                self.assertEqual(result['status'],'unknown')
                self.assertFalse(result['expiry_verified'])

    def test_invalid_timestamp_and_url_diagnostics_stay_json_safe(self):
        for value in ['NaN','inf','-inf','-1','invalid','1&timestamp=2']:
            result=inspect_rtsp_url_expiry('rtsp://camera/live?timestamp='+value,now=1)
            self.assertEqual(result['status'],'unparseable')
            self.assertIsNone(result['age_seconds'])
            json.dumps(result,allow_nan=False)
        for url in ['rtsp://camera:secret/live','rtsp://[invalid','rtsp:///live']:
            result=inspect_rtsp_url_expiry(url)
            self.assertEqual(result['status'],'invalid_url')
            self.assertNotIn('secret',repr(result))
        self.assertEqual(inspect_rtsp_url_expiry('rtsp://camera?expires=1',now=float('nan'))['status'],'unparseable')


if __name__ == "__main__":
    unittest.main()
