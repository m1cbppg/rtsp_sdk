import io
import json
import unittest

from rtsp_annotator.playback_source import (
    PlaybackRequest, PlaybackUrlClient, build_ctseelink_playback_payload,
    redact_url,
)


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


class PlaybackSourceTests(unittest.TestCase):
    def test_builds_documented_playback_request(self):
        payload = build_ctseelink_playback_payload(
            "44180209031322001021", "2026-09-11 08:00:00")
        self.assertEqual(payload["deviceCode"], "44180209031322001021")
        self.assertNotIn('playback',payload)
        self.assertEqual(payload["playbackTime"], "2026-09-11 08:00:00")

    def test_playback_request_limits(self):
        with self.assertRaises(ValueError):
            build_ctseelink_playback_payload("44180209031322001021", "x")

    def test_live_endpoint_is_rejected_without_a_request(self):
        def opener(*args, **kwargs):
            self.fail('must reject before opening the live URL endpoint')
        client=PlaybackUrlClient('https://example.test/ctseelink/devices/rtsp',opener=opener)
        with self.assertRaisesRegex(ValueError,'live-only'):
            client.resolve_recording(PlaybackRequest('44180209031322001021','s','e',{}))

    def test_recording_requires_matching_interval_and_preserves_offset(self):
        body=build_ctseelink_playback_payload('44180209031322001021','2026-09-08T04:00:10+00:00')
        self.assertEqual(body['playbackTime'],'2026-09-08 12:00:10')
        data={'rtspUrl':'rtsp://example.test/recording?token=secret',
              'recordStartTime':'2026-09-08 12:00:00','recordEndTime':'2026-09-08 12:05:00','offsetSeconds':10}
        client=PlaybackUrlClient('https://example.test/playback/rtsp/by-time',
                                 opener=lambda *a,**kw:_Response({'code':200,'data':data}))
        request=PlaybackRequest('44180209031322001021','s','e',body)
        recording=client.resolve_recording(request)
        self.assertEqual(recording.offset_seconds,10)
        self.assertFalse(recording.metadata()['content_time_verified'])
        self.assertNotIn('secret',repr(recording))
        data['offsetSeconds']=0
        with self.assertRaisesRegex(RuntimeError,'does not cover'):
            client.resolve_recording(request)
        data['offsetSeconds']=10
        data['recordStartTime']='2026-09-11 12:00:00'
        with self.assertRaisesRegex(RuntimeError,'does not cover'):
            client.resolve_recording(request)

    def test_live_response_cannot_masquerade_as_recording(self):
        client=PlaybackUrlClient('https://example.test/playback/rtsp/by-time',
                                 opener=lambda *a,**kw:_Response({'code':200,'data':{'url':'rtsp://live/?token=secret'}}))
        with self.assertRaisesRegex(RuntimeError,'metadata') as error:
            client.resolve_recording(PlaybackRequest('44180209031322001021','s','e',{}))
        self.assertNotIn('secret',str(error.exception))

    def test_redact_url_removes_query_credentials(self):
        result = redact_url("rtsp://host:8757/live/a?token=secret&origin=private")
        self.assertEqual(result, "<redacted>")
        self.assertNotIn("secret", result)

    def test_resolves_nested_rtsp_url_without_logging_it(self):
        seen = {}

        def opener(request, timeout):
            seen["body"] = request.data
            seen["timeout"] = timeout
            return _Response({"code": 200, "data": {"rtspUrl": "rtsp://host/live/a?token=secret"}})

        client = PlaybackUrlClient("https://example.test/play", opener=opener)
        request = PlaybackRequest("44180209031322001021", "start", "end", {
            "deviceCode": "44180209031322001021", "start": "start", "end": "end"
        })
        url = client.resolve(request)
        self.assertTrue(url.startswith("rtsp://"))
        self.assertIn("secret", url)  # caller holds it only in memory
        self.assertEqual(json.loads(seen["body"])["deviceCode"], request.device_code)

    def test_missing_url_is_rejected_without_echoing_payload(self):
        def opener(request, timeout):
            return _Response({"code": 200, "message": "token=private"})

        client = PlaybackUrlClient("https://example.test/play", opener=opener)
        with self.assertRaisesRegex(RuntimeError, "did not contain"):
            client.resolve(PlaybackRequest("44180209031322001021", "s", "e", {}))

    def test_non_success_code_is_rejected(self):
        def opener(request, timeout):
            return _Response({"code": 500, "data": {"url": "rtsp://host/a?secret=x"}})

        client = PlaybackUrlClient("https://example.test/play", opener=opener)
        with self.assertRaisesRegex(RuntimeError, "non-success"):
            client.resolve(PlaybackRequest("44180209031322001021", "s", "e", {}))

    def test_request_validation(self):
        with self.assertRaises(ValueError):
            PlaybackRequest("bad", "s", "e", {}).validate()


if __name__ == "__main__":
    unittest.main()
