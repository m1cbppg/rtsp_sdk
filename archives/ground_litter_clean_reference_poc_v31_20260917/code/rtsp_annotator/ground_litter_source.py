"""Isolated native-resolution readers for the ground-litter pilot."""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import time
import urllib.request
from urllib.parse import parse_qs, urlsplit

import numpy as np

ENDPOINT = ('https://qyzcapi.dgjx0769.com/ims-mainte-pc/p-api/v1/monitor/'
            'play/ctseelink/devices/rtsp')

SOURCE_COUNTERS=('decoded_frames','corrupt_frames','missing_pts_frames','nonmonotonic_pts_frames',
                 'connections','published_frames','mailbox_lock_skips','decode_read_seconds',
                 'convert_seconds','pts_observed_frames','trace_lock_skips')

TRACE_FIELDS = ('sequence', 'arrival_monotonic', 'generation', 'source_pts',
                'read_seconds', 'keyframe', 'corrupt', 'decoder')


def inspect_rtsp_url_expiry(url: str, now: float | None = None) -> dict:
    """Inspect timestamp hints, without inferring expiry or source availability.

    No provider expiry contract is configured here. Even an ``expires`` field
    is only a hint; ``TimeStamp`` may be issuance/signing time. Keep the legacy
    function name for callers, but never reject a URL using this diagnostic.
    """
    result = {'host': None, 'port': None, 'status': 'unknown',
              'timestamp_parameter': None, 'timestamp_epoch': None, 'age_seconds': None,
              'expiry_verified': False}
    try:
        parsed = urlsplit(url)
        host, port = parsed.hostname, parsed.port
        if parsed.scheme.lower() not in ('rtsp', 'rtsps') or not host:
            raise ValueError('invalid RTSP address')
    except (TypeError, ValueError):
        return {**result, 'status': 'invalid_url'}
    result.update(host=host, port=port)
    params = parse_qs(parsed.query)
    key = next((key for key in ('TimeStamp', 'timestamp', 'expires', 'Expires') if params.get(key)), None)
    if key is None:
        return result
    result['timestamp_parameter'] = key
    try:
        if len(params[key]) != 1:
            raise ValueError('ambiguous timestamp')
        epoch = float(params[key][0])
        if epoch > 10_000_000_000:
            epoch /= 1000.0
        current = time.time() if now is None else float(now)
        age = current - epoch
        if not all(math.isfinite(value) for value in (epoch, current, age)) or epoch < 0:
            raise ValueError('invalid timestamp')
    except (TypeError, ValueError):
        return {**result, 'status': 'unparseable'}
    return {**result, 'timestamp_epoch': epoch, 'age_seconds': age}


class SourceTimeline:
    """Validate source progress; PTS is not a calibrated wall clock."""
    def __init__(self):
        self.first=None
        self.last=None

    def observe(self,pts):
        if pts is None or not math.isfinite(pts):return None,'missing_pts'
        if self.last is not None and pts<=self.last:return None,'nonmonotonic_pts'
        if self.first is None:self.first=pts
        self.last=pts
        return pts-self.first,'source_pts'


class LatestImageMailbox:
    """One shared image per camera. Inference never holds the producer lock."""
    def __init__(self, ctx, width: int, height: int, trace_capacity: int = 0):
        if not 0 <= trace_capacity <= 4096:
            raise ValueError('trace_capacity must be in [0,4096]')
        self.shape = (height, width, 3)
        self.pixels = ctx.RawArray('B', width * height * 3)
        self.meta = ctx.RawArray('d', 4)  # sequence, capture monotonic, generation, video time
        self.lock = ctx.Lock()
        self.state = ctx.Value('i', 0)  # 0 starting, 1 live, 2 unavailable, 3 EOF, 4 resolution mismatch
        self.decoder = ctx.Value('i', 0)  # 1 PyAV, 2 OpenCV
        self.counters=ctx.RawArray('d',len(SOURCE_COUNTERS))
        self.trace_capacity = trace_capacity
        self.trace_rows = ctx.RawArray('d', trace_capacity * len(TRACE_FIELDS))
        self.trace_sequence = ctx.RawValue('q', 0)
        self.trace_lock = ctx.Lock()

    def trace_decoded(self, arrival, pts, read_seconds, generation, keyframe=None, corrupt=None):
        """Optional diagnostic ring: never block decoding on a telemetry consumer."""
        if not self.trace_capacity:
            return
        if not self.trace_lock.acquire(False):
            self.count('trace_lock_skips')
            return
        try:
            sequence = self.trace_sequence.value + 1
            row = (sequence, arrival, generation, pts, read_seconds, keyframe, corrupt, self.decoder.value)
            start = ((sequence - 1) % self.trace_capacity) * len(TRACE_FIELDS)
            self.trace_rows[start:start + len(TRACE_FIELDS)] = tuple(
                float('nan') if value is None else float(value) for value in row)
            self.trace_sequence.value = sequence
        finally:
            self.trace_lock.release()

    def trace_after(self, sequence):
        """Copy the bounded ring; gaps explicitly report overwritten telemetry."""
        if not self.trace_capacity or not self.trace_lock.acquire(False):
            return sequence, [], 0
        try:
            last = self.trace_sequence.value
            first = max(sequence + 1, last - self.trace_capacity + 1)
            rows = []
            for index in range(first, last + 1):
                start = ((index - 1) % self.trace_capacity) * len(TRACE_FIELDS)
                values = self.trace_rows[start:start + len(TRACE_FIELDS)]
                rows.append(dict(zip(TRACE_FIELDS, (v if math.isfinite(v) else None for v in values))))
            return last, rows, max(0, first - sequence - 1)
        finally:
            self.trace_lock.release()

    def count(self,name,value=1):
        # Single reader writer; approximate snapshots are telemetry only.
        self.counters[SOURCE_COUNTERS.index(name)]+=value

    def diagnostics(self):
        return {name:self.counters[i] for i,name in enumerate(SOURCE_COUNTERS)}

    def publish(self, image, timestamp, generation=0, video_time=0) -> bool:
        if image.shape != self.shape or image.dtype != np.uint8:
            raise ValueError('Unexpected native image format')
        if not self.lock.acquire(False):
            self.count('mailbox_lock_skips')
            return False
        try:
            np.copyto(np.frombuffer(self.pixels, np.uint8).reshape(self.shape), image)
            self.meta[0] += 1
            self.meta[1:] = (timestamp, generation, video_time)
            self.state.value = 1
            self.count('published_frames')
            return True
        finally:
            self.lock.release()

    def latest_after(self, sequence):
        if not self.lock.acquire(False):
            return None
        try:
            if self.meta[0] <= sequence:
                return None
            return (tuple(self.meta), np.frombuffer(self.pixels, np.uint8).reshape(self.shape).copy())
        finally:
            self.lock.release()


def resolve_rtsp(device_code: str) -> str:
    request = urllib.request.Request(ENDPOINT,
        data=json.dumps({'deviceCode': device_code}).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=12) as response:
        payload = json.load(response)
    url = (payload.get('data') or {}).get('url', '')
    if payload.get('code') != 200 or not url.startswith(('rtsp://', 'rtsps://')):
        raise RuntimeError('Stream URL unavailable')
    return url


def _read_camera_opencv(camera: dict, mailbox: LatestImageMailbox, stop,
                        source: str, local_video: bool, generation: int) -> bool:
    """Decode one source with OpenCV when PyAV is unavailable in a lean image."""
    import cv2

    # Open/read timeout must be supplied at open time; set() after open has no effect.
    os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS', 'rtsp_transport;tcp')
    params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 12000,
              cv2.CAP_PROP_READ_TIMEOUT_MSEC, 12000]
    if hasattr(cv2, 'CAP_PROP_N_THREADS'):
        params.extend([cv2.CAP_PROP_N_THREADS, 1])
    capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG, params)
    try:
        if not capture.isOpened():
            return False
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
        if not 1 <= fps <= 120:
            fps = 25.0
        count = 0
        first_clock = time.monotonic()
        next_due = 0.0
        while not stop.is_set():
            read_started=time.monotonic()
            ok, image = capture.read()
            arrived = time.monotonic()
            mailbox.count('decode_read_seconds',arrived-read_started)
            if not ok or image is None:
                return local_video
            mailbox.count('decoded_frames')
            mailbox.trace_decoded(arrived, None, arrived-read_started, generation)
            mailbox.count('missing_pts_frames')  # count/FPS is not source PTS
            elapsed = count / fps
            count += 1
            clock = elapsed if local_video else time.monotonic() - first_clock
            if clock < next_due:
                continue
            next_due = clock + 1 / camera['analysis_fps']
            if local_video and stop.wait(max(0, first_clock + elapsed - time.monotonic())):
                return True
            if image.shape != mailbox.shape:
                mailbox.state.value = 4
                return False
            mailbox.publish(image, time.monotonic() if local_video else arrived, generation, elapsed)
        return True
    finally:
        capture.release()


def read_camera(camera: dict, mailbox: LatestImageMailbox, stop,
                local_video: str | None = None, source_url: str | None = None):
    generation = 0
    retry = .5
    while not stop.is_set():
        generation += 1
        start = time.monotonic()
        next_due = 0.0
        try:
            mailbox.count('connections')
            source = (str(Path(local_video).resolve()) if local_video
                      else source_url or resolve_rtsp(camera['device_code']))
            try:
                import av
            except ImportError:
                mailbox.decoder.value = 2
                if _read_camera_opencv(camera, mailbox, stop, source, local_video is not None, generation):
                    if local_video:
                        mailbox.state.value = 3
                        return
                raise RuntimeError('video_decoder_unavailable')
            mailbox.decoder.value = 1
            av.logging.set_level(av.logging.PANIC)  # library errors can contain temporary credentials
            with av.open(source, options={} if local_video else {'rtsp_transport':'tcp'}, timeout=(12,12)) as container:
                video = container.streams.video[0]
                video.thread_type = 'SLICE'
                video.codec_context.thread_count = 1
                timeline=SourceTimeline();last_good=time.monotonic()
                decoded_frames=iter(container.decode(video=0))
                while not stop.is_set():
                    read_started=time.monotonic()
                    try:
                        decoded=next(decoded_frames)
                    except StopIteration:
                        break
                    finally:
                        arrived = time.monotonic()
                        mailbox.count('decode_read_seconds',arrived-read_started)
                    mailbox.count('decoded_frames')
                    pts=float(decoded.time) if decoded.time is not None else None
                    mailbox.trace_decoded(arrived, pts, arrived-read_started, generation,
                                          decoded.key_frame, decoded.is_corrupt)
                    if stop.is_set():
                        break
                    if decoded.is_corrupt:
                        mailbox.count('corrupt_frames')
                        mailbox.state.value = 2
                        if time.monotonic()-last_good>12:raise RuntimeError('corrupt_source_timeout')
                        continue
                    elapsed,kind=timeline.observe(pts)
                    if elapsed is None:
                        mailbox.count(kind+'_frames')
                        mailbox.state.value=2
                        if time.monotonic()-last_good>12:raise RuntimeError('source_timestamp_timeout')
                        continue
                    mailbox.count('pts_observed_frames')
                    last_good=time.monotonic()
                    if elapsed==0:
                        start = time.monotonic()
                    clock = elapsed if local_video else time.monotonic()
                    if clock < next_due:
                        continue
                    next_due = clock + 1 / camera['analysis_fps']
                    if local_video and stop.wait(max(0, start + elapsed - time.monotonic())):
                        break
                    if (decoded.height, decoded.width, 3) != mailbox.shape:
                        mailbox.state.value = 4
                        stop.wait(.2)
                        continue
                    convert_started=time.monotonic()
                    image=decoded.to_ndarray(format='bgr24')
                    mailbox.count('convert_seconds',time.monotonic()-convert_started)
                    mailbox.publish(image,time.monotonic() if local_video else arrived,generation,elapsed)
                    retry = .5
            if local_video:
                mailbox.state.value = 3
                return
        except Exception:
            # Never persist or print the exception body / temporary URL.
            mailbox.state.value = 2
        if stop.wait(retry):
            break
        retry = min(10, retry * 2)
