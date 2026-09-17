"""Bounded source-only probe; does not run a detector or create events."""
import argparse
from datetime import datetime,timezone,timedelta
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import signal
import sys
import time
from urllib.parse import urlsplit

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rtsp_annotator.ground_litter_source import LatestImageMailbox,read_camera
from rtsp_annotator import ground_litter_source


def read_source_url_file(path):
    """Validate before creating output or processes; error text excludes secrets."""
    try:
        with path.open('r', encoding='utf-8') as stream:
            raw = stream.read(16385)
        value = raw.strip()
        parsed = urlsplit(value)
        if (len(raw) > 16384 or any(c.isspace() for c in value)
                or parsed.scheme.lower() not in ('rtsp', 'rtsps')
                or not parsed.hostname or parsed.port == 0):
            raise ValueError('invalid address')
        return value
    except (OSError, UnicodeError, ValueError):
        raise ValueError('source URL file is unreadable or does not contain one valid RTSP address') from None


class SnapshotWriter:
    """Optional native JPEG samples, capped at 60; never blocks the reader lock."""
    def __init__(self, output, interval):
        self.output, self.interval = output, interval
        self.count, self.next_due, self.write_seconds = 0, 0., 0.

    def save(self, frame, record):
        if not self.interval or self.count >= 60 or record['arrival_elapsed'] < self.next_due:
            return
        import cv2
        started = time.monotonic()
        path = self.output / 'snapshots' / f"frame-{self.count:03d}.jpg"
        path.parent.mkdir(exist_ok=True)
        if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError('snapshot_write_failed')
        payload = {**record, 'image': str(path.relative_to(self.output)),
                   'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                   'size': [frame.shape[1], frame.shape[0]], 'osd_time_verified': False}
        with (self.output / 'snapshots.jsonl').open('a') as stream:
            stream.write(json.dumps(payload, allow_nan=False) + '\n')
        self.count += 1
        self.next_due = record['arrival_elapsed'] + self.interval
        self.write_seconds += time.monotonic() - started


def _percentile(values, q):
    values = sorted(float(v) for v in values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction


def atomic_json(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def summarize_samples(records, observation_seconds):
    # Arrival to mailbox is measured separately from the consumer's polling time.
    times = [r.get('arrival_elapsed', r['probe_elapsed']) for r in records]
    intervals = [b-a for a, b in zip(times, times[1:])]
    ages = [r['local_age'] for r in records]
    return {
        'count': len(records),
        'observed_mailbox_fps': len(records) / observation_seconds if observation_seconds > 0 else None,
        'wall_span_seconds': times[-1]-times[0] if len(times) > 1 else 0.0,
        'first_frame_seconds': times[0] if times else None,
        'interval_p50_seconds': _percentile(intervals, .5),
        'interval_p95_seconds': _percentile(intervals, .95),
        'interval_max_seconds': max(intervals) if intervals else None,
        'local_age_p95_seconds': _percentile(ages, .95),
        'local_age_max_seconds': max(ages) if ages else None,
        'inference_executed': False,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--local-video',type=Path)
    source.add_argument('--device-code')
    source.add_argument('--source-url-file',type=Path,
                        help='Read an RTSP URL from a protected temporary file; never include it in reports')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--run-seconds',type=float,default=20)
    p.add_argument('--width',type=int,default=2560)
    p.add_argument('--height',type=int,default=1440)
    p.add_argument('--require-pyav', action='store_true', help='Fail before opening the source if PyAV is unavailable')
    p.add_argument('--trace', action='store_true', help='Bounded raw decode timeline, maximum 120000 rows')
    p.add_argument('--snapshot-interval', type=float, default=0,
                   help='Native JPEG interval in seconds; 0 disables, at most 60 images')
    args=p.parse_args()
    if not 1<=args.run_seconds<=1800:p.error('run-seconds must be in [1,1800]')
    if not 1 <= args.width <= 4096 or not 1 <= args.height <= 4096:
        p.error('width and height must be in [1,4096]')
    if args.snapshot_interval != 0 and not 1 <= args.snapshot_interval <= 1800:
        p.error('snapshot-interval must be 0 or in [1,1800]')
    source_url = None
    if args.source_url_file:
        try:
            source_url = read_source_url_file(args.source_url_file)
        except ValueError as exc:
            p.error(str(exc))
    source_kind = ('local_video' if args.local_video else
                   'authorized_rtsp_url_file' if args.source_url_file else 'authorized_live_device')
    try:
        import av
        decoder_capability={'pyav_importable': True, 'pyav_version': getattr(av, '__version__', 'unknown')}
    except ImportError as exc:
        decoder_capability={'pyav_importable': False, 'pyav_error_type': type(exc).__name__}
    if args.require_pyav and not decoder_capability['pyav_importable']:
        p.error('PyAV is required; source was not opened')
    args.output.mkdir(parents=True,exist_ok=False)
    ctx=mp.get_context('spawn');stop=ctx.Event()
    mailbox=LatestImageMailbox(ctx,args.width,args.height,trace_capacity=2048 if args.trace else 0)
    camera={'device_code':args.device_code or 'local-probe','reference_size':[args.width,args.height], 'analysis_fps':1}
    process=ctx.Process(target=read_camera,args=(camera,mailbox,stop,
                        str(args.local_video) if args.local_video else None, source_url))
    sequence=0;records=[];started=time.monotonic()
    started_utc=datetime.now(timezone.utc).isoformat()
    snapshots = SnapshotWriter(args.output, args.snapshot_interval)
    stopped_by_signal=[]
    def request_stop(signum, _frame):
        stopped_by_signal.append(signum)
        # multiprocessing.Event uses locks also held by wait(); signal handlers
        # must not acquire them or SIGTERM can deadlock the parent mid-wait.
    previous_handlers={s:signal.signal(s, request_stop) for s in (signal.SIGTERM, signal.SIGINT)}
    trace_sequence=0;trace_count=0;trace_dropped=0;trace_limited=False
    trace_file=(args.output/'decode_timeline.jsonl').open('w') if args.trace else None
    def drain_trace():
        nonlocal trace_sequence, trace_count, trace_dropped, trace_limited
        trace_sequence, rows, dropped=mailbox.trace_after(trace_sequence)
        trace_dropped+=dropped
        for row in rows:
            if trace_count >= 120000:
                trace_limited=True
                break
            row['arrival_elapsed']=row.pop('arrival_monotonic')-started
            trace_file.write(json.dumps(row, allow_nan=False)+'\n')
            trace_count+=1
    def progress(status, elapsed):
        return {'status':status, 'started_utc':started_utc,
                'updated_utc':datetime.now(timezone.utc).isoformat(),
                'requested_seconds':args.run_seconds, 'observation_seconds':elapsed,
                'decoder':{1:'pyav',2:'opencv'}.get(mailbox.decoder.value,'unknown'),
                'source_state':mailbox.state.value, 'source_counters':mailbox.diagnostics(),
                'last_frame_local_age_seconds':time.monotonic()-mailbox.meta[1] if sequence else None,
                'publish_timing':summarize_samples(records, elapsed),
                'trace_rows':trace_count, 'trace_overwritten_rows':trace_dropped,
                'snapshot_count':snapshots.count, 'snapshot_write_seconds':snapshots.write_seconds,
                'end_to_end_freshness_verified':False, 'inference_executed':False}
    atomic_json(args.output/'run.json', {
        'started_utc':started_utc, 'requested_seconds':args.run_seconds,
        'python':sys.version, 'decoder_capability':decoder_capability,
        'require_pyav':args.require_pyav, 'analysis_fps':1,
        'frame_size':[args.width,args.height],
        'source_kind':source_kind,
        'snapshot_interval_seconds':args.snapshot_interval, 'snapshot_limit':60,
        'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (Path(__file__),Path(ground_litter_source.__file__))},
        'local_video_sha256':hashlib.sha256(args.local_video.read_bytes()).hexdigest() if args.local_video else None,
        'trace_capacity':mailbox.trace_capacity, 'trace_limit_rows':120000,
        'notifications':0})
    next_progress=0
    process.start()
    try:
        while time.monotonic()-started<args.run_seconds and process.is_alive() and not stopped_by_signal:
            packet=mailbox.latest_after(sequence)
            if packet is not None:
                meta,frame=packet;sequence=meta[0]
                records.append({'sequence':sequence,'probe_elapsed':time.monotonic()-started,
                                'arrival_elapsed':meta[1]-started,
                                'local_age':time.monotonic()-meta[1],
                                'generation':meta[2],'source_elapsed':meta[3]})
                snapshots.save(frame, {**records[-1], 'decoded_at_utc':(
                    datetime.fromisoformat(started_utc) + timedelta(seconds=meta[1]-started)).isoformat()})
            drain_trace()
            elapsed=time.monotonic()-started
            if elapsed>=next_progress:
                atomic_json(args.output/'progress.json',progress('running',elapsed))
                if trace_file:trace_file.flush()
                next_progress=elapsed+5
            stop.wait(.05)
    finally:
        observation_seconds=time.monotonic()-started
        stop.set();process.join(14)
        if process.is_alive():process.terminate();process.join(3)
        if process.is_alive():process.kill();process.join(3)
        drain_trace()
        if trace_file:trace_file.close()
        for s, handler in previous_handlers.items():signal.signal(s, handler)
    status=('interrupted' if stopped_by_signal else
            'completed' if observation_seconds>=args.run_seconds and process.exitcode==0 else 'reader_ended')
    atomic_json(args.output/'progress.json',progress(status,observation_seconds))
    report={'finished_utc':datetime.now(timezone.utc).isoformat(),
            'source_kind':source_kind,
            'elapsed_seconds':time.monotonic()-started,'decoder':{1:'pyav',2:'opencv'}.get(mailbox.decoder.value,'unknown'),
            'status':status,'started_utc':started_utc,'observation_seconds':observation_seconds,
            'stopped_by_signal':stopped_by_signal,
            'decoder_capability': decoder_capability,
            'source_counters':mailbox.diagnostics(),'samples':records,'reader_alive':process.is_alive(),
            'exitcode':process.exitcode,'end_to_end_freshness_verified':False,'notifications':0,
            'inference_executed':False,'visual_quality_verified':False,
            'corrupt_flag_available':mailbox.decoder.value==1,
            'source_pts_available':mailbox.decoder.value==1,
            'trace_rows':trace_count,'trace_overwritten_rows':trace_dropped,'trace_limit_reached':trace_limited,
            'snapshot_count':snapshots.count,'snapshot_write_seconds':snapshots.write_seconds,
            'publish_timing':summarize_samples(records,observation_seconds)}
    atomic_json(args.output/'summary.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='samples'}))


if __name__=='__main__':main()
