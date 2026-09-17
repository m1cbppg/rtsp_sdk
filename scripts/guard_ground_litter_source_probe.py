"""Run a bounded source probe in its own Docker container, protecting existing streams.

Run on the authorized test host. The release needs src/, scripts/ and a manifest.json
mapping their relative paths to SHA-256. The image must already contain PyAV. Only the
uniquely named test container is started/stopped; production access is read-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

PRODUCTION = ('rtsp-yolo-api', 'camera-control', 'rtsp-web-gateway', 'rtsp-mediamtx')
METRICS_CODE = '''import json,urllib.request
c=json.load(open('/app/config/api.json'))
r=urllib.request.Request('http://127.0.0.1:8080/v1/streams',headers={'X-API-Key':c['api']['key']})
d=json.load(urllib.request.urlopen(r,timeout=5))
rows=d if isinstance(d,list) else d.get('streams')
if not isinstance(rows,list):raise RuntimeError('unknown_stream_response')
keys=('publish_fps','unique_publish_fps','duplicate_publish_fps','pipeline_healthy','metrics_updated_at_unix')
print(json.dumps([{'id':x.get('stream_id',x.get('id')),'status':x.get('status'),
 'metrics':{k:(x.get('metrics') or {}).get(k) for k in keys}} for x in rows]))
'''


def capture(argv, timeout=10):
    return subprocess.check_output(argv, text=True, stderr=subprocess.DEVNULL, timeout=timeout)


def snapshot():
    rows=json.loads(capture(['docker','inspect',*PRODUCTION]))
    return {'utc':datetime.now(timezone.utc).isoformat(), 'checked_unix':time.time(),
            'containers':[{'id':r['Id'],'name':r['Name'],'started':r['State']['StartedAt'],
                           'restarts':r['RestartCount'],'running':r['State']['Running']} for r in rows],
            'streams':json.loads(capture(['docker','exec','rtsp-yolo-api','python3','-c',METRICS_CODE]))}


def protection_failure(baseline, current, minimum_fps=23):
    if baseline['containers']!=current['containers']:
        return 'production_containers_changed'
    if not all(r['running'] for r in current['containers']):
        return 'production_container_not_running'
    if {r['id'] for r in baseline['streams']}!={r['id'] for r in current['streams']}:
        return 'production_streams_changed'
    for row in current['streams']:
        m=row['metrics']
        updated=m.get('metrics_updated_at_unix')
        if (not isinstance(updated,(int,float)) or not math.isfinite(updated)
                or not -5 <= current['checked_unix']-updated <= 30):
            return 'production_metrics_stale_or_unknown'
        if row['status']!='running' or m.get('pipeline_healthy') is not True:
            return 'production_stream_unhealthy'
        for key in ('publish_fps','unique_publish_fps'):
            value=m.get(key)
            if not isinstance(value,(int,float)) or not math.isfinite(value) or value<minimum_fps:
                return 'production_fps_below_limit'
        if m.get('duplicate_publish_fps')!=0:
            return 'production_duplicate_or_unknown'
    return None


def validate_release(release):
    manifest=json.loads((release/'manifest.json').read_text())
    required={'src/rtsp_annotator/ground_litter_source.py','src/rtsp_annotator/__init__.py',
              'scripts/probe_ground_litter_source.py'}
    if not required <= manifest.keys():
        raise ValueError('incomplete_release_manifest')
    for relative, expected in manifest.items():
        path=(release/relative).resolve()
        if not path.is_relative_to(release) or not path.is_file():
            raise ValueError('invalid_release_path')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('release_hash_mismatch')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--image',required=True)
    p.add_argument('--device-code',required=True)
    p.add_argument('--run-seconds',type=int,choices=(60,600,1800),required=True)
    args=p.parse_args()
    release=args.release.resolve();output=args.output.resolve()
    manifest=validate_release(release)
    output.mkdir(parents=True,exist_ok=False)
    reason=None;returncode=None;baseline=None;final=None;process=None
    name='ground-litter-source-'+uuid.uuid4().hex[:12]
    signals=[]
    def stop_requested(signum,_frame):signals.append(signum)
    for s in (signal.SIGINT,signal.SIGTERM):signal.signal(s,stop_requested)
    started=time.monotonic()
    def save_status(status):
        payload={'status':status,'reason':reason,'updated_utc':datetime.now(timezone.utc).isoformat(),
                 'container':name,'returncode':returncode,'elapsed_seconds':time.monotonic()-started,
                 'production_baseline':baseline,'production_final':final,'manifest':manifest,
                 'image_id':image_id,'requested_seconds':args.run_seconds,'notifications':0}
        temp=output/'guardian.tmp'
        temp.write_text(json.dumps(payload,indent=2)+'\n');temp.replace(output/'guardian.json')
    image_id=None
    with (output/'container.log').open('w') as log, (output/'coexistence.jsonl').open('w') as coexistence:
        try:
            baseline=snapshot();final=baseline
            reason=protection_failure(baseline,baseline)
            if reason:return 2
            image_id=json.loads(capture(['docker','image','inspect',args.image]))[0]['Id']
            command=['docker','run','--rm','--name',name,'--network','host',
                     '--user',f'{os.getuid()}:{os.getgid()}',
                     '--cpus','1.5','--memory','1g','--pids-limit','128','--read-only',
                     '--tmpfs','/tmp:rw,size=64m','--cap-drop','ALL',
                     '--security-opt','no-new-privileges','--entrypoint','python3',
                     '-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONUNBUFFERED=1',
                     '-e','PYTHONPATH=/probe/src','-v',f'{release}:/probe:ro',
                     '-v',f'{output}:/output',image_id,
                     '/probe/scripts/probe_ground_litter_source.py','--require-pyav','--trace',
                     '--device-code',args.device_code,'--run-seconds',str(args.run_seconds),
                     '--output','/output/result']
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            save_status('running')
            next_check=0
            while process.poll() is None:
                if signals:reason='guardian_signal';break
                if time.monotonic()-started>args.run_seconds+45:reason='probe_deadline';break
                if time.monotonic()>=next_check:
                    final=snapshot()
                    coexistence.write(json.dumps(final)+'\n');coexistence.flush()
                    reason=protection_failure(baseline,final)
                    save_status('running')
                    if reason:break
                    next_check=time.monotonic()+5
                time.sleep(.2)
            returncode=process.poll()
        except Exception as exc:
            reason='guard_error_'+type(exc).__name__  # Do not log credentials or URL-bearing exception bodies.
        finally:
            if process is not None and process.poll() is None:
                try:
                    subprocess.run(['docker','stop','-t','25',name],stdout=log,stderr=log,timeout=35)
                    process.wait(timeout=5)
                except (subprocess.TimeoutExpired,OSError):
                    reason=reason or 'test_container_stop_failed'
                    subprocess.run(['docker','kill',name],stdout=log,stderr=log,timeout=10)
                    process.wait(timeout=5)
                returncode=process.poll()
            try:
                final=snapshot()
                if baseline is not None:reason=reason or protection_failure(baseline,final)
            except Exception as exc:reason=reason or 'postflight_error_'+type(exc).__name__
            result=output/'result/summary.json'
            if reason is None:
                if returncode!=0 or not result.is_file():reason='probe_failed_or_summary_missing'
                elif json.loads(result.read_text()).get('status')!='completed':reason='probe_incomplete'
            save_status('stopped' if reason else 'completed')
            print(json.dumps({'status':'stopped' if reason else 'completed','reason':reason,'output':str(output)}))
    return 0 if reason is None else 2


if __name__=='__main__':raise SystemExit(main())
