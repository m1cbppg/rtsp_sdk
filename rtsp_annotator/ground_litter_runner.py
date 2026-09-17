"""Run a local/isolated multi-camera litter recognition pilot without notifications."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import html
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import time
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import cv2

from .ground_litter_geometry import polygon_points
from .ground_litter_inventory import GroundLitterInventory
from .ground_litter_runtime import CameraAnalysis,EvidenceWindow,LitterModel,pixels,validate_profiles
from .ground_litter_source import LatestImageMailbox,read_camera
from .ground_litter_diagnostics import runtime_fingerprint
from .ground_litter_journal import PilotJournal,retain_samples,output_bytes
from .ground_litter_review import ReviewCollector
from .ground_litter_batch import _review_page
from .playback_source import (PlaybackRequest, PlaybackUrlClient,
                              build_ctseelink_playback_payload)


def atomic_text(path, value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(value,encoding='utf-8')
    tmp.replace(path)


def save_image(path,frame):
    ok,content=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,88])
    if not ok:
        raise RuntimeError('Image encoding failed')
    tmp=path.with_suffix('.tmp')
    tmp.write_bytes(content.tobytes());tmp.replace(path)


def analysis_mode(mode, timezone_name, now=None):
    if mode != 'auto':
        return mode
    local = (now or datetime.now(timezone.utc)).astimezone(ZoneInfo(timezone_name))
    return 'day' if 7 <= local.hour < 19 else 'night'


def dashboard(output,cameras,stats,items,*,stopped=False):
    cards=[]
    for camera in cameras:
        code=camera['device_code'];status=stats[code]
        cards.append(f'<article><h2>{code}</h2><p>{html.escape(json.dumps(status,ensure_ascii=False))}</p>'
                     f'<img src="{code}.jpg?t={time.time():.0f}" alt="等待当前摄像头画面">'
                     '<p>疑似框需持续证据确认；商户绑定尚未确认时不发送任何通知。</p></article>')
    evidence=[]
    for item in items:
        identity=item['item_id']
        evidence.append(f'<p>{html.escape(item["camera_id"])} / {html.escape(item["region_id"])} '
                        f'/ {html.escape(item["state"])} / {identity[:12]} '
                        f'<a href="evidence/{identity}.jpg">首次确认截图</a></p>')
    label='已停止，显示最后一次结果' if stopped else '识别试运行 · 通知关闭'
    doc=('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
         '<meta http-equiv="refresh" content="3"><title>地面零散垃圾识别</title>'
         '<style>body{font:15px/1.6 system-ui;background:#f2f6f8;color:#213744;margin:24px}'
         'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(500px,1fr));gap:18px}'
         'article{background:white;padding:16px;border-radius:10px}img{width:100%}p{overflow-wrap:anywhere}</style>'
         f'<h1>{label}</h1><p>更新时间：{datetime.now().astimezone().isoformat(timespec="seconds")}</p>'
         '<p>这是低频分析画面，不是原始25FPS播放器。看不清、断流或无垃圾框均不等于已经清理。</p>'
         '<main>'+''.join(cards)+'</main><h2>垃圾记录</h2>'+''.join(evidence)+'</html>')
    atomic_text(output/'index.html',doc)
    atomic_text(output/'status.json',json.dumps({'stopped':stopped,'updated_at':time.time(),'cameras':stats},ensure_ascii=False,indent=2))
    atomic_text(output/'items.json',json.dumps(items,ensure_ascii=False,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profiles',type=Path,default=Path('config/ground_litter_profiles.example.json'))
    ap.add_argument('--devices',nargs='+',required=True)
    ap.add_argument('--output',type=Path,required=True,help='Keep same output directory to restore existing item IDs')
    ap.add_argument('--inventory',type=Path,help='Explicit persistent SQLite path shared by successive runs of the same view')
    ap.add_argument('--device',default='cpu',help='PyTorch device: cpu, cuda:0 or mps')
    ap.add_argument('--actor-imgsz',type=int,default=1280,help='Audited baseline 1280; 640 is an explicit experiment')
    ap.add_argument('--diagnostic-candidates',action='store_true',help='Bounded raw and NMS candidate traces')
    ap.add_argument('--mode',choices=['day','night','auto'],default='day',
                    help='auto uses --timezone: day 07:00-19:00; both modes still require alignment')
    ap.add_argument('--timezone',default='Asia/Shanghai',help='Camera IANA timezone, independent of server clock')
    ap.add_argument('--run-seconds',type=float,default=120,help='0 runs until interrupted')
    ap.add_argument('--allow-draft',action='store_true',help='Explicitly allow draft ROIs for shadow testing')
    ap.add_argument('--local-video',type=Path,help='Paced local replay, only with one device; no network')
    ap.add_argument('--playback-time', help='Resolve one documented replay URL for the selected device')
    ap.add_argument('--playback-endpoint',help='Reachable playback/rtsp/by-time route; devices/rtsp is live-only')
    args=ap.parse_args()
    try:
        ZoneInfo(args.timezone)
    except (ValueError, ZoneInfoNotFoundError):
        ap.error('Invalid camera timezone')
    if args.run_seconds<0 or (args.local_video and len(args.devices)!=1) or (args.playback_time and len(args.devices)!=1):
        ap.error('Invalid duration or replay camera count')
    if args.local_video and args.playback_time:
        ap.error('Choose either --local-video or --playback-time')
    if args.playback_time and not args.playback_endpoint:
        ap.error('--playback-time requires an explicit --playback-endpoint for cloud recordings')
    profiles=json.loads(args.profiles.read_text())
    cameras=[c for c in profiles['cameras'] if c['device_code'] in args.devices]
    if len(cameras)!=len(set(args.devices)):
        ap.error('Camera missing from profile')
    profiles['cameras']=cameras
    validate_profiles(profiles,allow_draft=args.allow_draft)
    if args.local_video and not args.local_video.is_file():
        ap.error('Local video missing')
    source_url = None
    recording = None
    if args.playback_time:
        payload = build_ctseelink_playback_payload(args.devices[0], args.playback_time)
        request = PlaybackRequest(args.devices[0], args.playback_time, args.playback_time, payload)
        recording = PlaybackUrlClient(args.playback_endpoint).resolve_recording(request)
        source_url = recording.url
    os.environ['YOLO_AUTOINSTALL']='false'
    os.environ.setdefault('YOLO_CONFIG_DIR','/tmp/litter_pilot')
    import torch
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    current_mode=analysis_mode(args.mode,args.timezone)
    model=LitterModel(profiles['model']['path'],device=args.device,actor_imgsz=args.actor_imgsz,
                      diagnostic_candidates=args.diagnostic_candidates)
    analyses={c['device_code']:CameraAnalysis(c,profiles['model'],current_mode) for c in cameras}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'evidence').mkdir(exist_ok=True)
    (args.output/'samples').mkdir(exist_ok=True)
    run_id=uuid4().hex
    journal=PilotJournal(args.output,run_id)
    inventory_path=args.inventory or args.output/'items.sqlite3'
    inventory_path.parent.mkdir(parents=True,exist_ok=True)
    inventory=GroundLitterInventory(inventory_path,
        clear_seconds=profiles['inventory']['clear_seconds'],
        max_observation_gap=profiles['inventory']['max_observation_gap_seconds'])
    ctx=mp.get_context('spawn');stop=ctx.Event()
    mailboxes={c['device_code']:LatestImageMailbox(ctx,*c['reference_size']) for c in cameras}
    readers=[ctx.Process(target=read_camera,args=(c,mailboxes[c['device_code']],stop,
              str(args.local_video) if args.local_video else None, source_url),
              name='litter-reader-'+c['device_code'][-4:]) for c in cameras]
    stats={c['device_code']:dict(state='starting',analyzed=0,rejected_frames=0,
             skipped_analysis_frames=0,created_items=0,cleared_items=0) for c in cameras}
    sequence={c['device_code']:0 for c in cameras}
    started=time.monotonic();last_dashboard=0;last_storage_check=0
    last_sample={c['device_code']:0 for c in cameras}
    last_capture={c['device_code']:started for c in cameras}
    source_states={c['device_code']:'starting' for c in cameras}
    reviews={c['device_code']:ReviewCollector(args.output/'review'/c['device_code'],limit=48,audit_interval=60)
             for c in cameras}
    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,lambda *_:stop.set())
    run_info={'run_id':run_id,'mode':args.mode,'timezone':args.timezone,'inference_device':args.device,
        'started_at':datetime.now(timezone.utc).isoformat(),'devices':args.devices,
        'source':('local_video' if args.local_video else
                  'playback_api' if args.playback_time else 'device_code_api'),
        'playback_time': args.playback_time,
        'recording':recording.metadata() if recording else None,
        'draft_profiles':args.allow_draft,'profile_sha256':hashlib.sha256(args.profiles.read_bytes()).hexdigest(),
        'actor_imgsz':model.actor_imgsz,'diagnostic_candidates':args.diagnostic_candidates,
        'inventory_path':str(inventory_path.resolve()),
        'runtime':runtime_fingerprint([profiles['model']['path'],'models/yolo26s.pt'])}
    atomic_text(args.output/'run.json',json.dumps(run_info,indent=2))
    journal.write('run_started', **{k:v for k,v in run_info.items() if k!='run_id'})
    try:
        for reader in readers:reader.start()
        while not stop.is_set() and (args.run_seconds==0 or time.monotonic()-started<args.run_seconds):
            next_mode=analysis_mode(args.mode,args.timezone)
            if next_mode!=current_mode:
                analyses={c['device_code']:CameraAnalysis(c,profiles['model'],next_mode) for c in cameras}
                journal.write('mode_changed',previous=current_mode,current=next_mode)
                current_mode=next_mode
            for camera in cameras:
                code=camera['device_code'];stat=stats[code];mailbox=mailboxes[code]
                stat['source_diagnostics']=mailbox.diagnostics()
                stat['source_timestamp_kind']='source_pts' if mailbox.decoder.value==1 else 'estimated_frame_count'
                packet=mailbox.latest_after(sequence[code])
                if packet is None:
                    state=mailbox.state.value
                    if state in (2,3,4) or time.monotonic()-last_capture[code]>15:
                        stat['state']={2:'source_unavailable',3:'replay_finished',4:'resolution_changed'}.get(state,'source_stalled')
                        if source_states[code]!=stat['state']:
                            analyses[code].evidence=EvidenceWindow(camera,current_mode)
                            inventory.mark_unobserved(code,camera['view_id'])
                            journal.write('source_state',camera_id=code,state=stat['state'])
                            source_states[code]=stat['state']
                    continue
                meta,frame=packet;seq,captured,generation,source_time=meta
                last_capture[code]=captured
                stat.update(last_frame_at=datetime.now(timezone.utc).isoformat(),mode=current_mode,
                            decoder={1:'pyav',2:'opencv'}.get(mailbox.decoder.value,'unknown'))
                if source_states[code]!='live':
                    journal.write('source_state',camera_id=code,state='live',generation=generation,decoder=stat['decoder'])
                    source_states[code]='live'
                if time.monotonic()-last_sample[code]>=60:
                    sample='sample-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'-'+code+'.jpg'
                    save_image(args.output/'samples'/sample,frame)
                    save_image(args.output/(code+'-raw.jpg'),frame)
                    retain_samples(args.output/'samples')
                    journal.write('sample',camera_id=code,file='samples/'+sample)
                    last_sample[code]=time.monotonic()
                stat['skipped_analysis_frames']+=max(0,int(seq-sequence[code]-1));sequence[code]=seq
                if time.monotonic()-captured>camera['frame_max_age_seconds']:
                    stat['rejected_frames']+=1;stat['state']='stale_input'
                    analyses[code].evidence.unknown(captured)
                    inventory.observe(code,camera['view_id'],captured,scene_observable=False)
                    journal.write('rejected',camera_id=code,reason=stat['state'],frame_sequence=seq,
                                  capture_monotonic=captured,generation=generation,
                                  source_time_seconds=source_time,result_age_seconds=time.monotonic()-captured,
                                  inference_seconds=0,diagnostics={})
                    continue
                start=time.monotonic()
                try:
                    proposals,actors,confirmed,result=analyses[code].consume(model,inventory,frame,captured,generation)
                except Exception as exc:
                    stat['rejected_frames']+=1
                    reason=str(exc)
                    allowed={'image_quality_unknown','view_alignment_unknown','view_changed_recalibration_required','analysis_result_stale'}
                    stat['state']=reason if reason in allowed else 'analysis_error_'+type(exc).__name__
                    if reason in {'image_quality_unknown','view_alignment_unknown','analysis_result_stale'}:
                        analyses[code].evidence.unknown(captured)
                    else:
                        analyses[code].evidence=EvidenceWindow(camera,current_mode)
                    reasons=stat.setdefault('rejection_reasons',{})
                    reasons[stat['state']]=reasons.get(stat['state'],0)+1
                    inventory.observe(code,camera['view_id'],captured,scene_observable=False)
                    journal.write('rejected',camera_id=code,reason=stat['state'],frame_sequence=seq,
                                  capture_monotonic=captured,generation=generation,
                                  source_time_seconds=source_time,result_age_seconds=time.monotonic()-captured,
                                  inference_seconds=time.monotonic()-start,
                                  diagnostics=analyses[code].last_diagnostics)
                    save_image(args.output/(code+'.jpg'),frame)
                    reviews[code].consider(frame,frame,[],captured-started,int(seq),{})
                    continue
                stat.update(state='analyzing',analyzed=stat['analyzed']+1,
                    source_time_seconds=round(source_time,2),
                    result_age_seconds=round(time.monotonic()-captured,3),
                    inference_seconds=round(time.monotonic()-start,3),
                    candidate_count=len(proposals),confirmed_visible=len(confirmed))
                stat['created_items']+=len(result.created_ids);stat['cleared_items']+=len(result.cleared_ids)
                annotation=frame.copy();h,w=frame.shape[:2]
                for zone in camera['zones']:
                    cv2.polylines(annotation,[polygon_points(zone['polygon'],w,h)],True,(0,220,220),3)
                    for exclusion in zone['exclude_zones']:
                        cv2.polylines(annotation,[polygon_points(exclusion,w,h)],True,(0,0,255),2)
                for proposal in proposals:
                    x,y,r,b=proposal['box'];cv2.rectangle(annotation,(x,y),(r,b),(0,220,220),2)
                for item,identity in zip(confirmed,result.item_ids):
                    x,y,r,b=pixels(item.rectangle,w,h)
                    cv2.rectangle(annotation,(x,y),(r,b),(0,0,255),3)
                    cv2.putText(annotation,'litter '+identity[:8],(x,max(30,y-10)),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2)
                save_image(args.output/(code+'.jpg'),annotation)
                for identity in result.created_ids:
                    save_image(args.output/'evidence'/(identity+'.jpg'),annotation)
                entry={'camera_id':code,'capture_monotonic':captured,'generation':generation,
                       'frame_sequence':seq,'mode':current_mode,
                       'source_time_seconds':source_time,'inference_seconds':stat['inference_seconds'],
                       'result_age_seconds':stat['result_age_seconds'],
                       'candidates':proposals,'visible_item_ids':result.item_ids,
                       'created_item_ids':result.created_ids,'cleared_item_ids':result.cleared_ids,
                       'diagnostics':analyses[code].last_diagnostics}
                journal.write('observation', **entry)
                reviews[code].consider(frame,annotation,proposals,captured-started,int(seq),analyses[code].last_diagnostics)
                if stop.is_set():break
            now=time.monotonic()
            if now-last_storage_check>=60:
                for collector in reviews.values():
                    collector.save()
                    _review_page(collector.output,[],collector.entries())
                size=output_bytes(args.output)
                journal.write('storage',bytes=size,limit_bytes=2*1024**3)
                last_storage_check=now
                if size>=2*1024**3:
                    journal.write('storage_limit_reached')
                    stop.set()
            if now-last_dashboard>=2:
                for stat in stats.values():stat['effective_analysis_fps']=round(stat['analyzed']/max(1,now-started),3)
                items=[row for c in cameras for row in inventory.active(c['device_code'],c['view_id'])]
                dashboard(args.output,cameras,stats,items);last_dashboard=now
            if args.local_video and all(m.state.value==3 for m in mailboxes.values()):break
            stop.wait(.05)
    finally:
        stop.set()
        for reader in readers:
            if reader.pid is None:continue
            reader.join(2)
            if reader.is_alive():reader.terminate();reader.join(2)
            if reader.is_alive():reader.kill();reader.join()
        elapsed=time.monotonic()-started
        for stat in stats.values():stat['effective_analysis_fps']=round(stat['analyzed']/max(1,elapsed),3)
        items=[row for c in cameras for row in inventory.active(c['device_code'],c['view_id'])]
        dashboard(args.output,cameras,stats,items,stopped=True)
        for collector in reviews.values():
            collector.save()
            _review_page(collector.output,[],collector.entries())
        atomic_text(args.output/'summary.json',json.dumps({'elapsed_seconds':elapsed,'cameras':stats,
            'active_items':len(items),'reader_processes_alive':sum(p.is_alive() for p in readers if p.pid),
            'notifications_sent':0},ensure_ascii=False,indent=2))
        journal.write('run_stopped',elapsed_seconds=elapsed,cameras=stats)
        journal.close()
        inventory.close()
        print(json.dumps({'output':str(args.output),'elapsed_seconds':round(elapsed,1),'cameras':stats},ensure_ascii=False))


if __name__=='__main__':
    main()
