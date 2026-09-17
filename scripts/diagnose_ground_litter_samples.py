"""Replay frozen native images with stage traces; never invent temporal evidence."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
os.environ.setdefault('YOLO_CONFIG_DIR', '/tmp/litter_pilot')

import cv2
import numpy as np

from rtsp_annotator.ground_litter_alignment import ViewAlignment
from rtsp_annotator.ground_litter_diagnostics import runtime_fingerprint
from rtsp_annotator.ground_litter_geometry import prepare
from rtsp_annotator.ground_litter_runtime import LitterModel


def intersects(left, right, minimum_iou=.3):
    a,b,c,d=left; x,y,r,s=right
    intersection=max(0,min(c,r)-max(a,x))*max(0,min(d,s)-max(b,y))
    union=(c-a)*(d-b)+(r-x)*(s-y)-intersection
    return union>0 and intersection/union>=minimum_iou


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profiles', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--sample-period',type=float,default=0,help='Minimum wall interval between offline probes')
    parser.add_argument('--actor-imgsz', type=int, default=1280)
    parser.add_argument('--mode', choices=['day','night'], default='night')
    parser.add_argument('--confidence', type=float)
    parser.add_argument('--tile-layout', choices=['minimal','full_grid'], default='minimal')
    args=parser.parse_args()
    if args.confidence is not None and not 0 < args.confidence <= 1:
        parser.error('confidence must be in (0,1]')
    if not 0 <= args.sample_period <= 30:
        parser.error('sample-period must be in [0,30]')
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    torch.set_num_threads(4)
    if args.device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(.25,device=args.device)
    cv2.setNumThreads(2)
    payload=json.loads(args.profiles.read_text())
    camera=payload['cameras'][0]
    if args.confidence is not None:
        camera[args.mode]['minimum_confidence']=args.confidence
    reference=cv2.imread(camera['reference_image'])
    if reference is None:
        raise ValueError('Missing reference image')
    masks,tiles,area=prepare(camera, reference, 640, payload['model']['tile_overlap'])
    if args.tile_layout=='full_grid':
        h,w=reference.shape[:2];mask=np.maximum.reduce(list(masks.values()))
        xs=sorted(set([*range(0,max(1,w-639),512),w-640]))
        ys=sorted(set([*range(0,max(1,h-639),512),h-640]))
        tiles=[(x,y,x+640,y+640) for x in xs for y in ys if mask[y:y+640,x:x+640].any()]
        coverage=np.zeros_like(mask)
        for x,y,r,b in tiles: coverage[y:b,x:r]=1
        assert not np.any(mask & ~coverage)
    model=LitterModel(payload['model']['path'],device=args.device,
                      actor_imgsz=args.actor_imgsz,diagnostic_candidates=True)
    alignment=ViewAlignment(reference,camera)
    info={'kind':'frozen_image_diagnostics_not_temporal_or_accuracy',
          'device':args.device, 'actor_imgsz':args.actor_imgsz,'sample_period':args.sample_period,
          'confidence':camera[args.mode]['minimum_confidence'], 'tiles':tiles,
          'tile_layout':args.tile_layout, 'roi_area_px':area,
          'profile_sha256':hashlib.sha256(args.profiles.read_bytes()).hexdigest(),
          'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
          'runtime':runtime_fingerprint([payload['model']['path'],'models/yolo26s.pt'])}
    (args.output/'run.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
    rows=[]
    with (args.output/'results.jsonl').open('w') as stream:
        for sample in map(json.loads,args.manifest.read_text().splitlines()):
            sample_started=time.monotonic()
            path=Path(sample['path'])
            if hashlib.sha256(path.read_bytes()).hexdigest()!=sample['sha256']:
                raise ValueError('Sample hash changed: '+sample['sample_id'])
            frame=cv2.imread(str(path));started=time.perf_counter()
            alignment_status='accepted'
            try: alignment.check(frame)
            except ValueError as exc: alignment_status=str(exc)
            alignment_seconds=time.perf_counter()-started
            # Intentional independent model probe even on rejected images;
            # no inventory, confirmation, clean event or live result emitted.
            proposals,actors=model.analyze(frame,camera,tiles,masks,args.mode)
            stats=model.last_stats
            targets=[]
            for target in sample['targets']:
                box=target['box'];x,y,r,b=box
                stages={stage:[d for d in detections if intersects(box,d['box'])]
                        for stage,detections in [('raw',stats.get('raw_candidates',[])),
                        ('nms_suppressed',stats.get('nms_suppressed',[])),
                        ('rejected',stats['rejected_candidates']),('retained',proposals)]}
                targets.append({**target,'inside_roi':bool(any(m[(y+b)//2,(x+r)//2] for m in masks.values())),
                                'covering_tiles':[i for i,(a,c,d,e) in enumerate(tiles) if a<=x and c<=y and d>=r and e>=b],
                                'stages':stages})
            row={'sample_id':sample['sample_id'],'sample_path':sample['path'],
                 'alignment_status':alignment_status,'alignment_seconds':alignment_seconds,
                 'alignment':alignment.diagnostics,'candidates':proposals,'actors':actors,
                 'diagnostics':stats,'targets':targets}
            stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();rows.append(row)
            print(json.dumps({'sample':sample['sample_id'],'alignment':alignment_status,
                'raw':stats['raw_before_nms'],'retained':len(proposals),
                'targets':{t['target_id']:{k:len(v) for k,v in t['stages'].items()} for t in targets}}),flush=True)
            time.sleep(max(0,args.sample_period-(time.monotonic()-sample_started)))
    timings={key:{'p50':float(np.percentile([r['diagnostics']['stage_seconds'].get(key,0) for r in rows],50)),
                  'p95':float(np.percentile([r['diagnostics']['stage_seconds'].get(key,0) for r in rows],95))}
             for key in sorted({k for r in rows for k in r['diagnostics']['stage_seconds']})}
    summary={'samples':len(rows),'alignment':dict(Counter(r['alignment_status'] for r in rows)),
             'stage_seconds':timings,'candidate_count':sum(len(r['candidates']) for r in rows),
             'accuracy':None,'temporal_validation':False,'notifications':0}
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
