"""Render per-camera draft ROIs and optionally run local candidate inference.

This is a calibration tool, not the stream API or a precision benchmark.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np


from rtsp_annotator.ground_litter_geometry import polygon_points, prepare, box_overlap_fraction

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profiles', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--infer', action='store_true')
    args = ap.parse_args()
    profiles = json.loads(args.profiles.read_text())
    if profiles['schema_version'] != 2:
        ap.error('Expected design profile schema 2')
    if len({c['device_code'] for c in profiles['cameras']}) != len(profiles['cameras']):
        ap.error('Duplicate camera ID')
    model = actor_model = None
    if args.infer:
        os.environ.setdefault('YOLO_CONFIG_DIR', '/tmp/litter_eval/ultralytics')
        os.environ['YOLO_AUTOINSTALL'] = 'false'
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(4)
        weights = Path(profiles['model']['path'])
        actor_weights = Path('models/yolo26s.pt')
        if not weights.is_file() or not actor_weights.is_file():
            ap.error('Local weights required; no automatic downloads')
        model, actor_model = YOLO(str(weights)), YOLO(str(actor_weights))
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'status':'draft_calibration_single_frame_not_accuracy',
              'profiles_sha256':hashlib.sha256(args.profiles.read_bytes()).hexdigest(),
              'cameras':[]}
    for camera in profiles['cameras']:
        frame = cv2.imread(camera['reference_image'])
        if frame is None:
            raise ValueError('Calibration image unavailable')
        h,w = frame.shape[:2]
        masks,tiles,area = prepare(camera, frame, profiles['model']['tile_size_px'], profiles['model']['tile_overlap'])
        annotation = frame.copy()
        colors = [(0,230,230),(230,230,0),(230,0,230)]
        for index,zone in enumerate(camera['zones']):
            pts=polygon_points(zone['polygon'],w,h)
            color=colors[index%len(colors)]
            cv2.polylines(annotation,[pts],True,color,5)
            x,y=pts[0]
            cv2.putText(annotation,zone['region_id'],(x,max(35,y-15)),cv2.FONT_HERSHEY_SIMPLEX,.8,color,2)
            for exclusion in zone['exclude_zones']:
                cv2.polylines(annotation,[polygon_points(exclusion,w,h)],True,(0,0,255),5)
        for exclusion in camera.get('overlay_exclude_zones',[]):
            cv2.polylines(annotation,[polygon_points(exclusion,w,h)],True,(0,0,255),3)
        detections=[]; actors=[]; elapsed=None; raw_count=0
        if model is not None:
            start=time.monotonic()
            actor_result=actor_model.predict(frame,imgsz=1280,classes=[0,1,2,3,5,7],conf=.25,device='cpu',verbose=False)[0]
            actors=actor_result.boxes.xyxy.cpu().tolist()
            for x,y,r,b in tiles:
                result=model.predict(frame[y:b,x:r],imgsz=640,conf=.2,device='cpu',verbose=False)[0]
                for row in result.boxes.data.cpu().tolist():
                    x1,y1,x2,y2,confidence,category=row[:6]
                    box=[round(x1+x),round(y1+y),round(x2+x),round(y2+y)]
                    detections.append({'box':box,'confidence':confidence,'label':result.names[int(category)]})
            if detections:
                nms_boxes=[[d['box'][0],d['box'][1],d['box'][2]-d['box'][0],d['box'][3]-d['box'][1]] for d in detections]
                keep=cv2.dnn.NMSBoxes(nms_boxes,[d['confidence'] for d in detections],.2,.5)
                detections=[detections[int(i)] for i in np.asarray(keep).reshape(-1)]
            raw_count=len(detections)
            for d in detections:
                x,y,r,b=d['box']; cx,cy=min(w-1,(x+r)//2),min(h-1,(y+b)//2)
                regions=[]
                for zone in camera['zones']:
                    if (masks[zone['region_id']][cy,cx] and min(r-x,b-y)>=zone['minimum_short_side_px']
                            and (r-x)*(b-y)>=zone['minimum_box_area_px']):
                        regions.append(zone['region_id'])
                d['regions']=regions
                d['actor_overlap']=max((box_overlap_fraction(d['box'],a) for a in actors),default=0)
                d['retained']=bool(regions) and d['actor_overlap']<.2
                if d['retained']:
                    cv2.rectangle(annotation,(x,y),(r,b),(0,220,0),3)
                    cv2.putText(annotation,d['label'],(x,max(y-8,25)),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,220,0),2)
            elapsed=round(time.monotonic()-start,3)
        cv2.putText(annotation,'DRAFT ROI - ownership & night conditions not validated',
                    (25,h-25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,230,230),2)
        cv2.imwrite(str(args.output/(camera['device_code']+'.jpg')),annotation)
        row={'device_code':camera['device_code'],'reference_sha256':hashlib.sha256(Path(camera['reference_image']).read_bytes()).hexdigest(),
             'native_size':[w,h],'ground_area_fraction':round(area/(w*h),4),'tile_count':len(tiles),
             'tiles':tiles,'raw_after_nms':raw_count,'retained_candidates':sum(d['retained'] for d in detections),
             'actor_boxes':actors,'detections':detections,'cpu_inference_seconds':elapsed}
        report['cameras'].append(row)
        (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print({k:v for k,v in row.items() if k not in {'tiles','actor_boxes','detections'}},flush=True)


if __name__ == '__main__':
    main()
