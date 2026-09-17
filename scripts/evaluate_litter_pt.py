#!/usr/bin/env python3
"""Evaluate a YOLO .pt litter detector inside a hand-authored camera ROI."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# Left storefront sidewalk in the supplied camera view. Coordinates are normalized.
SIDEWALK_ROI = [[0.02, 0.08], [0.37, 0.08], [0.46, 0.96], [0.02, 0.96]]
# Known fixed objects to exclude from the litter candidate area (normalized polygons).
EXCLUSION_ZONES = [
    [[0.22, 0.20], [0.39, 0.20], [0.43, 0.46], [0.21, 0.46]],  # bins/scooters
    [[0.00, 0.52], [0.17, 0.52], [0.20, 0.96], [0.00, 0.96]],  # storefront fixtures
]

def inside(pt, poly):
    return cv2.pointPolygonTest(np.asarray(poly, np.float32), pt, False) >= 0

def run(video: Path, model: YOLO, out_dir: Path, every_seconds: float, conf: float):
    cap = cv2.VideoCapture(str(video)); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0); stride = max(1, round(fps * every_seconds))
    frames = []; counts = {}; saved = 0; i = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if i % stride: i += 1; continue
        result = model.predict(frame, imgsz=1280, conf=conf, verbose=False, device="cpu")[0]
        detections = []
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.cpu().numpy(); scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            h, w = frame.shape[:2]
            for box, score, cls in zip(xyxy, scores, classes):
                x1,y1,x2,y2 = map(float, box); center=((x1+x2)/(2*w),(y1+y2)/(2*h))
                if not inside(center, SIDEWALK_ROI) or any(inside(center,z) for z in EXCLUSION_ZONES): continue
                label = result.names[int(cls)]
                item={"label":label,"confidence":round(float(score),4),"box":[round(x1),round(y1),round(x2),round(y2)]}
                detections.append(item); counts[label]=counts.get(label,0)+1
        frames.append({"time_s":round(i/fps,2),"detections":detections})
        if detections and saved < 20:
            ann=frame.copy(); h,w=ann.shape[:2]
            cv2.polylines(ann,[np.asarray([[round(x*w),round(y*h)] for x,y in SIDEWALK_ROI])],True,(0,255,255),3)
            for z in EXCLUSION_ZONES: cv2.polylines(ann,[np.asarray([[round(x*w),round(y*h)] for x,y in z])],True,(255,0,255),2)
            for d in detections:
                x1,y1,x2,y2=d['box']; cv2.rectangle(ann,(x1,y1),(x2,y2),(0,200,0),3)
                cv2.putText(ann,f"{d['label']} {d['confidence']:.2f}",(x1,max(20,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,200,0),2)
            cv2.imwrite(str(out_dir/f"{video.stem}_{saved:02d}.jpg"),ann); saved+=1
        i += 1
    cap.release()
    return {"duration_s":round(n/fps,2),"sampled_frames":len(frames),"frames_with_detections":sum(bool(x['detections']) for x in frames),"detection_count":sum(counts.values()),"counts_by_label":counts,"frames":frames}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',type=Path,required=True); ap.add_argument('--videos',type=Path,nargs='+',required=True); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--report',type=Path,required=True); ap.add_argument('--every-seconds',type=float,default=10); ap.add_argument('--confidence',type=float,default=.25); a=ap.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True); model=YOLO(str(a.model)); result={"model":str(a.model),"roi":SIDEWALK_ROI,"exclusion_zones":EXCLUSION_ZONES,"confidence":a.confidence,"videos":{v.name:run(v,model,a.output_dir,a.every_seconds,a.confidence) for v in a.videos}}
    a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({"model":result['model'],"videos":{k:{x:v for x,v in d.items() if x!='frames'} for k,d in result['videos'].items()}},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
