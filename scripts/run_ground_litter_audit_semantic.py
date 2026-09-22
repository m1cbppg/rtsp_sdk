#!/usr/bin/env python3
"""Run low-threshold full-frame and tiled trash-model proposals on extracted frames."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO


def iou(a, b):
    x1,y1,x2,y2=a; X1,Y1,X2,Y2=b
    inter=max(0,min(x2,X2)-max(x1,X1))*max(0,min(y2,Y2)-max(y1,Y1))
    return inter/max(1,(x2-x1)*(y2-y1)+(X2-X1)*(Y2-Y1)-inter)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--geometry",type=Path,required=True); p.add_argument("--model",required=True); p.add_argument("--conf",type=float,default=.025); p.add_argument("--frame-id",default=None); p.add_argument("--per-frame-limit",type=int,default=20); p.add_argument("--output",type=Path,default=None); a=p.parse_args()
    manifest=json.loads((a.root/"frames_manifest.json").read_text()); geometry=json.loads(a.geometry.read_text()); model=YOLO(a.model)
    frame_rows=[row for row in manifest["frames"] if a.frame_id is None or row["frame_id"]==a.frame_id]
    output=[]
    for index,row in enumerate(frame_rows):
        image=cv2.imread(str(a.root/row["paths"]["current"])); h,w=image.shape[:2]
        poly=np.asarray([[round(x*w),round(y*h)] for x,y in geometry["roi"]],np.int32)
        mask=np.zeros((h,w),np.uint8); cv2.fillPoly(mask,[poly],255)
        radius=round(min(h,w)*.08); mask=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))
        proposals=[]
        full=model.predict(image,conf=a.conf,imgsz=1280,verbose=False,device=0)[0]
        for box,conf,cls in zip(full.boxes.xyxy.cpu().numpy(),full.boxes.conf.cpu().numpy(),full.boxes.cls.cpu().numpy()):
            x1,y1,x2,y2=map(float,box); cx,cy=int((x1+x2)/2),int((y1+y2)/2)
            if 0<=cx<w and 0<=cy<h and mask[cy,cx]: proposals.append({"bbox":[round(x1),round(y1),round(x2),round(y2)],"score":float(conf),"class_id":int(cls),"class_name":model.names[int(cls)],"source":"semantic_full"})
        tile=768; step=576
        tiles=[]
        for y in list(range(0,max(1,h-tile+1),step))+[max(0,h-tile)]:
            for x in list(range(0,max(1,w-tile+1),step))+[max(0,w-tile)]:
                if np.count_nonzero(mask[y:y+tile,x:x+tile])>tile*tile*.08: tiles.append((x,y,image[y:y+tile,x:x+tile]))
        if tiles:
            results=model.predict([t[2] for t in tiles],conf=a.conf,imgsz=768,verbose=False,device=0,batch=8)
            for (ox,oy,_),result in zip(tiles,results):
                for box,conf,cls in zip(result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy(),result.boxes.cls.cpu().numpy()):
                    x1,y1,x2,y2=box; bbox=[round(x1+ox),round(y1+oy),round(x2+ox),round(y2+oy)]; cx,cy=(bbox[0]+bbox[2])//2,(bbox[1]+bbox[3])//2
                    if mask[cy,cx]: proposals.append({"bbox":bbox,"score":float(conf),"class_id":int(cls),"class_name":model.names[int(cls)],"source":"semantic_tile"})
        kept=[]
        for proposal in sorted(proposals,key=lambda x:x["score"],reverse=True):
            if all(iou(proposal["bbox"],old["bbox"])<.5 for old in kept): kept.append(proposal)
        selected=kept if a.per_frame_limit<=0 else kept[:a.per_frame_limit]
        output.append({"frame_id":row["frame_id"],"proposals":selected,"raw_kept_count":len(kept)})
        print(json.dumps({"frame":index+1,"total":len(frame_rows),"detections":len(kept),"saved":len(selected)}),flush=True)
    target=a.output or (a.root/"semantic_proposals.json")
    target.write_text(json.dumps({"model_names":model.names,"frames":output},ensure_ascii=False,indent=2))
    return 0
if __name__=="__main__": raise SystemExit(main())
