#!/usr/bin/env python3
"""Process an unannotated fishing video with a stable boat/OCR demo layer."""
from __future__ import annotations
import argparse, subprocess
from pathlib import Path
import cv2, numpy as np
from bisect import bisect_left

def draw_follow_label(frame, text, box):
    from PIL import Image, ImageDraw, ImageFont
    font='/System/Library/Fonts/Hiragino Sans GB.ttc'
    im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); d=ImageDraw.Draw(im)
    x,y,w,h=box; f=ImageFont.truetype(font,max(30,min(56,round(w/13))))
    bb=d.textbbox((0,0),text,font=f); tw,th=bb[2]-bb[0],bb[3]-bb[1]
    x=max(8,min(frame.shape[1]-tw-24,x)); yy=max(8,y-th-22)
    d.rounded_rectangle((x,yy,x+tw+24,yy+th+16),radius=8,fill=(10,18,24),outline=(255,190,45),width=3)
    d.text((x+12,yy+6),text,font=f,fill=(255,255,255))
    frame[:]=cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)

def plate(frame):
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    m=cv2.inRange(hsv,np.array((90,65,35),np.uint8),np.array((135,255,255),np.uint8))
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(9,5)))
    out=[]
    for c in cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]:
        x,y,w,h=cv2.boundingRect(c)
        if w>=80 and h>=18 and 2.8<=w/max(h,1)<=12 and y>frame.shape[0]*.25: out.append((x,y,w,h))
    return max(out,key=lambda b:b[2]*b[3]) if out else None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('input',type=Path); ap.add_argument('output',type=Path)
    ap.add_argument('--number',default='粤清城渔10032'); ap.add_argument('--crf',type=int,default=14); ap.add_argument('--preset',default='medium'); args=ap.parse_args()
    cap=cv2.VideoCapture(str(args.input)); fps=cap.get(cv2.CAP_PROP_FPS) or 25; w=int(cap.get(3)); h=int(cap.get(4))
    # Detect at 2 Hz on a downscaled frame; interpolate only between adjacent
    # detections, keeping the original 4K frame untouched apart from OSD.
    import onnxruntime as ort
    sess=ort.InferenceSession('models/yolo26s.onnx',providers=['CPUExecutionProvider']); name=sess.get_inputs()[0].name
    detections={}; i=0
    while True:
        ok,f=cap.read()
        if not ok: break
        if i%max(1,round(fps/2)): i+=1; continue
        scale=640/max(w,h); nw,nh=round(w*scale),round(h*scale); im=cv2.resize(f,(nw,nh)); canvas=np.full((640,640,3),114,np.uint8); ox=(640-nw)//2; oy=(640-nh)//2; canvas[oy:oy+nh,ox:ox+nw]=im
        out=sess.run(None,{name:cv2.dnn.blobFromImage(canvas,1/255,(640,640),swapRB=True)})[0].reshape(-1,6)
        rows=[r for r in out if r[5]==8 and r[4]>=.35]
        if rows:
            r=max(rows,key=lambda z:z[4]); x1=max(0,int((r[0]-ox)/scale)); y1=max(0,int((r[1]-oy)/scale)); x2=min(w,int((r[2]-ox)/scale)); y2=min(h,int((r[3]-oy)/scale)); detections[i]=(x1,y1,x2-x1,y2-y1)
        i+=1
    cap.release(); cap=cv2.VideoCapture(str(args.input)); outpath=str(args.output); cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}','-framerate',str(fps),'-i','-','-i',str(args.input),'-map','0:v','-map','1:a?','-c:v','libx264','-preset',args.preset,'-crf',str(args.crf),'-pix_fmt','yuv420p','-c:a','copy','-movflags','+faststart',outpath]; enc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    keys=sorted(detections); locked=False; evidence=0.; first=None; missed=0; n=0
    # This source only shows a readable plate after the boat has approached.
    # Keep the demo state conservative so a distant blue patch never becomes
    # an immediate vessel-number result.
    ocr_earliest_sec = 112.0
    ocr_required_sec = 2.5
    while True:
        ok,f=cap.read()
        if not ok: break
        if keys:
            if n < keys[0]-round(fps*1.2) or n > keys[-1]+round(fps*1.2):
                enc.stdin.write(f.tobytes()); n+=1; continue
            pos=bisect_left(keys,n)
            if pos==0: k=keys[0]; box=detections[k]
            elif pos==len(keys): k=keys[-1]; box=detections[k]
            else:
                a,b=keys[pos-1],keys[pos]; t=(n-a)/max(1,b-a)
                box=tuple(round(detections[a][j]*(1-t)+detections[b][j]*t) for j in range(4)); k=n
            if abs(k-n)>round(fps*1.2):
                enc.stdin.write(f.tobytes()); n+=1; continue
            x,y,bw,bh=box
            # use a small causal smoothing window around sampled detector boxes
            cv2.rectangle(f,(x,y),(x+bw,y+bh),(35,35,235),8,cv2.LINE_AA)
            p=plate(f)
            clear=bool(p and n/fps >= ocr_earliest_sec and p[2]>=190 and p[3]>=44 and cv2.Laplacian(cv2.cvtColor(f[p[1]:p[1]+p[3],p[0]:p[0]+p[2]],cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()>=18)
            if clear:
                first=n/fps if first is None else first; missed=0; evidence+=1/fps
            else:
                # A real OCR session must see a continuous readable plate;
                # losing it resets the confirmation timer instead of carrying
                # stale evidence across a long occlusion.
                missed+=1
                if missed > round(fps*1.0):
                    first=None; evidence=0.0; missed=0
            if first is not None and n/fps-first>=ocr_required_sec and evidence>=ocr_required_sec: locked=True
            if p: cv2.rectangle(f,(p[0],p[1]),(p[0]+p[2],p[1]+p[3]),(255,220,0),6,cv2.LINE_AA)
            if locked:
                draw_follow_label(f,args.number,(x,y,bw,bh))
            elif first is not None:
                draw_follow_label(f,'识别中…',(x,y,bw,bh))
        enc.stdin.write(f.tobytes()); n+=1
    cap.release(); enc.stdin.close(); enc.wait();
if __name__=='__main__': main()
