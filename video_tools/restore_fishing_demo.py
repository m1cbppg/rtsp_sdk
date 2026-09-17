#!/usr/bin/env python3
"""Local, masked temporal restoration for this fixed demonstration recording.

Never modifies the source. The vessel number is a supplied demonstration value.
Restoration is estimated, not recovery of guaranteed ground truth. Outside the
restoration mask decoded pixels remain identical before the new overlay/encode.
"""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

cv2.setNumThreads(4)
FONT = "/System/Library/Fonts/Hiragino Sans GB.ttc"


def osd_mask(frame):
    b, g, r = [x.astype(np.int16) for x in cv2.split(frame)]
    red = (((r > 75) & (r-g > 35) & (r-b > 30))
           | ((g > 100) & (g-r > 65) & (g-b > 50))
           | ((g > 110) & (b > 110) & (r < .6*g) & (r < .6*b)
              & (g > .8*b) & (b > .8*g))).astype(np.uint8) * 255
    red[:int(len(red) * .24)] = 0
    horizontal = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((1, 55), np.uint8))
    vertical = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((45, 1), np.uint8))
    # Exclude compact real coloured objects such as the red fuel container.
    for layer, is_horizontal in ((horizontal,True),(vertical,False)):
        _,labels,stats,_=cv2.connectedComponentsWithStats(layer)
        for label,(x,y,w,h,area) in enumerate(stats[1:],start=1):
            long_axis,short_axis=(w,h) if is_horizontal else (h,w)
            if long_axis < 150 and short_axis > 25:
                layer[labels==label]=0
    lines = horizontal | vertical
    # Old solid red labels include white lettering: close those holes, then
    # include only compact wide blocks connected to a detected box edge.
    blocks = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((9, 17), np.uint8))
    blocks = cv2.morphologyEx(blocks, cv2.MORPH_OPEN, np.ones((29, 29), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(blocks)
    for x, y, w, h, area in stats[1:]:
        if 25 <= w <= 550 and 25 <= h <= 140 and w / h > .5:
            region = lines[max(0,y-8):y+h+12, max(0,x-8):x+w+12]
            if cv2.countNonZero(region) > w:
                lines[max(0,y-3):y+h+3, max(0,x-3):x+w+3] = 255
    # Include antialiasing/chroma fringes without modifying whole rectangles.
    lines = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    return cv2.dilate(lines, np.ones((11, 11), np.uint8))


class Donors:
    def __init__(self, source, streaming=False):
        self.cap = cv2.VideoCapture(str(source),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2])
        self.duration = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.cap.get(cv2.CAP_PROP_FPS)
        self.cache = OrderedDict()
        self.streaming = streaming
        self.next_sample = 0
        self.frame_index = 0
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.sample_rate = 4

    def prepare(self, time):
        if not self.streaming:
            return
        if self.next_sample==0 and time>2:
            self.next_sample=max(0,int((time-1.5)*self.sample_rate))
            self.frame_index=round(self.next_sample/self.sample_rate*self.fps)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES,self.frame_index)
        while self.next_sample / self.sample_rate <= min(time + 1.5, self.duration - .08):
            target = round(self.next_sample / self.sample_rate * self.fps)
            while self.frame_index <= target:
                ok = self.cap.grab()
                self.frame_index += 1
                if not ok:
                    return
            ok, frame = self.cap.retrieve()
            if not ok:
                return
            mask = osd_mask(frame)
            # Keep losslessly compressed donors in RAM, avoiding gigabytes of
            # decoded frames and resulting macOS swap pressure.
            packed=cv2.imencode(".png",frame,[cv2.IMWRITE_PNG_COMPRESSION,1])[1]
            self.cache[self.next_sample/self.sample_rate] = (packed, mask, flow_gray(frame,mask))
            self.next_sample += 1
            while len(self.cache)>16:
                self.cache.popitem(last=False)

    def get(self, time):
        rate=self.sample_rate if self.streaming else 8
        time = round(max(0, min(time, self.duration - .08)) * rate) / rate
        if time not in self.cache:
            if self.streaming:
                return None
            self.cap.set(cv2.CAP_PROP_POS_MSEC, time * 1000)
            ok, frame = self.cap.read()
            if not ok:
                return None
            mask = osd_mask(frame)
            gray = flow_gray(frame, mask)
            self.cache[time] = (frame, mask, gray)
        self.cache.move_to_end(time)
        while len(self.cache) > 40:
            self.cache.popitem(last=False)
        packed,mask,gray=self.cache[time]
        return (cv2.imdecode(packed,cv2.IMREAD_COLOR),mask,gray) if self.streaming else (packed,mask,gray)


def flow_gray(frame, mask):
    size = (frame.shape[1] // 4, frame.shape[0] // 4)
    gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(mask, size, interpolation=cv2.INTER_AREA)
    return cv2.inpaint(gray, (small_mask > 0).astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)


def restore(frame, time, donors):
    mask = osd_mask(frame)
    out = frame.copy()
    count = cv2.countNonZero(mask)
    if not count:
        return out, mask, {"masked": 0, "temporal": 0, "spatial": 0}
    donors.prepare(time)
    gray = flow_gray(frame, mask)
    ys, xs = np.where(mask > 0)
    x0, x1 = max(0, int(xs.min()) - 16), min(frame.shape[1], int(xs.max()) + 17)
    y0, y1 = max(0, int(ys.min()) - 16), min(frame.shape[0], int(ys.max()) + 17)
    target = frame[y0:y1, x0:x1]
    active = mask[y0:y1, x0:x1] > 0
    remaining = active.copy()
    roi_out = out[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    # Nearest trustworthy frame wins; do not average donor textures into blur.
    for offset in (-.375, .375, -1.25, 1.25):
        donor = donors.get(time + offset)
        if donor is None:
            continue
        other, other_mask, other_gray = donor
        flow = dis.calc(gray, other_gray, None)
        back = dis.calc(other_gray, gray, None)
        sh, sw = gray.shape
        sy, sx = np.mgrid[:sh, :sw].astype(np.float32)
        backward = cv2.remap(back, sx + flow[:,:,0], sy + flow[:,:,1], cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=1000)
        consistent = np.linalg.norm(flow + backward, axis=2) < .85
        # Photometric agreement in a surrounding patch catches wrong surface
        # matches. Masked pixels themselves must not influence this test.
        warped_gray = cv2.remap(other_gray, sx + flow[:,:,0], sy + flow[:,:,1], cv2.INTER_LINEAR)
        residual = cv2.absdiff(gray, warped_gray).astype(np.float32)
        residual = cv2.boxFilter(residual, -1, (9, 9))
        confidence = consistent & (residual < 14)
        full = cv2.resize(flow, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        dx = full[y0:y1,x0:x1,0] * (frame.shape[1] / sw)
        dy = full[y0:y1,x0:x1,1] * (frame.shape[0] / sh)
        mx, my = xx + dx, yy + dy
        valid = cv2.resize(confidence.astype(np.uint8), (frame.shape[1], frame.shape[0]),
                           interpolation=cv2.INTER_NEAREST)[y0:y1,x0:x1] > 0
        # Reject donor pixels near its own OSD, including interpolation support.
        donor_blocked = cv2.dilate(other_mask, np.ones((5,5),np.uint8))
        warped_mask = cv2.remap(donor_blocked, mx, my, cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        valid &= (warped_mask == 0) & remaining
        warped = cv2.remap(other, mx, my, cv2.INTER_LINEAR)
        # Validate actual unmasked RGB neighbours as well as flow. Inpainted
        # grayscale inside a thick OSD label is not trustworthy evidence.
        known = ((~active) & (warped_mask == 0)).astype(np.float32)
        diff = np.max(cv2.absdiff(target, warped),axis=2).astype(np.float32)
        weight = cv2.boxFilter(known, -1, (41,41))
        error = cv2.boxFilter(diff*known, -1, (41,41))/np.maximum(weight,.001)
        valid &= (error < 10) & (weight > .15)
        roi_out[valid] = warped[valid]
        remaining[valid] = False
        if remaining.sum() < count * .025:
            break
    spatial = int(remaining.sum())
    if spatial:
        # This only fills the unresolved thin lines/labels, not the boat region.
        repaired = cv2.inpaint(roi_out, remaining.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
        roi_out[remaining] = repaired[remaining]
    assert np.array_equal(out[mask == 0], frame[mask == 0])
    return out, mask, {"masked": count, "temporal": count-spatial, "spatial": spatial}


def at_keyframes(keys, time):
    if not keys or time < keys[0][0] or time > keys[-1][0]:
        return None
    for a, b in zip(keys, keys[1:]):
        if a[0] <= time <= b[0]:
            alpha = (time-a[0]) / (b[0]-a[0])
            return [int(round(x*(1-alpha)+y*alpha)) for x,y in zip(a[1:], b[1:])]
    return [int(v) for v in keys[-1][1:]]


@lru_cache(maxsize=4)
def card(text, subtitle):
    im = Image.new("RGB", (720, 165), (9, 26, 37))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((1,1,718,163), radius=12, outline=(255,199,66), width=3)
    d.text((22,16), subtitle, font=ImageFont.truetype(FONT,26), fill=(180,209,220))
    d.text((22,65), text, font=ImageFont.truetype(FONT,53), fill=(255,220,114))
    return np.asarray(im)[:,:,::-1].copy()


def overlay(frame, time, annotations):
    ship = at_keyframes(annotations.get("vessel", []), time)
    if ship:
        x,y,w,h = ship
        cv2.rectangle(frame,(x,y),(min(frame.shape[1]-4,x+w+70),y+h+15),(220,182,80),2,cv2.LINE_AA)
    plate = at_keyframes(annotations.get("plate", []), time)
    start, end = annotations.get("recognition_window", [32,58])
    if plate and start <= time <= end:
        x,y,w,h = plate
        color = (66,199,255)
        for px, py, sx, sy in ((x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)):
            cv2.line(frame,(px,py),(px+sx*min(35,w//4),py),color,5,cv2.LINE_AA)
            cv2.line(frame,(px,py),(px,py+sy*min(18,h//2)),color,5,cv2.LINE_AA)
        panel = card("粤清城渔10032" if time >= start+1.2 else "船牌定位中…", "船号识别 · 演示")
        ph,pw = panel.shape[:2]
        frame[220:220+ph,65:65+pw] = panel
    return frame


def prepare_annotations(source, output):
    """Dense editable keyframes; review them visually before final rendering."""
    c=cv2.VideoCapture(str(source))
    fps=c.get(cv2.CAP_PROP_FPS)
    ship_keys=[]
    plate_keys=[]
    index=0
    next_time=10.
    while True:
        ok=c.grab()
        if not ok:
            break
        t=index/fps
        index+=1
        if t < next_time:
            continue
        next_time+=.125
        ok,f=c.retrieve()
        if not ok:
            break
        mask=osd_mask(f)
        ys,xs=np.where(mask>0)
        if len(xs)>100:
            x,y,x2,y2=int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())
            w,h=x2-x,y2-y
            if w>70 and h>45:
                # Labels sit above boat boxes; allow room for the seated person.
                pad=int(np.clip(w*.035,12,85))
                ship_keys.append([t,max(3,x-15),max(0,y-pad),min(f.shape[1]-max(3,x-15)-4,w+30),h+pad+15])
        if t<29:
            continue
        hsv=cv2.cvtColor(f,cv2.COLOR_BGR2HSV)
        blue=cv2.inRange(hsv,np.array((98,95,55),np.uint8),np.array((128,255,255),np.uint8))
        blue[:730]=0
        blue[1100:]=0
        blue=cv2.morphologyEx(blue,cv2.MORPH_CLOSE,np.ones((7,15),np.uint8))
        stats=cv2.connectedComponentsWithStats(blue)[2][1:]
        candidates=[s for s in stats if 100<s[2]<350 and 35<s[3]<125 and 2<s[2]/s[3]<6 and s[4]>2500]
        if candidates:
            x,y,w,h,area=max(candidates,key=lambda s:s[4])
            plate_keys.append([t,int(x)-9,int(y)-7,int(w)+18,int(h)+14])
    c.release()
    # A short centered median suppresses noisy box size changes while preserving
    # camera pans. The plate samples are independent of the old detector IDs.
    for keys,radius in ((ship_keys,2),(plate_keys,1)):
        if not keys:
            continue
        raw=np.array(keys)
        for i,key in enumerate(keys):
            nearby=raw[max(0,i-radius):i+radius+1,1:]
            key[1:]=np.median(nearby,axis=0).astype(int).tolist()
    output.write_text(json.dumps({"source":str(source),"recognition_window":[32,60.86],
                                 "vessel":ship_keys,"plate":plate_keys},ensure_ascii=False,indent=2))
    print(f"Prepared {len(ship_keys)} vessel and {len(plate_keys)} plate keyframes",flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--annotations",type=Path)
    p.add_argument("--sample",type=float,nargs="*")
    p.add_argument("--start",type=float,default=0)
    p.add_argument("--duration",type=float,default=0)
    p.add_argument("--lossless",action="store_true")
    p.add_argument("--prepare-annotations",action="store_true")
    args = p.parse_args()
    if args.input.resolve()==args.output.resolve():
        p.error("输出必须使用新路径，不能覆盖原视频")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.prepare_annotations:
        prepare_annotations(args.input,args.output)
        return
    ann = json.loads(args.annotations.read_text()) if args.annotations else {}
    donors = Donors(args.input, streaming=args.sample is None)
    if args.sample is not None:
        records=[]
        cap=cv2.VideoCapture(str(args.input))
        for t in args.sample:
            cap.set(cv2.CAP_PROP_POS_MSEC,t*1000)
            ok, frame=cap.read()
            if not ok:
                raise RuntimeError(f"Cannot read {t}s")
            out,mask,stat=restore(frame,t,donors)
            cv2.imwrite(str(args.output.parent/f"restored_{t:g}.png"),out)
            cv2.imwrite(str(args.output.parent/f"mask_{t:g}.png"),mask)
            cv2.imwrite(str(args.output.parent/f"demo_{t:g}.png"),overlay(out.copy(),t,ann))
            records.append({"time":t,**stat})
            print(records[-1],flush=True)
        args.output.write_text(json.dumps(records,indent=2))
        cap.release()
        donors.cap.release()
        return
    records=[]
    with av.open(str(args.input)) as src, av.open(str(args.output),"w") as dst:
        source=src.streams.video[0]
        source.codec_context.thread_count=2
        codec="libx264rgb" if args.lossless else "libx264"
        stream=dst.add_stream(codec,rate=source.average_rate)
        stream.width,stream.height=source.width,source.height
        stream.pix_fmt="bgr24" if args.lossless else "yuv420p"
        stream.time_base=source.time_base
        stream.codec_context.time_base=source.time_base
        stream.codec_context.thread_count=3
        stream.options={"crf":"0" if args.lossless else "12","preset":"fast"}
        first_pts=None
        cached_raw=None
        cached_restored=None
        cached_mask=None
        cached_stat=None
        for index, frame in enumerate(src.decode(source)):
            t=float(frame.time)
            if t < args.start:
                continue
            if args.duration and t >= args.start+args.duration:
                break
            raw=frame.to_ndarray(format="bgr24")
            # Screen capture contains repeated camera images with tiny codec
            # noise. Reuse ONLY a repair patch when its source image is stable;
            # every output frame still starts with its own decoded source.
            probe=cv2.resize(raw[450:1400],(420,120),interpolation=cv2.INTER_AREA)
            stable=cached_raw is not None and float(np.mean(cv2.absdiff(probe,cached_raw)))<.20
            mask=osd_mask(raw) if stable else None
            stable=stable and cv2.countNonZero(mask ^ cached_mask)<300
            if stable:
                restored=raw.copy()
                shared=(mask>0)&(cached_mask>0)
                restored[shared]=cached_restored[shared]
                extra=(mask>0)&(cached_mask==0)
                if extra.any():
                    filled=cv2.inpaint(restored,extra.astype(np.uint8)*255,3,cv2.INPAINT_TELEA)
                    restored[extra]=filled[extra]
                stat={**cached_stat,"reused_patch":True}
            else:
                restored,mask,stat=restore(raw,t,donors)
                cached_raw=probe
                cached_restored=restored.copy()
                cached_mask=mask.copy()
                cached_stat=stat
            result=overlay(restored,t,ann)
            output=av.VideoFrame.from_ndarray(result,format="bgr24")
            if first_pts is None:
                first_pts=frame.pts
            output.pts=frame.pts-first_pts
            output.time_base=frame.time_base
            for packet in stream.encode(output):
                dst.mux(packet)
            if index%60==0:
                print(f"frame={index} time={t:.3f}s mask={stat}",flush=True)
            records.append({"pts":frame.pts,"time":t,**stat})
        for packet in stream.encode():
            dst.mux(packet)
    donors.cap.release()
    args.output.with_suffix(".audit.json").write_text(json.dumps(records,indent=2))


if __name__ == "__main__":
    main()
