"""Optional reviewed non-litter appearance checks; never learn from detections.

Templates are calibrated per camera/view/mode. A mismatch abstains, so new
objects adjacent to a tool are not hidden by a permanent ground exclusion.
"""
from __future__ import annotations

import hashlib
from datetime import date
import math
from pathlib import Path

import cv2
import numpy as np

from .ground_litter_geometry import box_overlap_fraction


def _box(value, width, height):
    if (not isinstance(value, list) or len(value) != 4
            or any(type(v) is not int for v in value)):
        raise ValueError('Facility boxes must contain four integer native pixels')
    x,y,r,b=value
    if not 0 <= x < r <= width or not 0 <= y < b <= height:
        raise ValueError('Facility box outside reference')
    return tuple(value)


def validate_facilities(camera, mode):
    raw=camera.get(mode,{}).get('reviewed_non_litter',[])
    if not isinstance(raw,list) or len(raw)>16:
        raise ValueError('At most 16 reviewed non-litter templates per mode')
    identifiers=set();width,height=camera['reference_size']
    for item in raw:
        if not isinstance(item,dict):raise ValueError('Invalid facility template')
        for key in ['id','reference_image','reference_sha256','reviewed_by','reviewed_at','reason']:
            if not isinstance(item.get(key),str) or not item[key].strip():
                raise ValueError('Facility template requires '+key)
        digest=item['reference_sha256']
        if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid facility reference fingerprint')
        try:date.fromisoformat(item['reviewed_at'])
        except ValueError:raise ValueError('Facility reviewed_at must be YYYY-MM-DD') from None
        if item['id'] in identifiers:raise ValueError('Duplicate facility ID')
        identifiers.add(item['id'])
        if item.get('camera_id')!=camera['device_code'] or item.get('view_id')!=camera['view_id']:
            raise ValueError('Facility camera/view mismatch')
        box=_box(item.get('box'),width,height)
        context=_box(item.get('context_box'),width,height)
        x,y,r,b=box;a,c,d,e=context
        if a>x or c>y or d<r or e<b or max(d-a,e-c)>512:
            raise ValueError('Facility context must contain item and be at most 512 pixels')
        if (d-a)*(e-c)<2*(r-x)*(b-y):
            raise ValueError('Facility context must include surroundings')
        for key,default in [('minimum_context_correlation',.9),('minimum_item_correlation',.9)]:
            value=item.get(key,default)
            if not isinstance(value,(int,float)) or not math.isfinite(value) or not .8<=value<=1:
                raise ValueError('Invalid facility correlation threshold')
    return raw


def _correlation(left,right):
    a=cv2.cvtColor(left,cv2.COLOR_BGR2GRAY)
    b=cv2.cvtColor(right,cv2.COLOR_BGR2GRAY)
    # Uniform rectangles cannot establish that a tool is the same object.
    if min(float(a.std()),float(b.std()))<5:return None
    return float(cv2.matchTemplate(b,a,cv2.TM_CCOEFF_NORMED)[0,0])


class ReviewedNonLitter:
    def __init__(self,camera,mode):
        self.shape=(camera['reference_size'][1],camera['reference_size'][0],3)
        self.templates=[]
        for item in validate_facilities(camera,mode):
            path=Path(item['reference_image'])
            with path.open('rb') as stream:
                if hashlib.sha256(stream.read()).hexdigest()!=item['reference_sha256']:
                    raise ValueError('Facility reference hash mismatch')
            frame=cv2.imread(str(path))
            if frame is None or list(frame.shape[1::-1])!=camera['reference_size']:
                raise ValueError('Facility reference size mismatch')
            x,y,r,b=item['box'];a,c,d,e=item['context_box']
            self.templates.append((dict(item),frame[y:b,x:r].copy(),frame[c:e,a:d].copy()))

    def filter(self,frame,proposals):
        if frame.shape!=self.shape:
            raise ValueError('Facility frame resolution changed')
        kept=[];decisions=[]
        # Context and item comparisons run once per template, not per candidate.
        matches=[]
        for item,reference,context in self.templates:
            x,y,r,b=item['box'];a,c,d,e=item['context_box']
            current=frame[y:b,x:r];surrounding=frame[c:e,a:d]
            if current.shape!=reference.shape or surrounding.shape!=context.shape:
                continue
            inner=_correlation(reference,current);outer=_correlation(context,surrounding)
            # Also require color appearance: grayscale shape alone can miss a
            # new colored bag laid over a tool. No exposure normalization here.
            color_error=float(np.mean(np.abs(current.astype(np.float32)-reference)))
            same=(inner is not None and outer is not None and
                  inner>=item.get('minimum_item_correlation',.9) and
                  outer>=item.get('minimum_context_correlation',.9) and color_error<=15)
            matches.append((item,same,inner,outer,color_error))
        for proposal in proposals:
            box=proposal['box'];matched=None
            for item,same,inner,outer,error in matches:
                anchor=item['box']
                if (box_overlap_fraction(box,anchor)<.8 or
                        box_overlap_fraction(anchor,box)<.5):continue
                decision={'box':box,'facility_id':item['id'],'suppressed':same,
                          'item_correlation':inner,'context_correlation':outer,
                          'color_error':error,'reason':'reviewed_non_litter' if same else 'appearance_changed_or_uncertain'}
                if len(decisions)<128:decisions.append(decision)
                if same:matched=item;break
            if matched is None:kept.append(proposal)
        return kept,decisions
