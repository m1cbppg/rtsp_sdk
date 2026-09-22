"""Model, temporal evidence and reference-ground checks for the live pilot."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from .event_engine import NormalizedRect
from .ground_litter_geometry import prepare, box_overlap_fraction
from .ground_litter_inventory import ConfirmedLitter, _same_location
from .ground_litter_change import ChangeConfig, compare_patch, propose_changes
from .ground_litter_profile import BackgroundProfile
from .ground_litter_alignment import ViewAlignment
from .ground_litter_diagnostics import timed_stage
from .ground_litter_facility import ReviewedNonLitter,validate_facilities


def normalized(box, width, height):
    x,y,r,b=box
    return NormalizedRect(x/width,y/height,(r-x)/width,(b-y)/height)


def pixels(rectangle, width, height):
    return [round(rectangle.left*width),round(rectangle.top*height),
            round((rectangle.left+rectangle.width)*width),round((rectangle.top+rectangle.height)*height)]


def validate_profiles(payload, *, allow_draft=False):
    if payload.get('schema_version') != 2 or not payload.get('cameras'):
        raise ValueError('Expected camera profile schema 2')
    if len({c['device_code'] for c in payload['cameras']}) != len(payload['cameras']):
        raise ValueError('Duplicate device code')
    model=payload['model']
    if model['tile_size_px'] != 640 or not 0 <= model['tile_overlap'] < .8:
        raise ValueError('Unsupported native tile configuration')
    for camera in payload['cameras']:
        code=camera['device_code']
        if not isinstance(code,str) or not code.isdigit() or len(code)!=20:
            raise ValueError('Invalid device code')
        if not allow_draft and (not camera['enabled'] or camera['calibration_status']!='verified'):
            raise ValueError('Camera not calibrated; --allow-draft permits shadow testing only')
        for key,low,high in [('analysis_fps',.2,5),('frame_max_age_seconds',.1,10)]:
            value=camera[key]
            if not math.isfinite(value) or not low<=value<=high:
                raise ValueError('Invalid '+key)
        if not camera.get('view_id') or not camera.get('zones'):
            raise ValueError('Fixed view and ground regions required')
        if camera.get('view_alignment','sift') not in {'sift','orb'}:
            raise ValueError('view_alignment must be orb or sift')
        for mode in ('day','night'):
            settings=camera[mode]
            BackgroundProfile.from_camera(camera, mode)
            validate_facilities(camera,mode)
            if not (0<settings['minimum_confidence']<=1 and 1<=settings['confirm_seconds']<=300):
                raise ValueError('Invalid confirmation threshold')
        rules=camera['evidence']
        if not (1<=rules['minimum_hits']<=rules['hit_window']<=20
                and 0<rules['minimum_hit_fraction']<=1 and 0<=rules['actor_clear_seconds']<=30):
            raise ValueError('Invalid evidence rules')
        w,h=camera['reference_size']
        if not (640<=w<=7680 and 640<=h<=4320):
            raise ValueError('Unsupported reference resolution')
        prepare(camera,np.zeros((h,w,3),np.uint8),640,model['tile_overlap'])
    return payload


@dataclass
class EvidenceTrack:
    anchor: NormalizedRect
    first: float
    last: float
    region_id: str
    merchant_id: str | None
    history: deque = field(default_factory=lambda:deque(maxlen=1500))
    visible_since: float = 0
    confirmed: bool = False
    observed_seconds: float = 0
    previous_observable: bool = False
    clear_observed_seconds: float = 0


class EvidenceWindow:
    """One-to-one anchored associations; misses count, repeated frames do not."""
    def __init__(self, camera, mode):
        self.camera=camera
        self.seconds=camera[mode]['confirm_seconds']
        self.rules=camera['evidence']
        self.max_gap=self.rules.get('max_observation_gap_seconds', max(3, 1.5/camera.get('analysis_fps',1)))
        self.tracks=[]
        self.previous=None
        self.diagnostics=[]
        self.reset_reason=None
        self.last_event=None
        self.interrupted=False

    def unknown(self,timestamp):
        """Pause a short unknown interval without turning it into evidence."""
        if self.last_event is not None and timestamp<=self.last_event:
            raise ValueError('Repeated or old frame')
        self.last_event=timestamp
        self.interrupted=True
        for track in self.tracks:
            track.previous_observable=False
            track.clear_observed_seconds=0
        self.diagnostics=[]
        self.reset_reason='unknown_observation'
        if self.previous is not None and timestamp-self.previous>self.max_gap:
            self.tracks=[]
            self.reset_reason='unknown_gap'

    def observe(self,timestamp,observations,actors=()):
        self.diagnostics=[]
        self.reset_reason=None
        if self.last_event is not None and timestamp<=self.last_event:
            raise ValueError('Repeated or old frame')
        delta=0 if self.previous is None or self.interrupted else timestamp-self.previous
        if self.previous is not None and timestamp-self.previous>self.max_gap:
            self.tracks=[]  # inventory identities are separate and survive
            self.reset_reason='observation_gap'
        self.previous=timestamp
        self.last_event=timestamp
        self.interrupted=False
        self.tracks=[t for t in self.tracks if timestamp-t.last<=self.max_gap]
        used=set()
        for observation in observations:
            available=[(i,t) for i,t in enumerate(self.tracks) if i not in used
                       and _same_location(t.anchor,observation.rectangle)]
            if available:
                i,track=min(available,key=lambda pair:sum((a-b)**2 for a,b in zip(pair[1].anchor.center,observation.rectangle.center)))
                track.last=timestamp
            else:
                i=len(self.tracks)
                track=EvidenceTrack(observation.rectangle,timestamp,timestamp,
                    observation.region_id,observation.merchant_id,visible_since=timestamp)
                self.tracks.append(track)
            used.add(i)
        confirmed=[]
        for i,track in enumerate(self.tracks):
            visible=i in used
            # Actor overlap resets clear visibility for this item only.
            actor_overlap=max((_overlap(track.anchor,a) for a in actors),default=0)
            observable=actor_overlap<=.2
            # Only intervals bounded by usable, unoccluded observations count.
            # Neither outages nor local occlusion can advance confirmation.
            if observable and track.previous_observable:
                track.observed_seconds+=delta
                track.clear_observed_seconds+=delta
            if not observable:
                track.clear_observed_seconds=0
            track.previous_observable=observable
            if actor_overlap>.2:
                track.visible_since=timestamp
                visible=False
            track.history.append(visible)
            recent=list(track.history)[-self.rules['hit_window']:]
            fraction=sum(track.history)/len(track.history)
            enough=(track.observed_seconds>=self.seconds
                    and track.clear_observed_seconds>=self.rules['actor_clear_seconds']
                    and sum(recent)>=self.rules['minimum_hits']
                    and fraction>=self.rules['minimum_hit_fraction'])
            track.confirmed |= enough and visible
            reasons=[]
            if not visible: reasons.append('actor_occluded' if actor_overlap>.2 else 'model_miss')
            if track.observed_seconds<self.seconds: reasons.append('confirm_duration')
            if track.clear_observed_seconds<self.rules['actor_clear_seconds']: reasons.append('actor_clear_duration')
            if sum(recent)<self.rules['minimum_hits']: reasons.append('recent_hits')
            if fraction<self.rules['minimum_hit_fraction']: reasons.append('hit_fraction')
            self.diagnostics.append(dict(
                rectangle=[track.anchor.left,track.anchor.top,track.anchor.width,track.anchor.height],
                span_seconds=round(timestamp-track.first,3),
                observed_seconds=round(track.observed_seconds,3),
                required_seconds=self.seconds, recent_hits=sum(recent),
                observations=len(track.history),hit_fraction=round(fraction,4),
                actor_overlap=round(actor_overlap,4),confirmed=track.confirmed,
                visible=visible,reasons=reasons))
            if track.confirmed and visible:
                confirmed.append(ConfirmedLitter(track.anchor,track.region_id,track.merchant_id))
        return tuple(confirmed)


def _overlap(left,right):
    return box_overlap_fraction(
        [left.left,left.top,left.left+left.width,left.top+left.height],
        [right.left,right.top,right.left+right.width,right.top+right.height])


def patch_unchanged(reference, current, box, *, minimum_change_area=160):
    """Conservative visible-ground comparison; never used without a supplied reference."""
    x,y,r,b=box
    if (r-x)*(b-y)<16:
        return False
    evidence=compare_patch(reference,current,tuple(box),
                           ChangeConfig(minimum_component_area=minimum_change_area))
    return evidence.change_fraction<.08 and evidence.largest_component_area<minimum_change_area


class LitterModel:
    def __init__(self, model_path, *, device='cpu', actor_imgsz=1280, diagnostic_candidates=False):
        from ultralytics import YOLO
        actor_path=Path('models/yolo26s.pt')
        if not Path(model_path).is_file() or not actor_path.is_file():
            raise ValueError('Both model files must already exist locally')
        if not isinstance(actor_imgsz, int) or not 320 <= actor_imgsz <= 1280:
            raise ValueError('actor_imgsz must be an integer in [320, 1280]')
        self.model=YOLO(str(model_path)); self.actor=YOLO(str(actor_path)); self.device=device
        # Keep the audited GPU baseline. 640 is an explicit A/B option, not a
        # verified five-camera performance or actor-recall improvement.
        self.actor_imgsz=actor_imgsz
        self.diagnostic_candidates=diagnostic_candidates
        self.last_stats={}

    def analyze(self, frame, camera, tiles, masks, mode):
        self.last_stats={'raw_before_nms':0, 'raw_after_nms':0,
                         'roi_size_rejected':0, 'actor_rejected':0,
                         'rejected_candidates':[], 'stage_seconds':{}, 'stage_calls':{}}
        with timed_stage(self.last_stats,'model_total'):
            return self._analyze(frame,camera,tiles,masks,mode)

    def _analyze(self, frame, camera, tiles, masks, mode):
        threshold=camera[mode]['minimum_confidence']
        h,w=frame.shape[:2]
        kwargs=dict(device=self.device,verbose=False)
        with timed_stage(self.last_stats,'actor_full'):
            actor_result=self.actor.predict(frame,imgsz=self.actor_imgsz,conf=.2,classes=[0,1,2,3,5,7],**kwargs)[0]
            actors=actor_result.boxes.xyxy.cpu().tolist()
        detections=[]
        for tile_index,(x,y,r,b) in enumerate(tiles):
            with timed_stage(self.last_stats,'litter_tiles'):
                result=self.model.predict(frame[y:b,x:r],imgsz=640,conf=threshold,**kwargs)[0]
                rows=result.boxes.data.cpu().tolist()
            for row in rows:
                a,c,d,e,confidence,category=row[:6]
                box=[max(0,round(a+x)),max(0,round(c+y)),min(w,round(d+x)),min(h,round(e+y))]
                if box[2]>box[0] and box[3]>box[1]:
                    detections.append(dict(box=box,confidence=confidence,label=result.names[int(category)],
                                           tile_index=tile_index))
        self.last_stats['raw_before_nms']=len(detections)
        if self.diagnostic_candidates:
            self.last_stats['raw_candidates']=[dict(d) for d in detections[:128]]
            self.last_stats['raw_trace_truncated']=len(detections)>128
        if detections:
            with timed_stage(self.last_stats,'nms'):
                boxes=[[d['box'][0],d['box'][1],d['box'][2]-d['box'][0],d['box'][3]-d['box'][1]] for d in detections]
                keep=np.asarray(cv2.dnn.NMSBoxes(boxes,[d['confidence'] for d in detections],threshold,.5)).reshape(-1)
                if self.diagnostic_candidates:
                    kept={int(i) for i in keep}
                    self.last_stats['nms_suppressed']=[dict(d) for i,d in enumerate(detections) if i not in kept][:128]
                detections=[detections[int(i)] for i in keep]
        self.last_stats['raw_after_nms']=len(detections)
        proposals=[]
        for detection in detections:
            box=detection['box'];x,y,r,b=box;cx,cy=min(w-1,(x+r)//2),min(h-1,(y+b)//2)
            eligible=[zone for zone in camera['zones'] if masks[zone['region_id']][cy,cx]
                      and min(r-x,b-y)>=zone['minimum_short_side_px']
                      and (r-x)*(b-y)>=zone['minimum_box_area_px']]
            if not eligible:
                self.last_stats['roi_size_rejected']+=1
                if len(self.last_stats['rejected_candidates'])<128:
                    inside=any(masks[z['region_id']][cy,cx] for z in camera['zones'])
                    self.last_stats['rejected_candidates'].append(dict(detection,reason='too_small' if inside else 'outside_roi'))
                continue
            if max((box_overlap_fraction(box,a) for a in actors),default=0)>.2:
                self.last_stats['actor_rejected']+=1
                if len(self.last_stats['rejected_candidates'])<128:
                    self.last_stats['rejected_candidates'].append(dict(detection,reason='actor_overlap'))
                continue
            # Same-view ROI crop improves actor detail where whole-frame
            # COCO misses small scooter/seat parts. Maximum one 640 patch per proposal.
            xx=max(0,min(w-640,cx-320)); yy=max(0,min(h-640,cy-320))
            with timed_stage(self.last_stats,'actor_local'):
                result=self.actor.predict(frame[yy:yy+640,xx:xx+640],imgsz=640,conf=.2,
                                          classes=[0,1,2,3,5,7],**kwargs)[0]
                local=[[a+xx,c+yy,d+xx,e+yy] for a,c,d,e in result.boxes.xyxy.cpu().tolist()]
            actors.extend(local)
            if max((box_overlap_fraction(box,a) for a in local),default=0)>.2:
                self.last_stats['actor_rejected']+=1
                if len(self.last_stats['rejected_candidates'])<128:
                    self.last_stats['rejected_candidates'].append(dict(detection,reason='local_actor_overlap'))
                continue
            # Ambiguous ownership stays unassigned, rather than notifying both shops.
            region=eligible[0]['region_id'] if len(eligible)==1 else 'ownership_pending'
            merchant=eligible[0]['merchant_id'] if len(eligible)==1 else None
            detection.update(region_id=region,merchant_id=merchant,source='model')
            proposals.append(detection)
        return proposals,actors


class CameraAnalysis:
    def __init__(self, camera, model_config, mode):
        self.camera=camera; self.mode=mode
        self.last_diagnostics={}
        w,h=camera['reference_size']
        self.masks,self.tiles,_=prepare(camera,np.zeros((h,w,3),np.uint8),
                                      model_config['tile_size_px'],model_config['tile_overlap'])
        self.evidence=EvidenceWindow(camera,mode)
        self.background=BackgroundProfile.from_camera(camera,mode)
        self.facilities=ReviewedNonLitter(camera,mode)
        self.clean=None
        if self.background.enabled:
            self.clean=cv2.imread(str(self.background.reference_image))
            if self.clean is None or self.clean.shape[:2]!=(h,w):
                raise ValueError('Background reference image invalid')
        self.generation=None
        self.reference=cv2.imread(camera.get(mode,{}).get('reference_image',camera['reference_image']))
        if self.reference is None or self.reference.shape[:2]!=(h,w):
            raise ValueError('View calibration image invalid')
        self.alignment=ViewAlignment(self.reference,camera)
        # Optional reviewed clean-ground frame; never learn unknown startup litter as clean.
        # The optional clean reference above is separate from the existing
        # conservative clear-evidence reference.  Neither reference is
        # learned automatically from an unreviewed live frame.
        clean_path=camera.get(mode,{}).get('clean_reference_image')
        self.clear_reference=None
        if clean_path:
            self.clear_reference=cv2.imread(clean_path)
            if self.clear_reference is None or self.clear_reference.shape[:2]!=(h,w):
                raise ValueError('Clean ground reference size invalid')

    def consume(self,model,inventory,frame,timestamp,generation, *, captured_at=None,
                enforce_freshness=True):
        self.last_diagnostics={}
        with timed_stage(self.last_diagnostics,'analysis_total'):
            return self._consume(model,inventory,frame,timestamp,generation,
                                 captured_at=captured_at,enforce_freshness=enforce_freshness)

    def _consume(self,model,inventory,frame,timestamp,generation, *, captured_at=None,
                 enforce_freshness=True):
        # Replay evidence uses source PTS, freshness uses local acquisition
        # monotonic time. Fast offline inference cannot shorten confirmation.
        acquired=timestamp if captured_at is None else captured_at
        if frame.shape != self.reference.shape:
            raise ValueError('resolution_changed')
        if self.generation!=generation:
            self.evidence=EvidenceWindow(self.camera,self.mode)
            self.generation=generation
        try:
            with timed_stage(self.last_diagnostics,'alignment'):
                self.alignment.check(frame)
        finally:
            self.last_diagnostics['alignment']=dict(self.alignment.diagnostics)
        try:
            proposals,actors=model.analyze(frame,self.camera,self.tiles,self.masks,self.mode)
        finally:
            for key,value in getattr(model,'last_stats',{}).items():
                if key in ('stage_seconds','stage_calls'):
                    self.last_diagnostics.setdefault(key,{}).update(value)
                else:
                    self.last_diagnostics[key]=value
        self.last_diagnostics['actors']=actors[:128]
        self.last_diagnostics['semantic_candidates']=len(proposals)
        if self.facilities.templates:
            with timed_stage(self.last_diagnostics,'reviewed_non_litter'):
                proposals,decisions=self.facilities.filter(frame,proposals)
            self.last_diagnostics['facility_decisions']=decisions
            self.last_diagnostics['facility_suppressed']=sum(d['suppressed'] for d in decisions)
        if self.background.enabled:
            changed=[]
            for proposal in proposals:
                evidence=compare_patch(self.clean,frame,tuple(proposal['box']),
                                       self.background.change)
                proposal['background_changed']=evidence.changed
                proposal['change_fraction']=round(evidence.change_fraction,4)
                proposal['largest_change_area']=evidence.largest_component_area
                if evidence.changed or not self.background.require_change:
                    changed.append(proposal)
            proposals=changed
            self.last_diagnostics['background_vetoed']=self.last_diagnostics['semantic_candidates']-len(proposals)
            if self.background.generate_proposals:
                mask=np.maximum.reduce(list(self.masks.values()))
                boxes,state=propose_changes(self.clean,frame,mask,actors,self.background.change)
                self.last_diagnostics['background_state']=state
                h,w=frame.shape[:2]
                for box in boxes:
                    x,y,r,b=box;cx,cy=(x+r)//2,(y+b)//2
                    eligible=[z for z in self.camera['zones'] if self.masks[z['region_id']][cy,cx]
                              and min(r-x,b-y)>=z['minimum_short_side_px'] and (r-x)*(b-y)>=z['minimum_box_area_px']]
                    if not eligible or any(box_overlap_fraction(box,d['box'])>.2 or
                        box_overlap_fraction(d['box'],box)>.2 for d in proposals):
                        continue
                    proposals.append(dict(box=box,confidence=None,label='GroundObject',source='change_only',
                        region_id=eligible[0]['region_id'] if len(eligible)==1 else 'ownership_pending',
                        merchant_id=eligible[0]['merchant_id'] if len(eligible)==1 else None,
                        background_changed=True,review_required=True))
        self.last_diagnostics['change_only_candidates']=sum(d.get('source')=='change_only' for d in proposals)
        if enforce_freshness and time.monotonic()-acquired>self.camera['frame_max_age_seconds']:
            raise ValueError('analysis_result_stale')
        h,w=frame.shape[:2]
        observations=tuple(ConfirmedLitter(normalized(d['box'],w,h),d['region_id'],d['merchant_id']) for d in proposals)
        # Pixel changes remain review proposals. They cannot supply semantic
        # hits to the litter confirmation window or create a litter identity.
        semantic=tuple(o for o,d in zip(observations,proposals)
                       if d.get('source')!='change_only')
        actor_rects=tuple(normalized(a,w,h) for a in actors)
        confirmed=self.evidence.observe(timestamp,semantic,actor_rects)
        self.last_diagnostics['confirmation']=self.evidence.diagnostics
        self.last_diagnostics['confirmation_reset']=self.evidence.reset_reason
        camera_id=self.camera['device_code'];view=self.camera['view_id']
        clear=[]
        if self.clear_reference is not None:
            for row in inventory.active(camera_id,view):
                rect=NormalizedRect(*json.loads(row['anchor']))
                if any(_overlap(rect,a)>.1 for a in actor_rects):
                    continue
                if any(_same_location(rect,o.rectangle) for o in observations):
                    continue
                if patch_unchanged(self.clear_reference,frame,pixels(rect,w,h)):
                    clear.append(row['item_id'])
        result=inventory.observe(camera_id,view,timestamp,confirmed,clear_item_ids=tuple(clear))
        return proposals,actors,confirmed,result
