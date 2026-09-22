#!/usr/bin/env python3
"""Fuse independent proposal sources into a bounded human-review dataset."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import hashlib
import cv2, numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from rtsp_annotator.ground_litter_audit_dataset import Proposal,dataset_fingerprint,dedupe_persistent_proposals,expanded_mask,merge_proposals,polygon_mask,random_grid_proposals,temporal_proposals,texture_proposals


def crop(image,box,scale):
    x1,y1,x2,y2=box; h,w=image.shape[:2]; cx,cy=(x1+x2)/2,(y1+y2)/2; bw=max(32,(x2-x1)*scale); bh=max(32,(y2-y1)*scale)
    side=max(bw,bh); return image[max(0,round(cy-side/2)):min(h,round(cy+side/2)),max(0,round(cx-side/2)):min(w,round(cx+side/2))]


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--geometry",type=Path,required=True); p.add_argument("--max-items",type=int,default=220); p.add_argument("--quota-profile",choices=("pilot","formal","adaptive"),default="pilot"); p.add_argument("--dataset",default="pilot_01030_20260920"); a=p.parse_args()
    manifest=json.loads((a.root/"frames_manifest.json").read_text()); semantic=json.loads((a.root/"semantic_proposals.json").read_text()); geom=json.loads(a.geometry.read_text()); sem={r["frame_id"]:r["proposals"] for r in semantic["frames"]}
    proposals=[]; images={}
    for row in manifest["frames"]:
        before=cv2.imread(str(a.root/row["paths"]["before"])); current=cv2.imread(str(a.root/row["paths"]["current"])); after=cv2.imread(str(a.root/row["paths"]["after"])); images[row["frame_id"]]={"before":before,"current":current,"after":after}
        core=polygon_mask(current.shape[:2],geom["roi"]); audit=expanded_mask(core,.08)
        for item in sem.get(row["frame_id"],[]):
            source=item["source"]
            if a.quota_profile=="adaptive":
                band="high" if item["score"]>=.7 else "mid" if item["score"]>=.3 else "low"
                source=f"{source}_{band}"
            proposals.append(Proposal(row["frame_id"],tuple(item["bbox"]),source,item["score"]))
        proposals += temporal_proposals(row["frame_id"],before,current,after,audit,4)
        proposals += texture_proposals(row["frame_id"],current,audit,limit=4)
        proposals += random_grid_proposals(row["frame_id"],current,audit,count=3)
    # The budgets sum to max-items so the source order cannot starve random
    # coverage.  Random cards are the independent miss-rate audit for every
    # automatic proposer and therefore need a real, visible share of the batch.
    if a.quota_profile=="pilot":
        quotas={"semantic_tile":60,"semantic_full":30,"temporal":40,"texture":40,"random_grid":50}
    elif a.quota_profile=="formal":
        weights={"semantic_tile":.45,"semantic_full":.15,"temporal":.05,"texture":.05,"random_grid":.30}
        quotas={key:int(a.max_items*value) for key,value in weights.items()}
        quotas["random_grid"]+=a.max_items-sum(quotas.values())
        proposals=dedupe_persistent_proposals(proposals)
    else:
        weights={"semantic_tile_high":.10,"semantic_tile_mid":.30,"semantic_tile_low":.20,"semantic_full_high":.025,"semantic_full_mid":.05,"semantic_full_low":.025,"random_grid":.25,"temporal":.025,"texture":.025}
        quotas={key:int(a.max_items*value) for key,value in weights.items()}
        quotas["random_grid"]+=a.max_items-sum(quotas.values())
        proposals=dedupe_persistent_proposals(proposals)
    selected=merge_proposals(proposals,quotas,a.max_items); review=a.root/"review"; assets=review/"assets"; assets.mkdir(parents=True,exist_ok=True); rows_by_id={r["frame_id"]:r for r in manifest["frames"]}; items=[]
    for index,item in enumerate(selected):
        triplet=images[item.frame_id]; image=triplet["current"]; x1,y1,x2,y2=item.bbox; x1,x2=sorted((max(0,x1),min(image.shape[1],x2))); y1,y2=sorted((max(0,y1),min(image.shape[0],y2)))
        if x2-x1<3 or y2-y1<3: continue
        marked=image.copy(); cv2.rectangle(marked,(x1,y1),(x2,y2),(0,0,255),max(2,image.shape[1]//900))
        context=crop(marked,(x1,y1,x2,y2),4.0); detail=crop(image,(x1,y1,x2,y2),1.8)
        rid=f"{item.frame_id}-{index:04d}"; cpath=assets/f"{rid}-context.jpg"; dpath=assets/f"{rid}-crop.jpg"; bpath=assets/f"{rid}-before.jpg"; apath=assets/f"{rid}-after.jpg"
        cv2.imwrite(str(cpath),context,[cv2.IMWRITE_JPEG_QUALITY,90]); cv2.imwrite(str(dpath),detail,[cv2.IMWRITE_JPEG_QUALITY,94]); cv2.imwrite(str(bpath),crop(triplet["before"],(x1,y1,x2,y2),4.0),[cv2.IMWRITE_JPEG_QUALITY,86]); cv2.imwrite(str(apath),crop(triplet["after"],(x1,y1,x2,y2),4.0),[cv2.IMWRITE_JPEG_QUALITY,86])
        meta=rows_by_id[item.frame_id]; items.append({"review_id":rid,"device_code":meta["device_code"],"timestamp":meta["timestamp"],"file_id":meta["file_id"],"frame_id":item.frame_id,"bbox":[x1,y1,x2,y2],"source":item.source,"score":round(float(item.score),6),"context_image":str(cpath.relative_to(review)),"crop_image":str(dpath.relative_to(review)),"before_image":str(bpath.relative_to(review)),"after_image":str(apath.relative_to(review))})
    # Blind, deterministic order: the reviewer should not work through one
    # proposal source at a time or infer the expected answer from the source.
    items.sort(key=lambda row: hashlib.sha256(("audit-order-v1:"+row["review_id"]).encode()).hexdigest())
    payload={"dataset":a.dataset,"count":len(items),"labels":["LITTER","NON_LITTER","BOX_WRONG","UNCERTAIN"],"items":items}; payload["fingerprint"]=dataset_fingerprint(items)
    (review/"review-data.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)); print(json.dumps({"items":len(items),"sources":{s:sum(i["source"]==s for i in items) for s in sorted({i["source"] for i in items})},"fingerprint":payload["fingerprint"]},ensure_ascii=False,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
