#!/usr/bin/env python3
"""Join review decisions to proposals and report source yield plus event-level dedupe."""
from __future__ import annotations
import argparse, json, math
from collections import Counter
from pathlib import Path


def iou(a,b):
    x1,y1,x2,y2=a; X1,Y1,X2,Y2=b
    inter=max(0,min(x2,X2)-max(x1,X1))*max(0,min(y2,Y2)-max(y1,Y1))
    return inter/max(1,(x2-x1)*(y2-y1)+(X2-X1)*(Y2-Y1)-inter)


def related(a,b):
    if a["device_code"]!=b["device_code"]: return False
    if iou(a["bbox"],b["bbox"])>=.2: return True
    def center(row): x1,y1,x2,y2=row["bbox"]; return ((x1+x2)/2,(y1+y2)/2,math.hypot(x2-x1,y2-y1))
    ax,ay,ad=center(a); bx,by,bd=center(b)
    return math.hypot(ax-bx,ay-by)<=max(24,.42*max(ad,bd))


def main():
    p=argparse.ArgumentParser(); p.add_argument("--review-dir",type=Path,required=True); a=p.parse_args()
    data=json.loads((a.review_dir/"review-data.json").read_text()); reviews=json.loads((a.review_dir/"reviews.json").read_text()); by_id={x["review_id"]:x for x in data["items"]}
    joined=[{**by_id[rid],**decision} for rid,decision in reviews["reviews"].items() if rid in by_id]
    source={}
    for name in sorted({x["source"] for x in data["items"]}):
        rows=[x for x in joined if x["source"]==name]; counts=Counter(x["label"] for x in rows); source[name]={"reviewed":len(rows),"labels":dict(counts),"litter_card_rate":round(counts["LITTER"]/len(rows),4) if rows else None}
    positives=[x for x in joined if x["label"]=="LITTER"]
    clusters=[]
    for row in positives:
        hit=next((group for group in clusters if any(related(row,old) for old in group)),None)
        if hit is None:
            clusters.append([row])
        else:
            hit.append(row)
    payload={"dataset":data["dataset"],"dataset_fingerprint":data["fingerprint"],"reviewed":len(joined),"label_counts":dict(Counter(x["label"] for x in joined)),"source_metrics":source,"positive_cards":len(positives),"estimated_distinct_positive_locations":len(clusters),"positive_clusters":[{"cluster_id":i+1,"cards":len(group),"sources":sorted({x["source"] for x in group}),"review_ids":[x["review_id"] for x in group],"timestamps":sorted({x["timestamp"] for x in group})} for i,group in enumerate(clusters)],"interpretation":{"selection_bias":"Candidate-card positive rates are proposal yield, not detector precision or recall.","next_batch":"Keep tiled semantic and random audit; reduce zero-yield temporal/texture share but retain a small audit quota."}}
    (a.review_dir/"REVIEW_REPORT.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)); print(json.dumps(payload,ensure_ascii=False,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
