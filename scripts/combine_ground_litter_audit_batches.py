#!/usr/bin/env python3
"""Combine per-camera review manifests without copying their image assets."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--dataset",required=True); p.add_argument("--device",action="append",required=True); a=p.parse_args()
    items=[]; inputs=[]
    for device in a.device:
        review=a.root/"devices"/device/"review"; data=json.loads((review/"review-data.json").read_text()); inputs.append({"device_code":device,"dataset":data["dataset"],"fingerprint":data["fingerprint"],"count":data["count"]})
        for original in data["items"]:
            row=dict(original); old=row["review_id"]; row["review_id"]=f"{device[-5:]}-{old}"
            for key in ("context_image","crop_image","before_image","after_image"):
                if row.get(key): row[key]=f"devices/{device}/review/{row[key]}"
            items.append(row)
    items.sort(key=lambda row:hashlib.sha256(("combined-audit-order-v1:"+row["review_id"]).encode()).hexdigest())
    stable=[{"review_id":r["review_id"],"bbox":r["bbox"],"source":r["source"]} for r in items]
    fingerprint=hashlib.sha256(json.dumps(stable,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    payload={"dataset":a.dataset,"fingerprint":fingerprint,"count":len(items),"labels":["LITTER","NON_LITTER","BOX_WRONG","UNCERTAIN"],"inputs":inputs,"items":items}
    (a.root/"review-data.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)); print(json.dumps({"dataset":a.dataset,"count":len(items),"fingerprint":fingerprint,"inputs":inputs},ensure_ascii=False,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
