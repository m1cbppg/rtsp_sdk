#!/usr/bin/env python3
"""Validate a review dataset and render a source-balanced contact sheet."""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path
import cv2, numpy as np


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); a=p.parse_args(); review=a.root/"review"
    data=json.loads((review/"review-data.json").read_text()); missing=[]
    for row in data["items"]:
        for key in ("context_image","crop_image"):
            if not (review/row[key]).is_file(): missing.append(row[key])
    selected=[]
    for source in sorted({row["source"] for row in data["items"]}):
        group=[row for row in data["items"] if row["source"]==source]
        chosen=[group[round(i*(len(group)-1)/3)] for i in range(4)] if len(group)>=4 else group
        selected.extend(chosen)
    cells=[]
    for row in selected:
        image=cv2.imread(str(review/row["context_image"])); image=cv2.resize(image,(360,240),interpolation=cv2.INTER_AREA)
        cv2.rectangle(image,(0,0),(360,32),(8,15,30),-1); cv2.putText(image,f"{row['source']} | {row['timestamp'][11:19]}",(8,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA); cells.append(image)
    while len(cells)%4: cells.append(np.zeros((240,360,3),np.uint8))
    sheet=np.vstack([np.hstack(cells[i:i+4]) for i in range(0,len(cells),4)]); cv2.imwrite(str(review/"contact_sheet.jpg"),sheet,[cv2.IMWRITE_JPEG_QUALITY,90])
    text=(a.root/"frames_manifest.json").read_text()+(a.root/"semantic_proposals.json").read_text()+(review/"review-data.json").read_text()
    report={"dataset":data["dataset"],"fingerprint":data["fingerprint"],"items":data["count"],"source_counts":dict(Counter(row["source"] for row in data["items"])),"missing_assets":missing,"temporary_ps_remaining":len(list((a.root/"raw").glob("*.ps"))),"signed_url_text_found":"http" in text.lower(),"contact_sheet":"contact_sheet.jpg"}
    (review/"VALIDATION.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)); print(json.dumps(report,ensure_ascii=False,indent=2)); return 0 if not missing and not report["temporary_ps_remaining"] and not report["signed_url_text_found"] else 2
if __name__=="__main__": raise SystemExit(main())
