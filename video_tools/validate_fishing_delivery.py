#!/usr/bin/env python3
"""Verify complete decode, original frame times and unedited RGB samples."""
import argparse
import itertools
import json
from pathlib import Path

import av
import numpy as np

from restore_fishing_demo import osd_mask,overlay


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("master",type=Path)
    p.add_argument("delivery",type=Path)
    p.add_argument("annotations",type=Path)
    p.add_argument("report",type=Path)
    args=p.parse_args()
    ann=json.loads(args.annotations.read_text())
    audit=json.loads(args.master.with_suffix(".audit.json").read_text())
    samples=[]
    count=0
    max_time_error=0.
    with av.open(str(args.source)) as a,av.open(str(args.master)) as b,av.open(str(args.delivery)) as c:
        streams=[x.streams.video[0] for x in (a,b,c)]
        for s in streams:s.codec_context.thread_count=2
        for i,frames in enumerate(itertools.zip_longest(a.decode(streams[0]),b.decode(streams[1]),c.decode(streams[2]))):
            if any(f is None for f in frames):raise AssertionError("Different decoded frame counts")
            source,master,delivery=frames
            if (source.width,source.height)!=(delivery.width,delivery.height):
                raise AssertionError("Resolution changed")
            error=abs(float(source.time)-float(delivery.time))
            max_time_error=max(max_time_error,error)
            if error>1e-7:raise AssertionError(f"Changed PTS at frame {i}: {error}")
            if i%300==0:
                raw=source.to_ndarray(format="bgr24")
                restored=master.to_ndarray(format="bgr24")
                result=delivery.to_ndarray(format="bgr24")
                editable=osd_mask(raw)>0
                new_overlay=overlay(np.zeros_like(raw),float(source.time),ann)
                editable|=np.any(new_overlay!=0,axis=2)
                unedited=~editable
                exact=bool(np.array_equal(raw[unedited],restored[unedited]))
                if not exact:raise AssertionError(f"Master changed unmasked pixels at frame {i}")
                diff=raw[unedited].astype(np.float32)-result[unedited].astype(np.float32)
                mse=float(np.mean(diff*diff))
                samples.append({"frame":i,"time":float(source.time),"master_unedited_exact":exact,
                                "delivery_unedited_rgb_psnr_db":10*np.log10(255**2/max(mse,1e-12))})
                print(samples[-1],flush=True)
            count+=1
    if count!=len(audit):raise AssertionError("Audit count differs")
    total_pixels=streams[0].width*streams[0].height
    report={"frames":count,"width":streams[0].width,"height":streams[0].height,
            "max_source_delivery_timestamp_error_seconds":max_time_error,
            "mean_masked_percent":float(np.mean([r['masked']/total_pixels*100 for r in audit])),
            "temporal_repair_pixel_fraction":sum(r['temporal'] for r in audit)/max(1,sum(r['masked'] for r in audit)),
            "samples":samples,"decode_errors":0,
            "limitation":"Masked source detail is estimated; PSNR only assesses unedited pixels, not restoration accuracy."}
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print("Full validation passed",flush=True)


if __name__=="__main__":main()
