#!/usr/bin/env python3
"""Export the RGB lossless master to compatible MP4 with original source PTS."""
import argparse
import json
from pathlib import Path

import av


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("master",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    if args.output.resolve() in {args.source.resolve(),args.master.resolve()}:
        p.error("输出必须使用新路径，不能覆盖原视频或母版")
    records=json.loads(args.master.with_suffix(".audit.json").read_text())
    if not records:
        raise RuntimeError("Empty master audit")
    with av.open(str(args.source)) as original:
        base=original.streams.video[0].time_base
        rate=original.streams.video[0].average_rate
    with av.open(str(args.master)) as src, av.open(str(args.output),"w",options={"movflags":"+faststart"}) as dst:
        source=src.streams.video[0]
        source.codec_context.thread_count=2
        stream=dst.add_stream("libx264",rate=rate)
        stream.width,stream.height=source.width,source.height
        stream.pix_fmt="yuv420p"
        stream.time_base=base
        stream.codec_context.time_base=base
        stream.codec_context.thread_count=4
        stream.codec_context.colorspace=1
        stream.codec_context.color_primaries=1
        stream.codec_context.color_trc=1
        stream.codec_context.color_range=1
        stream.options={"crf":"10","preset":"fast"}
        count=0
        for i,f in enumerate(src.decode(source)):
            if i>=len(records):
                raise RuntimeError("Master has more frames than audit")
            rgb=f.reformat(format="rgb24")
            out=rgb.reformat(format="yuv420p",dst_colorspace="ITU709")
            out.pts=records[i]["pts"]-records[0]["pts"]
            out.time_base=base
            for packet in stream.encode(out):dst.mux(packet)
            count+=1
            if count%300==0:print(f"exported {count}/{len(records)}",flush=True)
        if count!=len(records):raise RuntimeError("Incomplete master")
        for packet in stream.encode():dst.mux(packet)
    print(f"Export complete: {count} frames",flush=True)


if __name__=="__main__":main()
