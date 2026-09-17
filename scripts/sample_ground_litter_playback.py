"""Resolve one documented playback RTSP URL and sample it without logging secrets."""
from __future__ import annotations

import argparse
from pathlib import Path

from rtsp_annotator.ground_litter_replay import sample_stream
from rtsp_annotator.playback_source import (
    PlaybackRequest, PlaybackUrlClient, build_ctseelink_playback_payload,
    redact_url,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--playback-time", required=True,
                        help="camera-local replay start, e.g. 2026-09-11 08:00:00")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, default=60)
    parser.add_argument("--sample-fps", type=float, default=0.5)
    parser.add_argument("--endpoint", required=True,
                        help="Reachable cloud playback/rtsp/by-time endpoint; live URL endpoint is rejected")
    args = parser.parse_args()
    payload = build_ctseelink_playback_payload(args.device, args.playback_time)
    request = PlaybackRequest(args.device, args.playback_time,
                              args.playback_time, payload)
    recording = PlaybackUrlClient(args.endpoint).resolve_recording(request)
    result = sample_stream(recording.url, args.output, duration_seconds=args.duration_seconds,
                           sample_fps=args.sample_fps, start_offset_seconds=recording.offset_seconds,
                           source_label=redact_url(recording.url))
    import json
    metadata = recording.metadata()
    metadata.update({"requested_playback_time": args.playback_time,
                     "sample_offset_applied": recording.offset_seconds,
                     "content_time_verified": False})
    (args.output/'recording.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
