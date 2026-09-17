"""Sample a local PS/MP4 replay into reviewable JPEG frames."""
from __future__ import annotations

import argparse
from pathlib import Path

from rtsp_annotator.ground_litter_replay import sample_video


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-fps", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=10000)
    args = parser.parse_args()
    result = sample_video(args.input, args.output, sample_fps=args.sample_fps,
                          max_frames=args.max_frames)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
