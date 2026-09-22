#!/usr/bin/env python3
"""Serve the repository root so the static review UI can read local evidence."""
from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    handler = partial(SimpleHTTPRequestHandler, directory=str(REPO_ROOT))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(
        f"Local URL: http://127.0.0.1:{args.port}/tools/ground_litter_review_ui/dist/index.html",
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
