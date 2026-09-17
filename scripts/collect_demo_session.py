"""Passively collect a stream's health/detail JSON without moving a camera."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--stream-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key-env", default="RTSP_API_KEY")
    args = parser.parse_args()
    import os
    key = os.getenv(args.api_key_env)
    headers = {"X-API-Key": key} if key else {}
    root = args.base_url.rstrip("/")
    result: dict[str, object] = {"collected_at": time.time(), "stream_id": args.stream_id}
    for name, path in (("stream", f"/v1/streams/{args.stream_id}"), ("health", "/health")):
        try:
            with urlopen(Request(root + path, headers=headers), timeout=10) as response:
                result[name] = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            result[name + "_error"] = f"{type(exc).__name__}: {exc}"
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "session.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "errors": [k for k in result if k.endswith("_error")]}, ensure_ascii=False))
    return 0 if not any(k.endswith("_error") for k in result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
