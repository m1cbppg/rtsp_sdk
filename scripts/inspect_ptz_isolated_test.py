#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

from register_ptz_isolated_stream import load_env


def get_json(url: str, header: str, key: str) -> dict | list:
    request = Request(url, headers={header: key})
    with urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode())


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总隔离PTZ测试状态")
    parser.add_argument("--env-file", type=Path, default=Path(".env.ptz-test"))
    parser.add_argument("--api-url", default="http://127.0.0.1:18081")
    parser.add_argument("--control-url", default="http://127.0.0.1:19080")
    args = parser.parse_args()
    env = load_env(args.env_file)
    streams = get_json(
        f"{args.api_url}/v1/streams",
        "X-Api-Key",
        env["PTZ_TEST_API_KEY"],
    )
    report = get_json(
        f"{args.control_url}/v1/test/report",
        "X-Camera-Control-Key",
        env["PTZ_TEST_CONTROL_KEY"],
    )
    output: dict = {"streams": streams, "virtual_camera": report}
    if streams:
        stream_id = streams[0]["stream_id"]
        output["verification_jobs"] = get_json(
            f"{args.api_url}/v1/vessel-verifications?stream_id={stream_id}&limit=100",
            "X-Api-Key",
            env["PTZ_TEST_API_KEY"],
        )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
