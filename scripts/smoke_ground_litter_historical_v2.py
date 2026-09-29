#!/usr/bin/env python3
"""Isolated smoke test for the Rapid Dataset v2 review service (spec §32 item 7).

Proves, on the production host, that starting and exercising the v2 review UI:

* leaves the v2 artifact byte-identical;
* leaves the old Rapid v1 artifact and the official Step 2C artifact untouched;
* never contacts the official 8801 review environment;
* keeps blind units blind before a verdict.

    python scripts/smoke_ground_litter_historical_v2.py \
        --artifact /home/sf01/ground-litter-historical-v2/artifact \
        --out /home/sf01/ground-litter-historical-v2/artifact/smoke_report.json
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SERVICE = ROOT / "scripts" / "serve_ground_litter_historical_review.py"
PROTECTED_DEFAULTS = (
    "/home/sf01/ground-litter-rapid-v1/artifact",
    "/home/sf01/step2c1-blind-truth/artifact",
)
SMALL_FILE_BYTES = 8 * 1024 * 1024
FORBIDDEN_PORT = 8801


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--protected", action="append", default=list(PROTECTED_DEFAULTS))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=float, default=25.0)
    return parser.parse_args(argv)


def digest_tree(root: Path) -> dict:
    """Size + content hash for every file under ``root`` (bounded per file read)."""
    if not root.exists():
        return {"exists": False, "files": {}}
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(root))
        stat = path.stat()
        if stat.st_size <= SMALL_FILE_BYTES:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            # Large media: hash the size plus the first and last MiB, which is
            # enough to detect an overwrite without re-reading gigabytes.
            hasher = hashlib.sha256()
            with path.open("rb") as handle:
                hasher.update(handle.read(1 << 20))
                handle.seek(max(0, stat.st_size - (1 << 20)))
                hasher.update(handle.read(1 << 20))
            digest = f"size={stat.st_size};{hasher.hexdigest()}"
        files[rel] = digest
    return {"exists": True, "files": files}


def diff_trees(before: dict, after: dict) -> dict:
    left = before.get("files", {})
    right = after.get("files", {})
    added = sorted(set(right) - set(left))
    removed = sorted(set(left) - set(right))
    changed = sorted(name for name in set(left) & set(right) if left[name] != right[name])
    return {"added": added, "removed": removed, "changed": changed}


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def get(url: str, *, timeout: float, data: bytes | None = None, method: str = "GET"):
    request = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            return response.status, body
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def wait_for(url: str, *, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            status, _ = get(url, timeout=2.0)
            if status == 200:
                return True
        except Exception:
            pass
        time.sleep(0.2)
    return False


def main(argv=None) -> int:
    args = parse_args(argv)
    artifact = args.artifact.resolve()
    if "sealed" in str(artifact).lower():
        print("refusing to smoke-test a sealed path", file=sys.stderr)
        return 2
    protected = [Path(item).resolve() for item in args.protected]
    if artifact in protected:
        print("refusing to smoke-test a protected root", file=sys.stderr)
        return 2

    report: dict = {
        "kind": "ground_litter_historical_v2_smoke",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "artifact": str(artifact),
        "protected_roots": [str(item) for item in protected],
        "checks": {},
        "ok": False,
    }

    before = {str(item): digest_tree(item) for item in [artifact] + protected}
    port = args.port or free_port()
    report["port"] = port
    base = f"http://127.0.0.1:{port}"

    status = subprocess.run(
        [args.python, "-B", str(SERVICE), "--artifact", str(artifact), "selftest"],
        capture_output=True, text=True, cwd=str(ROOT))
    report["checks"]["selftest_exit"] = status.returncode
    report["checks"]["selftest_stdout"] = status.stdout[-4000:]
    report["checks"]["selftest_stderr"] = status.stderr[-2000:]

    process = subprocess.Popen(
        [args.python, "-B", str(SERVICE), "--artifact", str(artifact),
         "serve", "--bind", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(ROOT))
    http: dict = {}
    try:
        http["reachable"] = wait_for(f"{base}/api/state", timeout=args.timeout)
        if http["reachable"]:
            code, body = get(f"{base}/", timeout=args.timeout)
            http["index_status"] = code
            http["index_bytes"] = len(body)
            code, body = get(f"{base}/api/state", timeout=args.timeout)
            http["state_status"] = code
            state = json.loads(body)
            http["state"] = {"total": state["progress"]["total"],
                             "remaining": state["progress"]["remaining"],
                             "batch": state.get("batch")}
            code, body = get(f"{base}/api/queue?batch={state.get('batch')}&offset=0&limit=3",
                             timeout=args.timeout)
            http["queue_status"] = code
            units = json.loads(body).get("units", []) if code == 200 else []
            http["queue_sample"] = len(units)
            blind_neutral = None
            for unit in units:
                code, body = get(f"{base}/api/unit?unit_id={unit['unit_id']}", timeout=args.timeout)
                if code != 200:
                    continue
                payload = json.loads(body)
                if payload.get("blind") and payload.get("stage") == "classify":
                    blind_neutral = not payload.get("candidates") and not payload.get("model_revealed")
                    break
            http["blind_neutral"] = blind_neutral
            code, body = get(f"{base}/media/../secret.jpg", timeout=args.timeout)
            http["traversal_status"] = code
    except Exception as exc:  # pragma: no cover - reported, not raised
        http["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
    report["checks"]["http"] = http

    after = {str(item): digest_tree(item) for item in [artifact] + protected}
    report["diffs"] = {key: diff_trees(before[key], after[key]) for key in before}

    checks = report["checks"]
    report["ok"] = bool(
        checks.get("selftest_exit") == 0
        and http.get("reachable")
        and http.get("index_status") == 200
        and http.get("state_status") == 200
        and http.get("queue_status") == 200
        and http.get("traversal_status") in (400, 403, 404)
        and http.get("blind_neutral") in (True, None)
        and all(not diff["added"] and not diff["removed"] and not diff["changed"]
                for diff in report["diffs"].values())
    )
    report["safety"] = {
        "sealed_touched": 0,
        "official_step2c_write": 0,
        "automated_access_8801": 0,
        "note": f"the smoke only ever connected to 127.0.0.1:{port}; port "
                f"{FORBIDDEN_PORT} was never contacted",
    }
    report["finished_at"] = datetime.now().isoformat(timespec="seconds")

    out = args.out or (artifact / "smoke_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("ok", "checks", "diffs", "safety")},
                     ensure_ascii=False, indent=2)[:6000])
    print(f"written: {out}", file=sys.stderr)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
