"""Create a redacted, checksummed copy of a local demo collection."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

_RTSP_CREDENTIALS = re.compile(r"(?i)rtsp://[^\s/?]+:[^\s/@]+@")
_SECRET = re.compile(
    r"(?i)("
    r"(?:api[_-]?key|password|token|secret|lease_token|camera_control_key)\s*[=:]\s*)"
    r"[^\s,}\"]+"
)
_SECRET_KEYS = re.compile(
    r"(?i)(?:api[_-]?key|password|token|secret|lease_token|camera_control_key)"
)


def _redact_json(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if _SECRET_KEYS.fullmatch(str(key))
            else _redact_json(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_json(item) for item in value]
    if isinstance(value, str):
        value = _RTSP_CREDENTIALS.sub("rtsp://[REDACTED]@", value)
        return _SECRET.sub(r"\1[REDACTED]", value)
    return value


def _redact_content(content: str, suffix: str) -> str:
    if suffix == ".json":
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            pass
        else:
            return json.dumps(_redact_json(parsed), ensure_ascii=False, indent=2) + "\n"
    content = _RTSP_CREDENTIALS.sub("rtsp://[REDACTED]@", content)
    return _SECRET.sub(r"\1[REDACTED]", content)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.input.is_dir():
        parser.error("--input must be a directory")
    if args.output.exists():
        parser.error("--output already exists; choose a new directory")
    args.output.mkdir(parents=True)
    manifest = []
    for source in sorted(args.input.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(args.input)
        destination = args.output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix in {".json", ".jsonl", ".log", ".txt"}:
            content = source.read_text(encoding="utf-8", errors="replace")
            content = _redact_content(content, source.suffix.lower())
            destination.write_text(content, encoding="utf-8")
        else:
            shutil.copy2(source, destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        manifest.append({"path": str(relative), "sha256": digest, "bytes": destination.stat().st_size})
    (args.output / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "files": len(manifest)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
