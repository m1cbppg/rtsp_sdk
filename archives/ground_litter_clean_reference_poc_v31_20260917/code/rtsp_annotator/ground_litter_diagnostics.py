"""Bounded stage diagnostics for isolated litter analysis (no camera URLs)."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import platform
import time


@contextmanager
def timed_stage(stats, name):
    started = time.perf_counter()
    try:
        yield
    finally:
        timings = stats.setdefault('stage_seconds', {})
        timings[name] = timings.get(name, 0.0) + time.perf_counter() - started
        calls = stats.setdefault('stage_calls', {})
        calls[name] = calls.get(name, 0) + 1


def runtime_fingerprint(model_paths=()):
    paths = sorted(Path(__file__).parent.glob('ground_litter*.py'))
    paths += [Path(p) for p in model_paths]
    files = []
    for path in paths:
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        files.append({'path': str(path.resolve()), 'sha256': digest})
    return {'python': platform.python_version(), 'files': files}
