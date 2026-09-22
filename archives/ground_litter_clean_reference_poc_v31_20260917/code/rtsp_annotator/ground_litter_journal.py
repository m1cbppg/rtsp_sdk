"""Bounded local evidence/log storage for the isolated litter pilot."""
from datetime import datetime, timezone
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


class PilotJournal:
    def __init__(self, directory, run_id, *, max_bytes=10*1024*1024, backups=6):
        self.run_id = run_id
        self.handler = RotatingFileHandler(Path(directory)/'observations.jsonl',
            maxBytes=max_bytes, backupCount=backups, encoding='utf-8')
        self.handler.setFormatter(logging.Formatter('%(message)s'))

    def write(self, event, **fields):
        payload = {'event': event, 'run_id': self.run_id,
                   'observed_at': datetime.now(timezone.utc).isoformat(), **fields}
        record = logging.LogRecord('litter_pilot', logging.INFO, '', 0,
                                  json.dumps(payload, ensure_ascii=False), (), None)
        # Invoke directly: a full disk must stop the pilot, not silently lose evidence.
        if self.handler.shouldRollover(record):
            self.handler.doRollover()
        self.handler.stream.write(self.handler.format(record)+'\n')
        self.handler.flush()

    def close(self):
        self.handler.close()


def retain_samples(directory, limit=240):
    files = sorted(Path(directory).glob('sample-*.jpg'))
    for path in files[:max(0, len(files)-limit)]:
        path.unlink()


def output_bytes(directory):
    return sum(p.stat().st_size for p in Path(directory).rglob('*') if p.is_file())
