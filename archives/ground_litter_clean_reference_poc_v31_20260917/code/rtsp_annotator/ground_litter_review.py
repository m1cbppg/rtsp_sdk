"""Bounded evidence samples, including frames independent of detector output."""
from __future__ import annotations

import json
from pathlib import Path
import random

import cv2


class ReviewCollector:
    def __init__(self, output, *, limit=24, audit_interval=20, quality=88):
        if not 1 <= limit <= 1200 or audit_interval <= 0:
            raise ValueError("Invalid review sampling limits")
        self.output = Path(output)
        self.directory = self.output / "review_frames"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.limit, self.audit_interval, self.quality = limit, audit_interval, quality
        self.rng = random.Random(1021)
        self.seen = {"candidate": 0, "audit": 0}
        self.samples = {"candidate": [], "audit": []}
        self.next_audit = 0.0

    def _image(self, name, frame):
        ok, data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            raise RuntimeError("Review image encoding failed")
        path = self.directory / name
        temporary = path.with_suffix('.jpg.tmp')
        temporary.write_bytes(data.tobytes())
        temporary.replace(path)
        return str(path.relative_to(self.output))

    def _store(self, kind, frame, annotation, proposals, timestamp, sequence, diagnostics):
        self.seen[kind] += 1
        slot = (len(self.samples[kind]) if len(self.samples[kind]) < self.limit
                else self.rng.randrange(self.seen[kind]))
        if slot >= self.limit:
            return
        name = f"{kind}-{slot:04d}"
        image = self._image(name+'.jpg', annotation if kind == 'candidate' else frame)
        crops = []
        if kind == 'candidate':
            height, width = frame.shape[:2]
            for index, proposal in enumerate(proposals[:12]):
                x, y, r, b = proposal['box']
                margin = max(24, round(max(r-x, b-y)*.5))
                crop = frame[max(0,y-margin):min(height,b+margin),
                             max(0,x-margin):min(width,r+margin)]
                if crop.size:
                    crops.append(self._image(f'{name}-object-{index:02d}.jpg', crop))
        entry = {'review_id': name, 'kind': kind, 'source_time_seconds': round(timestamp,3),
                 'frame_sequence': sequence, 'image': image, 'crop_images': crops,
                 'candidates': proposals, 'confirmation': diagnostics.get('confirmation',[])}
        if slot == len(self.samples[kind]):
            self.samples[kind].append(entry)
        else:
            previous = self.samples[kind][slot]
            for old in set(previous['crop_images']) - set(crops):
                (self.output / old).unlink(missing_ok=True)
            self.samples[kind][slot] = entry

    def consider(self, frame, annotation, proposals, timestamp, sequence, diagnostics):
        if timestamp >= self.next_audit:
            self._store('audit', frame, annotation, proposals, timestamp, sequence, diagnostics)
            self.next_audit = timestamp + self.audit_interval
        if proposals:
            self._store('candidate', frame, annotation, proposals, timestamp, sequence, diagnostics)

    def entries(self):
        return sorted(self.samples['candidate'] + self.samples['audit'],
                      key=lambda entry: (entry['source_time_seconds'], entry['kind']))

    def save(self):
        path = self.output / 'candidate_review.json'
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(self.entries(),ensure_ascii=False,indent=2)+'\n')
        temporary.replace(path)
