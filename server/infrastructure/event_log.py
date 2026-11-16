"""Append-only session event log."""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading


class EventLog:
    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, event):
        line = json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())

    def read_all(self):
        if not self.path.is_file():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
