"""Continuous manual motion owned by the worker process.

The browser holds a lease and sends keepalives; it never drives the loop. If
keepalives stop arriving, motion stops on its own.
"""

from __future__ import annotations

import threading
import time

DEFAULT_KEEPALIVE_MS = 1500
MIN_KEEPALIVE_MS = 300
MAX_KEEPALIVE_MS = 10_000
MAX_MANUAL_SECONDS = 60.0


class ManualMotion:
    def __init__(self, backend, args, period_s, keepalive_ms, emit, lock=None, on_progress=None):
        self.backend = backend
        self.args = args
        self.period_s = max(0.01, float(period_s))
        self.keepalive_s = keepalive_ms / 1000.0
        self.emit = emit
        self.on_progress = on_progress
        self.manual_start_eef = None
        self._last_progress = 0.0
        self._lock = lock
        self._stop = threading.Event()
        self._last_keepalive = time.monotonic()
        self._thread = None
        self.steps = 0
        self.reason = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def keepalive(self):
        self._last_keepalive = time.monotonic()

    def stop(self, reason):
        if self.reason is None:
            self.reason = reason
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            # Wait for the final step AND manual_stopped before accepting reset/stop.
            self._thread.join()

    def _run(self):
        started = time.monotonic()
        while not self._stop.is_set():
            if time.monotonic() - self._last_keepalive > self.keepalive_s:
                self.reason = "lease_expired"
                break
            if time.monotonic() - started > MAX_MANUAL_SECONDS:
                self.reason = "manual_limit"
                break
            if self._lock is not None:
                with self._lock:
                    if self._stop.is_set():
                        break
                    result = self.backend.apply("jog", self.args)
            else:
                result = self.backend.apply("jog", self.args)
            self.steps += 1
            if self.on_progress is not None and time.monotonic() - self._last_progress > 0.5:
                self._last_progress = time.monotonic()
                self.on_progress(self.steps, result)
            if result.get("rejected"):
                self.reason = "rejected"
                break
            if result.get("terminated") or result.get("truncated"):
                self.reason = "episode_done"
                break
            self._stop.wait(self.period_s)
        if self.reason is None:
            self.reason = "stopped"
        self.emit(self.reason, self.steps)
