"""Worker-owned policy loop, serialized with queries and manual commands."""
from __future__ import annotations

import threading
import time


class PolicyMotion:
    def __init__(self, backend, period_s, lock, on_progress, on_stopped):
        self.backend = backend
        self.period_s = max(.01, float(period_s))
        self.lock = lock
        self.on_progress = on_progress
        self.on_stopped = on_stopped
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            # Pause/reset is acknowledged only after an in-flight physics step ends.
            self._thread.join()

    def _run(self):
        last_progress = 0.0
        detail = {}
        try:
            while not self._stop.is_set():
                started = time.monotonic()
                with self.lock:
                    if self._stop.is_set():
                        break
                    detail = self.backend.apply("step")
                if started - last_progress >= .25:
                    self.on_progress(detail)
                    last_progress = started
                if detail.get("rejected") or detail.get("terminated") or detail.get("truncated"):
                    if not self._stop.is_set():
                        self.on_stopped("rejected" if detail.get("rejected") else "episode_done", detail)
                    return
                self._stop.wait(max(0, self.period_s - (time.monotonic() - started)))
        except Exception as error:
            if not self._stop.is_set():
                self.on_stopped("policy_error", {**detail, "last_error": f"{type(error).__name__}: {error}"})
