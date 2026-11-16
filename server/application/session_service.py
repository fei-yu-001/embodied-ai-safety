"""Persist session commands before a worker can execute them."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import secrets
import threading
import uuid

from server.domain.control_lease import ControlLease
from server.domain.session_control import ControlError, SessionControl


class WorkerUnavailable(Exception):
    pass


class SessionService:
    def __init__(self, root, worker, clock=None):
        self.root = Path(root)
        self.worker = worker
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self._clock = self.clock
        self._lock = threading.RLock()
        self._cache = {}
        self._config = {}
        self._last_detail = {}
        self._telemetry = {}

    def create(self, config=None):
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        control = SessionControl(session_id, secrets.token_urlsafe(32), clock=self._clock)
        with self._lock:
            self._cache[session_id] = control
            self._config[session_id] = config or {}
            self._persist_new(control, control.create_event())
            self._persist_new(control, control.mark_loading())
            try:
                # The worker reports load completion through on_loaded, which
                # moves the session from LOADING to READY exactly once.
                self.worker.start(session_id, config or {})
            except Exception as error:
                if control.state != "INTERRUPTED":
                    self._persist_new(control, control.mark_worker_lost(f"仿真进程未能启动：{error}")[1])
                detail = str(error)
                if any(term in detail.lower() for term in ("memoryerror", "bad allocation", "out of memory")):
                    detail = "内存不足，厨房仿真未载入。请结束不再使用的会话后重新加载环境。"
                raise WorkerUnavailable(f"{detail}（会话 {session_id}）") from error
            return self.describe(session_id)

    def mark_ready(self, session_id):
        with self._lock:
            control = self._load(session_id)
            self._persist_new(control, control.mark_ready())
            return self.describe(session_id)

    def describe(self, session_id):
        """Session state plus what the worker reported about its backend."""
        control = self._load(session_id)
        snapshot = control.snapshot()
        info = getattr(self.worker, "info", {}).get(session_id) or {}
        snapshot["live"] = bool(getattr(self.worker, "alive", lambda *_: False)(session_id))
        snapshot["backend"] = info.get("backend")
        snapshot["loads_physics"] = bool(info.get("loads_physics", False))
        snapshot["environment"] = {
            **(info.get("state") if isinstance(info.get("state"), dict) else {}),
            **self._telemetry.get(session_id, {}),
        }
        snapshot["config"] = self._config.get(session_id, {})
        snapshot["last_command"] = self._last_detail.get(session_id)
        return snapshot

    def submit(self, envelope):
        with self._lock:
            control = self._load(session_id := envelope.session_id)
            before = len(control.events)
            ack, _event, created = control.submit(envelope, self.clock())
            if created:
                self._persist_tail(control, before)
            if not created or ack.acknowledgement != "accepted" or control.inflight_id != envelope.command_id:
                return ack
            try:
                self.worker.submit(session_id, envelope.command, envelope.command_id, envelope.args)
            except Exception as error:
                lost, _event = control.mark_worker_lost(f"仿真进程未能接收命令：{error}")
                self._persist_tail(control, before + 1)
                raise WorkerUnavailable(lost.message if lost else session_id) from error
            return ack

    def observe_applied(self, session_id, command_id, detail=None):
        with self._lock:
            control = self._load(session_id)
            before = len(control.events)
            ack, _event = control.mark_applied(command_id)
            self._persist_tail(control, before)
            if detail:
                self._last_detail[session_id] = detail
                self._absorb(session_id, detail)
            return ack

    def policy_progress(self, session_id, detail):
        with self._lock:
            self._absorb(session_id, detail)

    def policy_stopped(self, session_id, reason, detail):
        with self._lock:
            control = self._load(session_id)
            before = len(control.events)
            control.mark_policy_stopped(reason)
            self._persist_tail(control, before)
            self._absorb(session_id, detail)
            self._telemetry.setdefault(session_id, {})["policy_stop"] = {"reason": reason, "error": detail.get("last_error")}

    def manual_progress(self, session_id, steps, detail):
        with self._lock:
            self._absorb(session_id, detail)
            telemetry = self._telemetry.setdefault(session_id, {})
            # 连续点动的步数单独记，不覆盖环境的累计步数
            telemetry["manual"] = {"steps": steps}

    def _absorb(self, session_id, detail):
        if not isinstance(detail, dict):
            return
        telemetry = self._telemetry.setdefault(session_id, {})
        # frame_error / published_frame_seq 是帧发布状态，不是累计遥测：
        # 成功发布时必须能把 cleared 的 None 写回去，否则旧错会永久残留。
        for key in ("frame_error", "frame_warning", "published_frame_seq"):
            if key in detail:
                telemetry[key] = detail[key]
        for key in ("step", "episode", "sim_seconds", "frame_seq", "reward", "success", "terminated", "truncated"):
            if detail.get(key) is not None:
                telemetry[key] = detail[key]

    def crash_worker(self, session_id, reason="工作进程已退出，未自动重放命令"):
        self.worker.crash(session_id)
        with self._lock:
            control = self._load(session_id)
            before = len(control.events)
            _ack, event = control.mark_worker_lost(reason)
            if event is not None:
                self._persist_tail(control, before)
            return control.snapshot()

    def record_safety(self, session_id, message):
        with self._lock:
            control = self._load(session_id)
            before = len(control.events)
            event = control.record_safety(message)
            self._persist_tail(control, before)
            return event

    def get(self, session_id):
        with self._lock:
            return self.describe(session_id)

    def events_after(self, session_id, seq):
        with self._lock:
            return self._load(session_id).events_after(seq)

    def acquire_lease(self, session_id, holder, ttl_ms):
        with self._lock:
            return self._load(session_id).acquire_lease(holder, ttl_ms).to_dict()
    def scene(self, session_id, geometry_key=None):
        """World-frame state from the worker; the API process holds no sim state.

        `geometry_key` is what the client already has. Matching keys suppress the
        static geometry payload, so polling stays small.
        """
        with self._lock:
            self._load(session_id)
        try:
            return self.worker.query(session_id, "scene", args={"have_geometry_key": geometry_key})
        except Exception as error:
            raise WorkerUnavailable(str(error)) from error

    def renew_lease(self, session_id, lease_id, ttl_ms):
        with self._lock:
            view = self._load(session_id).renew_lease(lease_id, ttl_ms)
            # One heartbeat drives both the lease and the worker's own
            # keepalive timeout, so there is a single clock to reason about.
            try:
                self.worker.keepalive(session_id, lease_id)
            except Exception:
                pass
            return view.to_dict()

    def manual_stopped(self, session_id, reason, steps=0, measured=None):
        with self._lock:
            control = self._load(session_id)
            before = len(control.events)
            _ack, event = control.mark_manual_stopped(reason, steps)
            if event is not None:
                self._persist_tail(control, before)
            self._telemetry.setdefault(session_id, {})["manual"] = {
                "reason": reason,
                "steps": steps,
                "measured_m": measured,
            }
            return control.snapshot()

    def release_lease(self, session_id, lease_id):
        with self._lock:
            return self._load(session_id).release_lease(lease_id)

    def _load(self, session_id):
        if session_id in self._cache:
            return self._cache[session_id]
        events = self._log(session_id).read_all()
        if not events:
            raise KeyError(session_id)
        control = SessionControl.from_events(events)
        control.lease = ControlLease(clock=self._clock)
        if control.inflight_id and not self.worker.alive(session_id):
            _ack, event = control.mark_worker_lost("恢复时发现工作进程不在，原命令未重放")
            if event is not None:
                self._log(session_id).append(event)
        self._cache[session_id] = control
        return control

    def _persist_new(self, control, event):
        self._log(control.session_id).append(event)

    def _persist_tail(self, control, before):
        log = self._log(control.session_id)
        for event in control.events[before:]:
            log.append(event)

    def _log(self, session_id):
        if not session_id.startswith("sess_") or "/" in session_id or "\\" in session_id:
            raise KeyError(session_id)
        return __import__("server.infrastructure.event_log", fromlist=["EventLog"]).EventLog(
            self.root / session_id / "events.jsonl"
        )
