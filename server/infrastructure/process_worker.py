"""Client side of the worker process: spawn, command, observe, crash."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from server.infrastructure.worker_protocol import ProtocolError, decode, encode

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class WorkerStartError(Exception):
    pass


def _bytes(message):
    return encode(message).encode("utf-8")


def _close(stream):
    try:
        if stream is not None:
            stream.close()
    except (OSError, ValueError):
        pass


def _expect(process, expected, timeout):
    """Read protocol messages until `expected` arrives; ignore stray lines."""
    deadline = time.monotonic() + timeout
    skipped = 0
    while time.monotonic() < deadline:
        line = process.stdout.readline()
        if not line:
            raise WorkerStartError("仿真进程已退出")
        try:
            message = decode(line.decode("utf-8").strip())
        except (ProtocolError, UnicodeDecodeError):
            skipped += 1
            if skipped > 2000:
                raise WorkerStartError("仿真进程输出不符合协议")
            continue
        if message.get("type") == expected:
            return message
        if message.get("type") == "error":
            raise WorkerStartError(message.get("message") or "仿真进程报告错误")
    raise WorkerStartError(f"等待 {expected} 超时")


class ProcessWorker:
    """One subprocess per session.

    `on_applied(session_id, command_id)` fires after the child confirms a
    command. `on_lost(session_id, reason)` fires once if the child dies, so the
    session is marked INTERRUPTED instead of quietly pretending success.
    """

    def __init__(self, root, *, on_loaded=None, on_applied=None, on_lost=None, on_manual_stopped=None, on_manual_progress=None, on_policy_progress=None, on_policy_stopped=None, python=None, boot_timeout=30.0, load_timeout=300.0, env=None, max_sessions=0):
        self.max_sessions = max_sessions
        self.root = Path(root)
        self.on_loaded = on_loaded or (lambda *_: None)
        self.on_applied = on_applied or (lambda *_: None)
        self.on_lost = on_lost or (lambda *_: None)
        self.on_manual_stopped = on_manual_stopped or (lambda *_: None)
        self.on_manual_progress = on_manual_progress or (lambda *_: None)
        self.on_policy_progress = on_policy_progress or (lambda *_: None)
        self.on_policy_stopped = on_policy_stopped or (lambda *_: None)
        self.python = python or sys.executable
        self.boot_timeout = boot_timeout
        self.load_timeout = load_timeout
        self.env = dict(os.environ if env is None else env)
        self.env["PRISM_SESSION_ROOT"] = str(self.root.resolve())
        self.info = {}
        self._pending = {}
        self._request_seq = 0
        self._processes = {}
        self._locks = {}
        self._lock = threading.RLock()

    def start(self, session_id, config=None):
        with self._lock:
            if self.max_sessions and sum(p.poll() is None for p in self._processes.values()) >= self.max_sessions:
                raise WorkerStartError("已达到服务器仿真会话上限，请先结束不再使用的会话")
            if session_id in self._processes:
                raise WorkerStartError("会话已在运行")
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            log_dir = self.root / session_id
            log_dir.mkdir(parents=True, exist_ok=True)
            log = (log_dir / "worker.log").open("wb")
            try:
                process = subprocess.Popen(
                    [self.python, "-X", "utf8", "-u", "-m", "server.infrastructure.simulation_worker"],
                    cwd=str(PROJECT_ROOT), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=log, creationflags=flags, env=self.env,
                )
            except OSError as error:
                log.close()
                raise WorkerStartError(f"无法启动仿真进程：{error}") from error
            self._processes[session_id] = process
            self._locks[session_id] = threading.Lock()
            try:
                self.info[session_id] = _expect(process, "worker_booted", self.boot_timeout)
                process.stdin.write(_bytes({
                    "type": "load",
                    "session_id": session_id,
                    "config": config or {},
                }))
                process.stdin.flush()
                loaded = _expect(process, "loaded", self.load_timeout)
                if loaded.get("message"):
                    raise WorkerStartError(loaded["message"])
                self.info[session_id].update(loaded)
            except (BrokenPipeError, OSError, WorkerStartError) as error:
                self.crash(session_id)
                raise WorkerStartError(str(error)) from error
            finally:
                log.close()
            self.on_loaded(session_id)
            threading.Thread(target=self._watch, args=(session_id, process), daemon=True).start()

    def submit(self, session_id, command, command_id, args=None):
        process = self._process(session_id)
        with self._locks[session_id]:
            try:
                process.stdin.write(_bytes({
                    "type": "command",
                    "command": command,
                    "command_id": command_id,
                    "args": args or {},
                }))
                process.stdin.flush()
            except (BrokenPipeError, OSError) as error:
                raise WorkerStartError(f"仿真进程已不可用：{error}") from error

    def query(self, session_id, name, timeout=3.0, args=None):
        """Ask the worker for read-only state; returns its data or raises."""
        process = self._process(session_id)
        request_id = f"q{self._request_seq}"
        with self._lock:
            self._request_seq += 1
            slot = {"event": threading.Event(), "data": None, "error": "查询超时"}
            self._pending[request_id] = slot
        try:
            with self._locks[session_id]:
                process.stdin.write(_bytes({"type": "query", "name": name, "request_id": request_id,
                                            "args": args or {}}))
                process.stdin.flush()
            if not slot["event"].wait(timeout):
                raise WorkerStartError("查询超时")
            if slot["data"] is None:
                raise WorkerStartError(slot["error"] or "查询失败")
            return slot["data"]
        finally:
            with self._lock:
                self._pending.pop(request_id, None)

    def alive(self, session_id):
        process = self._processes.get(session_id)
        return process is not None and process.poll() is None

    def keepalive(self, session_id, lease_id):
        process = self._process(session_id)
        with self._locks[session_id]:
            try:
                process.stdin.write(_bytes({"type": "keepalive", "lease_id": lease_id}))
                process.stdin.flush()
            except (BrokenPipeError, OSError):
                return False
        return True

    def crash(self, session_id):
        process = self._processes.pop(session_id, None)
        self._locks.pop(session_id, None)
        self.info.pop(session_id, None)
        if process is None:
            return
        try:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass
        _close(process.stdin)
        _close(process.stdout)

    def _process(self, session_id):
        process = self._processes.get(session_id)
        if process is None or process.poll() is not None:
            raise WorkerStartError("仿真进程未运行")
        return process

    def _watch(self, session_id, process):
        expected_close = False
        try:
            for raw in process.stdout:
                try:
                    message = decode(raw.decode("utf-8").strip())
                except (ProtocolError, UnicodeDecodeError):
                    continue
                if message.get("type") == "applied" and message.get("command_id"):
                    detail = message.get("detail") or {}
                    self.on_applied(session_id, message["command_id"], detail)
                    expected_close = detail.get("state") == "CLOSED"
                elif message.get("type") == "query_result":
                    slot = self._pending.get(message.get("request_id"))
                    if slot is not None:
                        slot["data"] = message.get("data")
                        slot["event"].set()
                elif message.get("type") == "query_failed":
                    slot = self._pending.get(message.get("request_id"))
                    if slot is not None:
                        slot["error"] = message.get("message") or "查询失败"
                        slot["event"].set()
                elif message.get("type") == "policy_progress":
                    self.on_policy_progress(session_id, message.get("detail") or {})
                elif message.get("type") == "policy_stopped":
                    self.on_policy_stopped(session_id, message.get("reason") or "stopped", message.get("detail") or {})
                elif message.get("type") == "manual_progress":
                    self.on_manual_progress(session_id, message.get("steps") or 0, message.get("detail") or {})
                elif message.get("type") == "manual_stopped":
                    self.on_manual_stopped(session_id, message.get("reason") or "stopped", message.get("steps") or 0, message.get("measured_m"))
        except (OSError, ValueError):
            pass
        code = process.wait()
        _close(process.stdin)
        _close(process.stdout)
        with self._lock:
            if self._processes.get(session_id) is not process:
                return
            self._processes.pop(session_id, None)
            self._locks.pop(session_id, None)
        if not (expected_close and code == 0):
            self.on_lost(session_id, f"仿真进程退出，退出码 {code}")
