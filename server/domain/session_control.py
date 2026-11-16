"""In-memory session command state machine.

Replaying the emitted events rebuilds command results. It must not execute
commands again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


from server.domain.control_lease import ControlLease, LeaseError

ACTIVE_STATES = {"QUEUED", "LOADING", "READY", "RUNNING", "PAUSED", "STOPPING"}
TERMINAL_STATES = {"CLOSED", "FAILED", "INTERRUPTED"}
COMMAND_STATES = {
    "resume": {"READY", "PAUSED"},
    "pause": {"RUNNING"},
    "step": {"READY", "PAUSED"},
    "jog": {"READY", "PAUSED"},
    "edit": {"READY", "PAUSED"},
    "jog_start": {"READY", "PAUSED"},
    "jog_stop": {"RUNNING", "PAUSED"},
    "reset": {"READY", "PAUSED"},
    "stop": set(ACTIVE_STATES),
}
APPLIED_STATE = {
    "resume": "RUNNING",
    "pause": "PAUSED",
    "step": "PAUSED",
    "jog": "PAUSED",
    "edit": "PAUSED",
    "jog_start": "RUNNING",
    "jog_stop": "PAUSED",
    "reset": "PAUSED",
    "stop": "CLOSED",
}
LEASE_COMMANDS = {"jog_start", "jog_stop"}


class ControlError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class CommandEnvelope:
    command_id: str
    session_id: str
    fencing_token: str
    command: str
    issued_at: datetime
    expires_at: datetime
    expected_state_version: int | None = None
    args: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CommandAck:
    command_id: str
    acknowledgement: str
    state: str
    state_version: int
    event_seq: int
    code: str | None = None
    message: str = ""

    def to_dict(self):
        return {
            "command_id": self.command_id,
            "acknowledgement": self.acknowledgement,
            "state": self.state,
            "state_version": self.state_version,
            "event_seq": self.event_seq,
            "code": self.code,
            "message": self.message,
        }


def _aware(value, label):
    if not isinstance(value, datetime) or value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ControlError("invalid_timestamp", f"{label} 必须是带时区的时间")


class SessionControl:
    def __init__(self, session_id, fencing_token, clock=None):
        self.session_id = session_id
        self.fencing_token = fencing_token
        self.lease = ControlLease(clock=clock)
        self.mode = None
        self.state = "QUEUED"
        self.state_version = 0
        self.episode = 1
        self.applied_steps = 0
        self.events = []
        self.commands = {}
        self.inflight_id = None
        self._seq = 0

    @classmethod
    def from_events(cls, events):
        if not events or events[0].get("type") != "session_created":
            raise ControlError("invalid_log", "会话日志缺少创建事件")
        session = cls(events[0]["session_id"], events[0]["fencing_token"])
        session.events = []
        session._seq = 0
        for event in events:
            session._fold(event)
        return session

    def create_event(self):
        if self.events:
            raise ControlError("invalid_state", "会话已创建")
        return self._emit("session_created", fencing_token=self.fencing_token)

    def mark_loading(self):
        self._require_state("QUEUED", "session_not_queued", "只有新会话可以进入加载")
        self.state = "LOADING"
        return self._emit("worker_loading")

    def mark_ready(self):
        self._require_state("LOADING", "session_not_loading", "只有加载中的会话可以就绪")
        self.state = "READY"
        self.state_version += 1
        return self._emit("worker_ready")

    def record_safety(self, message, code="safety_event"):
        if not isinstance(message, str) or not message:
            raise ControlError("invalid_event", "安全事件需要明确消息")
        return self._emit("safety_event", code=code, message=message)

    def submit(self, envelope, now):
        self._validate_shape(envelope, now)
        stored = self.commands.get(envelope.command_id)
        if stored is not None:
            # Same command_id is always the same intent, even if the payload
            # differs. Replay the stored result and never execute again.
            return stored, None, False
        rejection = self._reject_if_needed(envelope, now)
        if rejection is not None:
            return rejection, self.events[-1], True
        if envelope.command == "stop" and self.inflight_id:
            self._supersede(self.inflight_id, "被停止命令取代，原命令未执行")
        if envelope.command == "stop":
            self.state = "STOPPING"
            self.state_version += 1
        message = "停止已受理，将在当前物理步结束后生效" if envelope.command == "stop" else "已受理，等待仿真进程执行"
        event = self._emit(
            "command_accepted",
            command_id=envelope.command_id,
            command=envelope.command,
            acknowledgement="accepted",
            message=message,
        )
        self.inflight_id = envelope.command_id
        return self._remember(event), event, True

    def mark_applied(self, command_id):
        if self.inflight_id != command_id:
            raise ControlError("command_not_inflight", "没有可确认的已受理命令")
        command = self._command_name(command_id)
        self.state = APPLIED_STATE[command]
        self.state_version += 1
        if command == "jog_start":
            self.mode = "manual"
        elif command in {"jog_stop", "pause", "stop", "reset"}:
            self.mode = None
        if command == "step":
            self.applied_steps += 1
        if command == "reset":
            self.episode += 1
        self.inflight_id = None
        event = self._emit(
            "command_applied",
            command_id=command_id,
            command=command,
            acknowledgement="applied",
            message="仿真进程已执行",
        )
        return self._remember(event), event

    def mark_policy_stopped(self, reason):
        if self.state != "RUNNING" or self.mode == "manual":
            return None
        self.state = "PAUSED"
        self.state_version += 1
        return self._emit("policy_stopped", reason=reason, message="策略步进已由仿真进程停止")

    def mark_manual_stopped(self, reason, steps=0):
        """The worker stopped manual motion on its own (lease, limit, stop)."""
        if self.state not in {"RUNNING", "PAUSED"}:
            return None, None
        self.mode = None
        self.state = "PAUSED"
        self.state_version += 1
        event = self._emit(
            "manual_stopped",
            reason=reason,
            steps=int(steps),
            message="连续点动已由仿真进程停止",
        )
        return event, event

    def mark_worker_lost(self, reason):
        if self.state in TERMINAL_STATES:
            return None, None
        command_id = self.inflight_id
        self.state = "INTERRUPTED"
        self.state_version += 1
        self.inflight_id = None
        self.mode = None
        self.lease.release_quietly()
        event = self._emit(
            "worker_lost",
            command_id=command_id,
            acknowledgement="rejected" if command_id else None,
            code="worker_lost",
            message=reason,
        )
        return (self._remember(event) if command_id else None), event

    def acquire_lease(self, holder, ttl_ms):
        try:
            return self.lease.acquire(holder, ttl_ms)
        except LeaseError as error:
            raise ControlError(error.code, error.message) from error

    def renew_lease(self, lease_id, ttl_ms):
        try:
            return self.lease.renew(lease_id, ttl_ms)
        except LeaseError as error:
            raise ControlError(error.code, error.message) from error

    def release_lease(self, lease_id):
        try:
            return self.lease.release(lease_id)
        except LeaseError as error:
            raise ControlError(error.code, error.message) from error

    def events_after(self, seq):
        if type(seq) is not int or seq < 0:
            raise ControlError("invalid_cursor", "after 必须是非负整数")
        return [dict(event) for event in self.events if event["seq"] > seq]

    def snapshot(self):
        return {
            "id": self.session_id,
            "state": self.state,
            "state_version": self.state_version,
            "episode": self.episode,
            "applied_steps": self.applied_steps,
            "inflight_command_id": self.inflight_id,
            "fencing_token": self.fencing_token,
            "last_event_seq": self._seq,
            "mode": self.mode,
            "lease": self.lease.describe(),
        }

    def _validate_shape(self, envelope, now):
        if envelope.session_id != self.session_id or not _token(envelope.command_id):
            raise ControlError("invalid_identity", "会话编号或 command_id 无效")
        if envelope.command not in COMMAND_STATES:
            raise ControlError("invalid_command", "command 必须是 " + " / ".join(sorted(COMMAND_STATES)))
        _aware(envelope.issued_at, "issued_at")
        _aware(envelope.expires_at, "expires_at")
        _aware(now, "now")
        version = envelope.expected_state_version
        if version is not None and type(version) is not int:
            raise ControlError("invalid_state_version", "expected_state_version 必须是整数")
        if not isinstance(envelope.args, dict):
            raise ControlError("invalid_args", "args 必须是对象")
        if envelope.command not in {"jog", "jog_start", "jog_stop", "edit"} and envelope.args:
            raise ControlError("invalid_args", "只有 jog / edit 命令接受参数")
        if envelope.command in LEASE_COMMANDS:
            lease_id = envelope.args.get("lease_id")
            if not isinstance(lease_id, str) or not lease_id:
                raise ControlError("invalid_args", "jog_start / jog_stop 需要 lease_id")
            try:
                self.lease.check(lease_id)
            except LeaseError as error:
                raise ControlError(error.code, error.message) from error

    def _reject_if_needed(self, envelope, now):
        code = message = None
        if envelope.fencing_token != self.fencing_token:
            code, message = "lease_mismatch", "控制租约已失效"
        elif now > envelope.expires_at:
            code, message = "expired", "命令已过期，未执行"
        elif self.state in TERMINAL_STATES:
            code, message = "session_terminal", "会话已结束，不能执行新命令"
        elif envelope.expected_state_version not in (None, self.state_version):
            code, message = "state_version_conflict", "状态版本已变化，请刷新后重试"
        elif self.inflight_id and envelope.command != "stop":
            code, message = "command_pending", "上一条命令尚未执行完成"
        elif self.state not in COMMAND_STATES[envelope.command]:
            code, message = "state_conflict", "当前状态不允许该命令"
        if code is None:
            return None
        event = self._emit(
            "command_rejected",
            command_id=envelope.command_id,
            command=envelope.command,
            acknowledgement="rejected",
            code=code,
            message=message,
        )
        return self._remember(event)

    def _supersede(self, command_id, message):
        event = self._emit(
            "command_superseded",
            command_id=command_id,
            command=self._command_name(command_id),
            acknowledgement="superseded",
            code="superseded",
            message=message,
        )
        self._remember(event)
        self.inflight_id = None

    def _remember(self, event):
        ack = CommandAck(
            command_id=event["command_id"],
            acknowledgement=event["acknowledgement"],
            state=event["state"],
            state_version=event["state_version"],
            event_seq=event["seq"],
            code=event.get("code"),
            message=event.get("message") or "",
        )
        self.commands[event["command_id"]] = ack
        return ack

    def _emit(self, event_type, **fields):
        self._seq += 1
        event = {
            "seq": self._seq,
            "type": event_type,
            "session_id": self.session_id,
            "state": self.state,
            "state_version": self.state_version,
            "episode": self.episode,
        }
        event.update(fields)
        self.events.append(event)
        return event

    def _fold(self, event):
        self._seq = event["seq"]
        self.state = event["state"]
        self.state_version = event["state_version"]
        self.episode = event["episode"]
        self.events.append(dict(event))
        if event.get("type") == "command_applied" and event.get("command") == "step":
            self.applied_steps += 1
        command_id = event.get("command_id")
        if command_id and event.get("acknowledgement"):
            self._remember(event)
        if event.get("type") == "command_accepted":
            self.inflight_id = command_id
        elif event.get("type") in {"command_applied", "command_superseded"} and command_id == self.inflight_id:
            self.inflight_id = None
        elif event.get("type") == "command_rejected" and command_id == self.inflight_id:
            self.inflight_id = None
        elif event.get("type") == "worker_lost":
            self.inflight_id = None

    def _command_name(self, command_id):
        for event in reversed(self.events):
            if event.get("command_id") == command_id and event.get("command"):
                return event["command"]
        raise ControlError("invalid_log", "找不到命令名称")

    def _require_state(self, expected, code, message):
        if self.state != expected:
            raise ControlError(code, message)


def _token(value):
    return isinstance(value, str) and 1 <= len(value) <= 100 and value.replace("-", "").replace("_", "").isalnum()
