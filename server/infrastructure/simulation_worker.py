"""Worker-side command loop.

Backend selection is explicit: `PRISM_BACKEND=robocasa` loads the real kitchen,
anything else keeps the state-only backend. The boot message states which one
is active and whether it loads physics, so a caller cannot mistake one for the
other.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

from server.infrastructure.manual_motion import ManualMotion
from server.infrastructure.worker_protocol import ProtocolError, decode, encode, validate_command


class StateOnlyBackend:
    name = "state_only"
    loads_physics = False

    def __init__(self):
        self.session_id = None
        self.episode = 1
        self.steps = 0

    def load(self, session_id, config=None):
        self.session_id = session_id
        return {"episode": self.episode, "steps": self.steps}

    def apply(self, command, args=None):
        if command == "step":
            self.steps += 1
            return {"state": "PAUSED", "step": self.steps, "episode": self.episode}
        if command == "jog":
            self.steps += 1
            return {"state": "PAUSED", "step": self.steps, "episode": self.episode, "jog": args or {}}
        if command == "reset":
            self.episode += 1
            self.steps = 0
            return {"state": "PAUSED", "step": 0, "episode": self.episode}
        if command == "resume":
            return {"state": "RUNNING"}
        if command == "pause":
            return {"state": "PAUSED"}
        if command == "stop":
            return {"state": "CLOSED"}
        raise ProtocolError("未知命令")


def build_backend(session_root, session_id):
    kind = os.environ.get("PRISM_BACKEND", "state_only").strip().lower()
    if kind == "robocasa":
        from server.runtime.adapters.robocasa_backend import RoboCasaBackend

        return RoboCasaBackend(Path(session_root) / session_id / "frames")
    return StateOnlyBackend()


def run(stdin=None, stdout=None, backend_factory=None, session_root="data/sessions"):
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    backend = None
    manual = {"motion": None}
    policy = {"motion": None}
    send_lock = __import__("threading").Lock()
    env_lock = __import__("threading").RLock()

    def send(message):
        with send_lock:
            stdout.write(encode(message))
            stdout.flush()

    send({
        "type": "worker_booted",
        "backend": os.environ.get("PRISM_BACKEND", "state_only"),
        "loads_physics": os.environ.get("PRISM_BACKEND", "state_only").strip().lower() == "robocasa",
    })
    def stop_policy():
        if policy["motion"] is not None:
            policy["motion"].stop()
            policy["motion"] = None

    try:
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = decode(line)
                kind = message.get("type")
                if kind == "load":
                    session_id = message.get("session_id")
                    if not isinstance(session_id, str) or not session_id:
                        raise ProtocolError("会话编号无效")
                    backend = (backend_factory or build_backend)(session_root, session_id)
                    state = backend.load(session_id, message.get("config"))
                    send({
                        "type": "loaded",
                        "session_id": session_id,
                        "backend": backend.name,
                        "loads_physics": backend.loads_physics,
                        "state": state,
                    })
                elif kind == "command":
                    if backend is None:
                        raise ProtocolError("环境尚未加载")
                    command, command_id = validate_command(message.get("command"), message.get("command_id"), message.get("args"))
                    args = message.get("args") or {}
                    if command in {"pause", "stop", "reset", "jog_start"}:
                        stop_policy()
                    if command == "jog_start":
                        if manual["motion"] is not None:
                            raise ProtocolError("连续点动已在运行")
                        keepalive_ms = args.get("keepalive_ms", 1500)
                        if type(keepalive_ms) is not int or not 300 <= keepalive_ms <= 10000:
                            raise ProtocolError("keepalive_ms 必须在 300 到 10000 之间")
                        motion_args = {key: value for key, value in args.items() if key in {"delta_m", "delta_rot", "gripper"}}

                        def emit(reason, steps, _cid=command_id):
                            manual["motion"] = None
                            measured = None
                            if hasattr(backend, "end_manual"):
                                try:
                                    measured = backend.end_manual()
                                except Exception:
                                    measured = None
                            send({"type": "manual_stopped", "command_id": _cid, "reason": reason,
                                  "steps": steps, "measured_m": measured})

                        def progress(steps, result, _cid=command_id):
                            send({"type": "manual_progress", "command_id": _cid, "steps": steps,
                                  "detail": result})

                        if hasattr(backend, "begin_manual"):
                            backend.begin_manual()
                        motion = ManualMotion(backend, motion_args, _period(backend), keepalive_ms, emit,
                                              env_lock, on_progress=progress)
                        manual["motion"] = motion
                        motion.start()
                        send({"type": "applied", "command_id": command_id, "detail": {"state": "RUNNING", "mode": "manual"}, "state": "RUNNING"})
                        continue
                    if command == "jog_stop":
                        if manual["motion"] is not None:
                            manual["motion"].stop("operator_stop")
                        send({"type": "applied", "command_id": command_id, "detail": {"state": "PAUSED"}, "state": "PAUSED"})
                        continue
                    if command in {"pause", "stop", "reset"} and manual["motion"] is not None:
                        manual["motion"].stop(f"command_{command}")
                    with env_lock:
                        result = backend.apply(command, args)
                    send({"type": "applied", "command_id": command_id, "detail": result,
                          "state": result.get("state")})
                    if command == "stop":
                        return 0
                    if command == "resume":
                        from server.infrastructure.policy_motion import PolicyMotion
                        motion = PolicyMotion(
                            backend, _period(backend), env_lock,
                            lambda detail: send({"type": "policy_progress", "detail": detail}),
                            lambda reason, detail: send({"type": "policy_stopped", "reason": reason, "detail": detail}),
                        )
                        policy["motion"] = motion
                        motion.start()
                elif kind == "query":
                    if backend is None:
                        raise ProtocolError("环境尚未加载")
                    name = message.get("name")
                    request_id = message.get("request_id")
                    if not isinstance(request_id, str) or not request_id:
                        raise ProtocolError("request_id 无效")
                    if name != "scene" or not hasattr(backend, "scene_state"):
                        send({"type": "query_failed", "request_id": request_id, "message": "不支持的查询"})
                        continue
                    with env_lock:
                        data = backend.scene_state(message.get("args") or {})
                    send({"type": "query_result", "request_id": request_id, "data": data})
                elif kind == "keepalive":
                    if manual["motion"] is not None:
                        manual["motion"].keepalive()
                elif kind == "shutdown":
                    stop_policy()
                    if manual["motion"] is not None:
                        manual["motion"].stop("shutdown")
                    send({"type": "shutdown_ack"})
                    return 0
                else:
                    raise ProtocolError("未知消息类型")
            except ProtocolError as error:
                send({"type": "error", "message": str(error)})
            except Exception as error:  # backend failures must not kill the loop silently
                send({"type": "error", "message": f"{type(error).__name__}: {error}"})
                if kind == "load":
                    return 1
        return 0
    finally:
        stop_policy()
        if manual["motion"] is not None:
            manual["motion"].stop("input_closed")
        if backend is not None and hasattr(backend, "close"):
            try:
                with env_lock:
                    backend.close()
            except Exception as error:
                print(f"Environment cleanup failed: {type(error).__name__}: {error}", file=sys.stderr)


def _period(backend):
    config = getattr(backend, "config", None) or {}
    freq = config.get("control_freq", 20)
    try:
        return 1.0 / float(freq)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.05


def main():
    # Library prints (robosuite/robocasa) must never share the protocol pipe.
    # Keep the real stdout for messages and send everything else to stderr.
    protocol_out = sys.stdout
    sys.stdout = sys.stderr
    raise SystemExit(run(stdout=protocol_out, session_root=os.environ.get("PRISM_SESSION_ROOT", "data/sessions")))


if __name__ == "__main__":
    main()
