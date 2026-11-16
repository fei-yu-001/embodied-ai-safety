"""Persistent local runs with one simulation worker at a time."""

from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import uuid

from .storage import replace_snapshot
from .scene_runtime import validate_camera


ROOT = Path(__file__).resolve().parents[2]
ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
ACTIVE = {"queued", "loading", "ready", "running", "paused", "cancelling"}
TERMINAL = {"completed", "cancelled", "failed", "interrupted"}
ARTIFACT_SUFFIXES = {".mp4", ".png", ".jpg", ".json", ".jsonl", ".md", ".txt", ".xml", ".npz", ".log"}


def read_json(path, fallback=None):
    try:
        value = Path(path).read_text(encoding="utf-8-sig")
        data = json.loads(value)
        # Persisted job/config/progress/summary records must be objects. A
        # truncated or malformed record must not break the whole history page.
        return fallback if isinstance(fallback, dict) and not isinstance(data, dict) else data
    except (json.JSONDecodeError, OSError, UnicodeError):
        return fallback


def write_json(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    replace_snapshot(temporary, path)


def _status(record):
    value = record.get("status")
    return value if isinstance(value, str) else None


def _inside_file(folder, path):
    """Do not read or advertise an artifact linked outside its run directory."""
    try:
        return path.resolve().is_relative_to(folder) and path.is_file()
    except (OSError, RuntimeError):
        return False


def _record(folder, name):
    path = folder / name
    return read_json(path, {}) if _inside_file(folder, path) else {}


class RunBusyError(Exception):
    pass


class RunManager:
    def __init__(self, runs_dir=None, legacy_dir=None, python=None, command_factory=None, *, cancel_timeout=90):
        self.runs_dir = Path(runs_dir or ROOT / "experiments/runs").resolve()
        self.legacy_dir = Path(legacy_dir or ROOT / "experiments/scripts/output").resolve()
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.python = python or sys.executable
        self.command_factory = command_factory
        self.cancel_timeout = cancel_timeout
        self.lock = threading.RLock()
        self.active_id = None
        self.process = None
        self.closing = False

    def recover_interrupted(self):
        """Recover stale records only after the server has bound its port.

        Construction is deliberately read-only for existing jobs: a second
        server that fails to bind must not mark the first server's job stopped.
        The caller must own the service before invoking this method.
        """
        recovered = []
        with self.lock:
            for entry in self.runs_dir.iterdir():
                if entry.name == self.active_id:
                    continue
                try:
                    folder = self.resolve(entry.name)
                except KeyError:
                    continue
                job = _record(folder, "job.json")
                if _status(job) not in ACTIVE:
                    continue
                progress = _record(folder, "progress.json")
                progress_status = _status(progress)
                job["status"] = progress_status if progress_status in TERMINAL else "interrupted"
                job["message"] = "服务重启；旧实验结果已保留。"
                job["finished_at"] = datetime.now().astimezone().isoformat()
                if job["status"] == "failed" and progress.get("error"):
                    job["error"] = progress["error"]
                write_json(folder / "job.json", job)
                recovered.append(entry.name)
        return recovered

    def resolve(self, run_id):
        if not isinstance(run_id, str) or not ID_PATTERN.fullmatch(run_id):
            raise KeyError("无效实验编号")
        if run_id.startswith("legacy_"):
            base, name = self.legacy_dir, run_id[len("legacy_"):]
        else:
            base, name = self.runs_dir, run_id
        try:
            folder = (base / name).resolve()
            if not name or folder.parent != base or not folder.is_dir():
                raise KeyError("实验不存在")
            if not any(_inside_file(folder, folder / name) for name in ("job.json", "summary.json")):
                raise KeyError("实验不存在")
        except (OSError, RuntimeError) as error:
            raise KeyError("实验不存在") from error
        return folder

    def list_runs(self):
        result = []
        for base, prefix in ((self.runs_dir, ""), (self.legacy_dir, "legacy_")):
            if not base.exists():
                continue
            for folder in base.iterdir():
                try:
                    result.append(self.describe(prefix + folder.name))
                except KeyError:
                    continue
        return sorted(result, key=lambda item: item["created_at"], reverse=True)

    def describe(self, run_id, detail=False):
        with self.lock:
            folder = self.resolve(run_id)
            job = _record(folder, "job.json")
            progress = _record(folder, "progress.json")
            summary = _record(folder, "summary.json")
            config = _record(folder, "config.json")
            live = self.active_id == run_id and self.process is not None and self.process.poll() is None
        legacy = run_id.startswith("legacy_")
        status = _status(job) or ("completed" if summary else "interrupted")
        if status in ACTIVE and _status(progress) in TERMINAL:
            status = _status(progress)
        if status in ACTIVE and status != "cancelling" and _status(progress) in ACTIVE:
            status = _status(progress)
        control = _record(folder, "control.request.json")
        completed = progress.get("step", summary.get("steps_completed", 0))
        total = config.get("steps", summary.get("steps_requested", 100))
        artifacts = []
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in ARTIFACT_SUFFIXES and _inside_file(folder, path):
                artifacts.append({"name": path.name, "bytes": path.stat().st_size,
                                  "url": f"/api/runs/{run_id}/artifacts/{path.name}"})
        created_at = job.get("created_at") or summary.get("created_at")
        if not isinstance(created_at, str):
            created_at = datetime.fromtimestamp(folder.stat().st_mtime).astimezone().isoformat()
        result = {
            "id": run_id, "legacy": legacy, "status": status,
            "created_at": created_at,
            "steps_completed": completed, "steps_requested": total,
            "phase": progress.get("phase", job.get("message", "历史补丁实验" if legacy else "等待工作进程")),
            "elapsed_seconds": progress.get("elapsed_seconds", summary.get("elapsed_seconds", 0)),
            "config": config, "summary": summary, "latest": progress.get("latest"),
            "artifacts": artifacts, "error": job.get("error") or progress.get("error"),
            "live": live,
            "episode": progress.get("episode", summary.get("episodes", 1)),
            "episode_step": progress.get("episode_step", summary.get("episode_step", completed)),
            "simulation_seconds": progress.get("simulation_seconds", summary.get("simulation_seconds", completed / summary.get("fps", 20))),
            "control_freq": progress.get("control_freq", summary.get("fps", 20)),
            "frame_index": progress.get("frame_index", summary.get("frames_observed", 0)),
            "last_frame_at": progress.get("last_frame_at"),
            "episode_done": bool(progress.get("episode_done", False)),
            "scene": progress.get("scene") or summary.get("scene"),
            "camera": progress.get("camera"),
            "contact_count": progress.get("contact_count", 0),
            "command_pending": live and control.get("seq", 0) > progress.get("control_seq", 0),
        }
        if detail:
            result["metrics"] = self.read_jsonl(folder / "metrics.jsonl") if _inside_file(folder, folder / "metrics.jsonl") else []
            result["events"] = self.read_jsonl(folder / "events.jsonl") if _inside_file(folder, folder / "events.jsonl") else []
            log = folder / "worker.log"
            if _inside_file(folder, log):
                with log.open("rb") as handle:
                    handle.seek(max(0, log.stat().st_size - 24000))
                    result["log"] = handle.read().decode("utf-8", errors="replace")
            else:
                result["log"] = "历史实验未保存工作进程日志。" if legacy else "工作进程正在启动…"
        return result

    @staticmethod
    def read_jsonl(path):
        try:
            lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        except OSError:
            return []
        result = []
        for line in lines:
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    result.append(value)
            except json.JSONDecodeError:
                pass  # The worker may be halfway through appending the last row.
        return result

    def start(self, config):
        with self.lock:
            if self.closing or self.active_id is not None:
                raise RunBusyError("已有实验正在运行或服务正在关闭，请完成或停止后再启动。")
            run_id = f"run_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
            folder = self.runs_dir / run_id
            folder.mkdir()
            job = {"id": run_id, "status": "loading", "created_at": datetime.now().astimezone().isoformat()}
            write_json(folder / "config.json", config.to_dict())
            write_json(folder / "job.json", job)
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            try:
                command = self.command_factory(folder) if self.command_factory else [
                    self.python, "-X", "utf8", "-u", str(ROOT / "experiments/scripts/run_system_experiment.py"),
                    "--run-dir", str(folder),
                ]
                with (folder / "worker.log").open("wb") as log:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                               stdin=subprocess.DEVNULL, creationflags=flags)
            except (OSError, ValueError, TypeError) as error:
                job.update(status="failed", error=str(error), finished_at=datetime.now().astimezone().isoformat())
                write_json(folder / "job.json", job)
                raise
            self.active_id, self.process = run_id, process
            threading.Thread(target=self._watch, args=(run_id, folder, process), daemon=True).start()
            return self.describe(run_id)

    def _watch(self, run_id, folder, process):
        returncode = process.wait()
        with self.lock:
            try:
                job = _record(folder, "job.json")
                progress = _record(folder, "progress.json")
                status = _status(progress)
                if status not in TERMINAL:
                    status = "cancelled" if (folder / "cancel.request").exists() else "failed"
                if returncode != 0 and status == "completed":
                    status = "failed"
                job.update(status=status, returncode=returncode, finished_at=datetime.now().astimezone().isoformat())
                if status == "failed":
                    job["error"] = progress.get("error") or f"工作进程退出，退出码 {returncode}，请查看日志。"
                write_json(folder / "job.json", job)
            finally:
                # A persistence error must not permanently occupy the worker slot.
                if self.active_id == run_id:
                    self.active_id, self.process = None, None

    def control(self, run_id, payload):
        """Send one acknowledged command to the resident simulation worker."""
        if not isinstance(payload, dict) or set(payload) != {"command"}:
            raise ValueError("控制请求必须只包含 command")
        command = payload["command"]
        allowed_states = {"resume": {"ready", "paused"}, "pause": {"running"},
                          "step": {"ready", "paused"}, "reset": {"ready", "paused"}}
        if not isinstance(command, str) or command not in allowed_states:
            raise ValueError("command 必须是 resume / pause / step / reset")
        with self.lock:
            folder = self.resolve(run_id)
            detail = self.describe(run_id)
            if not detail["live"] or detail["config"].get("run_mode", "batch") != "session":
                raise RunBusyError("该实验不是已加载的交互仿真会话")
            if detail["command_pending"]:
                raise RunBusyError("上一条控制命令正在执行")
            if detail["status"] not in allowed_states[command]:
                raise RunBusyError("当前环境状态不支持此操作；请等待加载完成，重置前先暂停")
            if detail["episode_done"] and command in {"resume", "step"}:
                raise RunBusyError("本轮环境已结束，请先重置")
            progress = _record(folder, "progress.json")
            seq = progress.get("control_seq", 0) + 1
            write_json(folder / "control.request.json", {"seq": seq, "command": command})
            return self.describe(run_id)

    def camera(self, run_id, payload):
        parameters = validate_camera(payload)
        with self.lock:
            folder = self.resolve(run_id)
            detail = self.describe(run_id)
            if not detail['live'] or detail['status'] not in {'ready', 'running', 'paused'}:
                raise RunBusyError('相机仅在环境就绪、运行或暂停时可控制')
            previous = _record(folder, 'camera.request.json')
            seq = max(previous.get('seq', 0), (detail.get('camera') or {}).get('seq', 0)) + 1
            write_json(folder / 'camera.request.json', {'seq': seq, 'parameters': parameters})
            return self.describe(run_id)

    def cancel(self, run_id):
        with self.lock:
            folder = self.resolve(run_id)
            if self.active_id != run_id or self.process is None or self.process.poll() is not None:
                raise RunBusyError("该实验已结束，无法停止。")
            # Retried HTTP requests share one cancellation timer.
            if not (folder / "cancel.request").exists():
                (folder / "cancel.request").touch()
                job = _record(folder, "job.json")
                job["status"] = "cancelling"
                write_json(folder / "job.json", job)
                threading.Thread(target=self._cancel_timeout, args=(self.process,), daemon=True).start()
            return self.describe(run_id)

    def _cancel_timeout(self, process):
        try:
            process.wait(timeout=self.cancel_timeout)
        except subprocess.TimeoutExpired:
            # Initialization may be stuck. Keep partial artifacts and bound stop
            # latency even if the worker ignores a graceful termination signal.
            try:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                # The watcher may have reaped it between poll and terminate.
                pass

    def shutdown(self):
        with self.lock:
            self.closing = True
            if self.active_id is not None and self.process is not None and self.process.poll() is None:
                self.cancel(self.active_id)
