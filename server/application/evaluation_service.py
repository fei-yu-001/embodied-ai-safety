"""Evaluation jobs: one run at a time, state read from disk."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import uuid

from server.domain.evaluation import SpecError, build_matrix, normalize_spec

ROOT = Path(__file__).resolve().parents[2]


class EvaluationBusy(Exception):
    pass


class EvaluationService:
    def __init__(self, root, python=None, command_factory=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.python = python or sys.executable
        self.command_factory = command_factory
        self._lock = threading.RLock()
        self._process = None
        self._active_id = None

    def create(self, raw_spec):
        spec = normalize_spec(raw_spec)
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise EvaluationBusy("已有评测在运行，请等待完成")
            run_id = f"eval_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
            run_dir = self.root / run_id
            run_dir.mkdir(parents=True, exist_ok=False)
            (run_dir / "spec.json").write_text(
                json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            status = {
                "state": "running",
                "started_at": datetime.now().astimezone().isoformat(),
                "matrix": build_matrix(spec),
                "completed": 0,
                "total": len(build_matrix(spec)) * len(spec["seeds"]),
                "current": None,
            }
            (run_dir / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            command = self.command_factory(run_dir) if self.command_factory else [
                self.python, "-X", "utf8", "-u", "-m", "server.runtime.evaluation_runner",
                "--run-dir", str(run_dir),
            ]
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            log = (run_dir / "runner.log").open("wb")
            try:
                self._process = subprocess.Popen(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                                                 stdin=subprocess.DEVNULL, creationflags=flags)
            finally:
                log.close()
            self._active_id = run_id
            return self.get(run_id)

    def get(self, run_id):
        if not run_id.startswith("eval_") or "/" in run_id or "\\" in run_id:
            raise KeyError(run_id)
        run_dir = self.root / run_id
        if not run_dir.is_dir():
            raise KeyError(run_id)
        status = _read_json(run_dir / "status.json") or {"state": "unknown"}
        summary = _read_json(run_dir / "summary.json")
        spec = _read_json(run_dir / "spec.json") or {}
        result = {"id": run_id, "spec": spec, "status": status}
        if summary:
            result["summary"] = summary
            result["gate"] = summary.get("gate")
            result["groups"] = summary.get("groups", [])
        log = run_dir / "runner.log"
        if log.is_file():
            with log.open("rb") as handle:
                handle.seek(max(0, log.stat().st_size - 8000))
                result["log"] = handle.read().decode("utf-8", errors="replace")
        return result

    def list(self):
        items = []
        for entry in sorted(self.root.iterdir(), reverse=True):
            if entry.is_dir() and entry.name.startswith("eval_"):
                try:
                    items.append(self.get(entry.name))
                except KeyError:
                    continue
        return items

    def close(self):
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._process.kill()


def _read_json(path):
    try:
        value = path.read_text(encoding="utf-8")
        data = json.loads(value)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


__all__ = ["EvaluationService", "EvaluationBusy", "SpecError"]
