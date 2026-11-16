"""Read-only adapter for historical experiment results.

新结果位于 experiments/runs/<run>，历史结果位于 experiments/scripts/output/<name>。
本模块只读取目录与已落盘的 JSON/产物，不创建目录、不写文件、不修补历史数据，
也不改写历史结论；损坏或字段缺失的记录被跳过，不让单条记录拖垮整个列表。
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from server.domain.legacy_contract import REQUIRED_VIEW_KEYS, map_legacy_status

ALLOWED_SUFFIXES = frozenset(
    {".mp4", ".png", ".jpg", ".json", ".jsonl", ".md", ".txt", ".xml", ".npz", ".log"}
)
MARKER_FILES = ("job.json", "summary.json")
LEGACY_PREFIX = "legacy_"
MAX_ARTIFACTS = 200
UNKNOWN_STATUS = "UNKNOWN"


def list_legacy_runs(runs_dir, output_dir):
    """Return every readable record under both roots, newest first."""
    records = []
    records.extend(_scan(runs_dir, legacy=False))
    records.extend(_scan(output_dir, legacy=True))
    records.sort(key=lambda record: (record["created_at"] or "", record["id"]), reverse=True)
    return records


def get_legacy_run(run_id, runs_dir, output_dir):
    """Return one record, or None when the id is unknown, unsafe, or damaged."""
    if not isinstance(run_id, str):
        return None
    legacy = run_id.startswith(LEGACY_PREFIX)
    name = run_id[len(LEGACY_PREFIX) :] if legacy else run_id
    base = _safe_root(output_dir if legacy else runs_dir)
    if base is None:
        return None
    return _build_record(base, name, legacy)


def get_artifact_path(run_id, name, runs_dir, output_dir):
    """Return the absolute path of one stored artifact, or None when it is not published."""
    if not isinstance(run_id, str) or not _is_plain_name(name):
        return None
    if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
        return None
    legacy = run_id.startswith(LEGACY_PREFIX)
    record_name = run_id[len(LEGACY_PREFIX) :] if legacy else run_id
    base = _safe_root(output_dir if legacy else runs_dir)
    if base is None:
        return None
    directory = _contained_dir(base, record_name)
    if directory is None or not _has_marker(directory):
        return None
    candidate = directory / name
    try:
        if candidate.is_symlink():
            return None
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if directory not in resolved.parents or not resolved.is_file():
        return None
    return str(resolved)


def _scan(root, legacy):
    base = _safe_root(root)
    if base is None:
        return
    try:
        names = sorted(item.name for item in base.iterdir())
    except OSError:
        return
    for name in names:
        record = _build_record(base, name, legacy)
        if record is not None:
            yield record


def _build_record(base, name, legacy):
    directory = _contained_dir(base, name)
    if directory is None:
        return None
    if not _has_marker(directory):
        return None
    documents = _load_documents(directory)
    if documents is None:
        return None
    try:
        return _view(directory, name, legacy, *documents)
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _view(directory, name, legacy, job, progress, summary_doc):
    record_id = f"{LEGACY_PREFIX}{name}" if legacy else name
    legacy_status = _pick(job.get("status"), summary_doc.get("status"), progress.get("status"))
    record = {
        "id": record_id,
        "legacy": legacy,
        "writable": False,
        "status": map_legacy_status(legacy_status),
        "legacy_status": legacy_status,
        "created_at": _pick(job.get("created_at"), summary_doc.get("created_at"))
        or _fallback_created_at(directory),
        "steps_completed": _pick_int(
            summary_doc.get("steps_completed"), progress.get("step"), job.get("steps_completed")
        ),
        "steps_requested": _pick_int(
            summary_doc.get("steps_requested"), progress.get("total"), job.get("steps_requested")
        ),
        "phase": _pick(progress.get("phase")),
        "error": _pick(job.get("error"), job.get("message"), progress.get("error")),
        "summary": _summary_block(job, progress, summary_doc),
        "artifacts": _artifacts(directory, record_id),
    }
    missing = [key for key in REQUIRED_VIEW_KEYS if key not in record]
    if missing:
        raise KeyError(",".join(missing))
    return record


def _summary_block(job, progress, summary_doc):
    return {
        "experiment": _pick(summary_doc.get("experiment"), job.get("experiment")),
        "env_id": _pick(summary_doc.get("env_id"), job.get("env_id")),
        "policy": _pick(summary_doc.get("policy"), job.get("policy")),
        "seed": _pick_int(summary_doc.get("seed"), job.get("seed")),
        "stop_reason": _pick(summary_doc.get("stop_reason")),
        "success_observed": _pick_bool(summary_doc.get("success_observed")),
        "elapsed_seconds": _pick_float(summary_doc.get("elapsed_seconds")),
    }


def _artifacts(directory, record_id):
    items = []
    try:
        entries = sorted(directory.iterdir(), key=lambda item: item.name)
    except OSError:
        return items
    for entry in entries:
        if len(items) >= MAX_ARTIFACTS:
            break
        try:
            if entry.is_symlink() or not entry.is_file():
                continue
            if entry.suffix.lower() not in ALLOWED_SUFFIXES:
                continue
            size = entry.stat().st_size
        except OSError:
            continue
        items.append({"name": entry.name, "bytes": size, "url": _artifact_url(record_id, entry.name)})
    return items


def _artifact_url(record_id, name):
    return f"/api/v1/legacy-runs/{quote(record_id, safe='')}/artifacts/{quote(name, safe='')}"


def _safe_root(path):
    if path is None:
        return None
    try:
        root = Path(path)
        if not root.is_dir():
            return None
        return root.resolve(strict=True)
    except (OSError, ValueError, RuntimeError):
        return None


def _contained_dir(base, name):
    if not _is_plain_name(name):
        return None
    candidate = base / name
    try:
        if candidate.is_symlink():
            return None
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if resolved == base or base not in resolved.parents:
        return None
    if not resolved.is_dir():
        return None
    return resolved


def _is_plain_name(name):
    if not isinstance(name, str) or not name:
        return False
    if name in {".", ".."}:
        return False
    if "\x00" in name:
        return False
    if name.startswith("~"):
        return False
    for separator in {"/", "\\", os.sep, os.altsep or os.sep, ":"}:
        if separator and separator in name:
            return False
    return Path(name).name == name


def _has_marker(directory):
    for marker in MARKER_FILES:
        try:
            if (directory / marker).is_file():
                return True
        except OSError:
            continue
    return False


def _load_documents(directory):
    job = _read_json(directory / "job.json")
    progress = _read_json(directory / "progress.json")
    summary_doc = _read_json(directory / "summary.json")
    if not (job or progress or summary_doc):
        return None
    return job, progress, summary_doc


def _read_json(path):
    try:
        if not path.is_file() or path.is_symlink():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _fallback_created_at(directory):
    try:
        stamp = directory.stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(stamp, tz=timezone.utc).isoformat()


def _pick(*values):
    for value in values:
        if isinstance(value, str) and value.strip():
            return value
    return None


def _pick_int(*values):
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    return None


def _pick_float(*values):
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _pick_bool(*values):
    for value in values:
        if isinstance(value, bool):
            return value
    return None
