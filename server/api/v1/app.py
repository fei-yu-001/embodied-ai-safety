"""Loopback FastAPI control plane for M1 session commands."""

from __future__ import annotations

import asyncio
from datetime import datetime
import os
import re
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, ValidationError

from server.application.evaluation_service import EvaluationBusy
from server.domain.evaluation import SpecError
from server.application.session_service import WorkerUnavailable
from server.domain.session_control import CommandEnvelope, ControlError


class SessionCreateIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    config: dict | None = None


class EvaluationIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spec: dict | None = None


class LeaseIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    holder: str
    ttl_ms: int = 1500


class LeaseRenewIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ttl_ms: int = 1500


class CommandIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    command_id: str
    fencing_token: str
    command: str
    issued_at: datetime
    expires_at: datetime
    expected_state_version: int | None = None
    args: dict | None = None


def _frame_publish_failed(session_state):
    """True when the newest frame on disk is not a successful publish.

    A live worker that just failed to publish must not be reported as a live
    frame: the API would otherwise serve an old JPEG under a live header.
    """
    if not isinstance(session_state, dict):
        return True
    environment = session_state.get("environment")
    if not isinstance(environment, dict):
        return False  # 后端未上报帧信息（如 state_only）：沿用 live 判定
    error = environment.get("frame_error")
    if isinstance(error, dict):
        published = environment.get("published_frame_seq")
        seq = error.get("seq")
        # 仅当失败的就是尚未被后续成功帧取代的那个 seq 才算失效
        if isinstance(published, int) and isinstance(seq, int) and published >= seq:
            return False
        return True
    return False


def create_app(service, token, legacy_runs_dir=None, legacy_output_dir=None, frontend_dir=None, frames_root=None, evaluations=None):
    app = FastAPI(title="PRISM Control Plane", version="0.3.0")
    app.state.service = service
    app.state.token = token
    app.state.legacy_runs_dir = legacy_runs_dir
    app.state.legacy_output_dir = legacy_output_dir
    app.state.frames_root = Path(frames_root) if frames_root else None
    app.state.evaluations = evaluations

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        host = request.url.hostname
        if host not in {"127.0.0.1", "localhost", "::1"}:
            return _error(403, "forbidden_host", "仅支持本机访问")
        if request.url.path.startswith("/api/v1") and request.url.path != "/api/v1/health":
            if request.headers.get("x-console-token") != token:
                return _error(403, "forbidden_token", "请刷新控制台后重试")
        return await call_next(request)

    @app.get("/api/v1/health")
    def health():
        return {"api": "ok", "worker": "separate_process", "simulation": "not_reported"}

    @app.post("/api/v1/sessions", status_code=201)
    def create_session(body: SessionCreateIn | None = None):
        try:
            return service.create(body.config if body else None)
        except ValidationError as error:
            return _error(400, "invalid_config", "会话配置无效")
        except Exception as error:  # config validation happens in the worker adapter
            message = str(error)
            if "配置" in message or "env_id" in message or "control_freq" in message:
                return _error(400, "invalid_config", message)
            return _error(503, "worker_unavailable", message)

    @app.get("/api/v1/sessions/{session_id}")
    def get_session(session_id: str):
        try:
            return service.get(session_id)
        except KeyError:
            return _error(404, "session_not_found", "会话不存在")

    @app.post("/api/v1/sessions/{session_id}/commands")
    def submit_command(session_id: str, body: CommandIn):
        if body.issued_at.tzinfo is None or body.expires_at.tzinfo is None:
            return _error(400, "invalid_timestamp", "时间必须带时区")
        envelope = CommandEnvelope(
            command_id=body.command_id,
            session_id=session_id,
            fencing_token=body.fencing_token,
            command=body.command,
            issued_at=body.issued_at,
            expires_at=body.expires_at,
            expected_state_version=body.expected_state_version,
            args=body.args or {},
        )
        try:
            ack = service.submit(envelope)
        except KeyError:
            return _error(404, "session_not_found", "会话不存在")
        except ControlError as error:
            return _error(400, error.code, error.message)
        except WorkerUnavailable as error:
            return _error(503, "worker_unavailable", str(error))
        return JSONResponse(ack.to_dict(), status_code=_status_for(ack))

    @app.post("/api/v1/sessions/{session_id}/lease")
    def acquire_lease(session_id: str, body: LeaseIn):
        try:
            return service.acquire_lease(session_id, body.holder, body.ttl_ms)
        except KeyError:
            return _error(404, "session_not_found", "会话不存在")
        except ControlError as error:
            return _error(409 if error.code == "lease_held" else 400, error.code, error.message)

    @app.post("/api/v1/sessions/{session_id}/lease/{lease_id}/renew")
    def renew_lease(session_id: str, lease_id: str, body: LeaseRenewIn):
        try:
            return service.renew_lease(session_id, lease_id, body.ttl_ms)
        except KeyError:
            return _error(404, "session_not_found", "会话不存在")
        except ControlError as error:
            return _error(409, error.code, error.message)

    @app.delete("/api/v1/sessions/{session_id}/lease/{lease_id}")
    def release_lease(session_id: str, lease_id: str):
        try:
            return {"released": service.release_lease(session_id, lease_id)}
        except KeyError:
            return _error(404, "session_not_found", "会话不存在")
        except ControlError as error:
            return _error(409, error.code, error.message)

    @app.get("/api/v1/sessions/{session_id}/events")
    def events(session_id: str, after: int = 0):
        try:
            return {"events": service.events_after(session_id, after)}
        except KeyError:
            return _error(404, "session_not_found", "会话不存在")
        except ControlError as error:
            return _error(400, error.code, error.message)

    @app.websocket("/api/v1/sessions/{session_id}/ws")
    async def session_stream(websocket: WebSocket, session_id: str, token: str = ""):
        """Push state and incremental events. Browsers cannot set headers on a
        WebSocket, so the loopback-only token travels as a query parameter."""
        if token != app.state.token:
            await websocket.close(code=1008)
            return
        host = websocket.client.host if websocket.client else ""
        if host not in {"127.0.0.1", "localhost", "::1", "testclient"}:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        cursor = 0
        try:
            while True:
                try:
                    snapshot = service.get(session_id)
                except KeyError:
                    await websocket.send_json({"type": "error", "code": "session_not_found"})
                    await websocket.close(code=1008)
                    return
                events = service.events_after(session_id, cursor)
                if events:
                    cursor = events[-1]["seq"]
                await websocket.send_json({"type": "snapshot", "session": snapshot, "events": events})
                await asyncio.sleep(0.2)
        except (WebSocketDisconnect, RuntimeError):
            return

    @app.get("/api/v1/legacy-runs")
    def legacy_runs():
        reader = _legacy_reader()
        if reader is None:
            return _error(503, "legacy_adapter_unavailable", "历史只读适配器尚未加载")
        try:
            return {"runs": reader.list_legacy_runs(app.state.legacy_runs_dir, app.state.legacy_output_dir)}
        except Exception as error:
            return _error(500, "legacy_read_failed", "历史记录读取失败")

    @app.get("/api/v1/legacy-runs/{run_id}")
    def legacy_run(run_id: str):
        reader = _legacy_reader()
        if reader is None:
            return _no_sniff(_error(503, "legacy_adapter_unavailable", "历史只读适配器尚未加载"))
        try:
            record = reader.get_legacy_run(run_id, app.state.legacy_runs_dir, app.state.legacy_output_dir)
        except Exception:
            return _no_sniff(_error(500, "legacy_read_failed", "历史记录读取失败"))
        if record is None:
            return _no_sniff(_error(404, "legacy_run_not_found", "历史记录不存在"))
        return _no_sniff(JSONResponse(record))

    @app.get("/api/v1/legacy-runs/{run_id}/artifacts/{name:path}")
    def legacy_artifact(run_id: str, name: str, download: int = 0):
        reader = _legacy_reader()
        if reader is None or not hasattr(reader, "get_artifact_path"):
            return _no_sniff(_error(503, "legacy_adapter_unavailable", "历史只读适配器尚未加载"))
        try:
            path = reader.get_artifact_path(
                run_id, name, app.state.legacy_runs_dir, app.state.legacy_output_dir
            )
        except Exception:
            return _no_sniff(_error(500, "legacy_read_failed", "历史记录读取失败"))
        if path is None:
            return _no_sniff(_error(404, "legacy_artifact_not_found", "历史产物不存在"))
        if download:
            return FileResponse(path, filename=os.path.basename(path), headers=_NO_SNIFF)
        return FileResponse(path, headers=_NO_SNIFF)

    @app.exception_handler(ValidationError)
    async def invalid_body(_request, _error):
        return _error(400, "invalid_body", "命令格式无效")

    @app.post("/api/v1/evaluations", status_code=201)
    def create_evaluation(body: EvaluationIn):
        if app.state.evaluations is None:
            return _error(503, "evaluations_unavailable", "评测服务未配置")
        try:
            return app.state.evaluations.create(body.spec)
        except SpecError as error:
            return _error(400, "invalid_spec", str(error))
        except EvaluationBusy as error:
            return _error(409, "evaluation_busy", str(error))
        except OSError as error:
            return _error(503, "evaluation_spawn_failed", f"无法启动评测进程：{error}")

    @app.get("/api/v1/evaluations")
    def list_evaluations():
        if app.state.evaluations is None:
            return _error(503, "evaluations_unavailable", "评测服务未配置")
        return {"evaluations": app.state.evaluations.list()}

    @app.get("/api/v1/evaluations/{run_id}")
    def get_evaluation(run_id: str):
        if app.state.evaluations is None:
            return _error(503, "evaluations_unavailable", "评测服务未配置")
        try:
            return app.state.evaluations.get(run_id)
        except KeyError:
            return _error(404, "evaluation_not_found", "评测不存在")

    @app.get("/api/v1/sessions/{session_id}/scene")
    def scene(session_id: str, geometry_key: str | None = None):
        try:
            return service.scene(session_id, geometry_key)
        except KeyError:
            return _error(404, "session_not_found", "会话不存在")
        except WorkerUnavailable as error:
            return _error(503, "scene_unavailable", str(error))

    @app.get("/api/v1/sessions/{session_id}/visual-assets/{filename}")
    def visual_asset(session_id: str, filename: str):
        if not re.fullmatch(r"sess_[A-Za-z0-9_]{1,80}", session_id) or not re.fullmatch(r"(?:mesh_(?:[0-9]{4,8}_)?[a-f0-9]{12}\.json|tex_(?:[0-9]{4,8}_)?[a-f0-9]{12}\.png)", filename):
            return _error(404, "asset_not_found", "场景资产不存在")
        if app.state.frames_root is None:
            return _error(503, "assets_unavailable", "未配置场景资产目录")
        try:
            service.get(session_id)
        except KeyError:
            return _error(404, "session_not_found", "会话不存在")
        root = app.state.frames_root.resolve()
        path = (root / session_id / "visual_assets" / filename).resolve()
        if not path.is_relative_to(root / session_id) or not path.is_file():
            return _error(404, "asset_not_found", "场景资产不存在")
        return FileResponse(path, media_type="image/png" if filename.endswith(".png") else "application/json",
                            headers={"Cache-Control": "private, max-age=86400, immutable", "X-Content-Type-Options": "nosniff"})

    @app.get("/api/v1/sessions/{session_id}/frame")
    def frame(session_id: str, seq: int | None = None):
        if app.state.frames_root is None:
            return _error(503, "frames_unavailable", "未配置帧目录")
        if not session_id.startswith("sess_") or "/" in session_id or "\\" in session_id:
            return _error(404, "session_not_found", "会话不存在")
        if seq is not None and seq < 0:
            return _error(400, "frame_seq_invalid", "seq 必须是非负整数")
        from server.infrastructure.frame_store import FrameStore

        store = FrameStore(app.state.frames_root / session_id / "frames")
        # seq 兼容：指定序号读编号帧；不指定就读当前最新编号帧，
        # 只有旧会话存在 latest.jpg 时才回退到它。
        resolved = store.describe(seq)
        if resolved is None:
            return _error(404, "frame_not_found", "该帧还不存在")
        # 字节快照：一次完整读入，句柄立即释放，不让响应期间持有文件
        try:
            payload = store.read_bytes(resolved["path"])
        except (OSError, ValueError):
            return _error(404, "frame_not_found", "该帧还不存在")
        # 磁盘上可能还留着上一轮会话的图。没有活动 worker 时必须标明这是历史帧，
        # 否则界面会把它当成实时画面。
        try:
            session_state = service.get(session_id)
            live = bool(session_state.get("live"))
        except KeyError:
            session_state = {}
            live = False
        # worker 活着不等于当前帧是刚发布的：最近一次帧发布失败时，
        # 磁盘上的旧帧不能称实时。frame_error 来自已发布 seq 与错误码。
        if _frame_publish_failed(session_state):
            live = False
        # Snapshot bytes in one read: the worker replaces frames while requests
        # are in flight, and a streaming FileResponse can then exceed the length
        # it advertised.
        from fastapi import Response

        response = Response(content=payload, media_type="image/jpeg")
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Prism-Frame-Live"] = "1" if live else "0"
        response.headers["X-Prism-Frame-Source"] = resolved["source"]
        if resolved["seq"] is not None:
            response.headers["X-Prism-Frame-Seq"] = str(resolved["seq"])
        return response

    if frontend_dir:
        _mount_frontend(app, Path(frontend_dir))

    return app


def _mount_frontend(app, directory):
    index = directory / "index.html"
    if not index.is_file():
        return
    assets = directory / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/")
    def workbench():
        response = FileResponse(index, media_type="text/html")
        response.headers["Cache-Control"] = "no-store"
        return response


def _legacy_reader():
    try:
        from server.infrastructure import legacy_runs
    except ImportError:
        return None
    if not hasattr(legacy_runs, "list_legacy_runs"):
        return None
    return legacy_runs


_NO_SNIFF = {"X-Content-Type-Options": "nosniff"}


def _no_sniff(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


def _status_for(ack):
    if ack.acknowledgement == "accepted":
        return 202
    if ack.code == "lease_mismatch":
        return 403
    if ack.acknowledgement in {"rejected", "superseded"}:
        return 409
    return 200


def _error(status, code, message):
    return JSONResponse({"code": code, "message": message}, status_code=status)
