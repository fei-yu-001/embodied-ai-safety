"""Loopback-only Flask API for the local experiment console."""

from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import secrets
from urllib.parse import urlsplit

from flask import Flask, Response, abort, jsonify, request, send_from_directory

from .config import CAMERAS, ENV_ID, ExperimentConfig
from .catalog import get_catalog
from .jobs import RunBusyError, RunManager
from .streaming import VIEWS, stream_frames


def create_app(manager=None):
    frontend_dir = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    app = Flask(__name__, static_folder=str(frontend_dir / "assets"), static_url_path="/assets")
    app.config.update(MAX_CONTENT_LENGTH=16 * 1024)
    manager = manager or RunManager()
    token = secrets.token_urlsafe(32)
    app.extensions["run_manager"] = manager
    app.extensions["console_token"] = token

    @app.before_request
    def local_requests_only():
        # Loopback binding plus Host/Origin/token checks prevent another website
        # from starting a costly local simulation through the user's browser.
        host = urlsplit("http://" + request.host).hostname
        if host not in ("127.0.0.1", "localhost", "::1"):
            return jsonify(error="仅支持本机访问"), 403
        if request.method in ("POST", "PUT", "DELETE", "PATCH"):
            origin = request.headers.get("Origin")
            if origin and origin != request.host_url.rstrip("/"):
                return jsonify(error="请求来源不匹配"), 403
            if request.headers.get("X-Console-Token") != token:
                return jsonify(error="请刷新控制台后重试"), 403

    @app.after_request
    def headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data: blob:; media-src 'self'; style-src 'self'; font-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'"
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(404)
    def missing(_):
        return jsonify(error="资源不存在"), 404

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error="请求过大"), 413

    @app.get("/")
    def index():
        if not (frontend_dir / "index.html").is_file():
            return jsonify(error="前端尚未构建，请在 frontend 目录运行 npm run build。"), 503
        return send_from_directory(frontend_dir, "index.html")

    @app.get("/api/bootstrap")
    def bootstrap():
        packages = {}
        for name in ("robocasa", "robosuite", "mujoco", "gymnasium", "flask"):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = None
        return jsonify(
            token=token, defaults=ExperimentConfig().to_dict(), cameras=list(CAMERAS), env_id=ENV_ID,
            catalog=get_catalog(),
            packages=packages, active_id=manager.active_id,
            modules=[
                {"name": "RoboCasa 仿真", "status": "ready", "detail": "常驻厨房环境 · 实时相机流"},
                {"name": "攻击注入", "status": "ready", "detail": "固定随机补丁 · 尺寸 / 位置 / 透明度"},
                {"name": "图像防御", "status": "baseline", "detail": "中值滤波基线 · 可关闭"},
                {"name": "策略", "status": "baseline", "detail": "随机动作占位 · VLA 待接入"},
                {"name": "运行监测", "status": "baseline", "detail": "观测变化提醒 · 动作合法性检查"},
                {"name": "ESP32 硬件", "status": "pending", "detail": "未连接 · 物理熔断待实现"},
            ],
        )

    @app.get("/api/health")
    def health():
        return jsonify(service="embodied-safety-console", status="ok", active_id=manager.active_id)

    @app.get("/api/runs")
    def runs():
        return jsonify(runs=manager.list_runs(), active_id=manager.active_id)

    @app.post("/api/runs")
    def start():
        try:
            config = ExperimentConfig.from_dict(request.get_json(silent=True))
            return jsonify(manager.start(config)), 201
        except ValueError as error:
            return jsonify(error=str(error)), 400
        except RunBusyError as error:
            return jsonify(error=str(error)), 409
        except OSError as error:
            return jsonify(error=f"无法创建工作进程：{error}"), 500

    @app.get("/api/runs/<run_id>")
    def detail(run_id):
        try:
            return jsonify(manager.describe(run_id, detail=True))
        except KeyError:
            abort(404)

    @app.post("/api/runs/<run_id>/control")
    def control(run_id):
        try:
            return jsonify(manager.control(run_id, request.get_json(silent=True))), 202
        except KeyError:
            abort(404)
        except ValueError as error:
            return jsonify(error=str(error)), 400
        except RunBusyError as error:
            return jsonify(error=str(error)), 409

    @app.get("/api/runs/<run_id>/stream")
    def stream(run_id):
        try:
            folder = manager.resolve(run_id)
        except KeyError:
            abort(404)
        view = request.args.get("view", "scene")
        if view not in VIEWS:
            return jsonify(error="未知相机视图"), 400
        response = Response(stream_frames(folder, VIEWS[view]),
                            mimetype="multipart/x-mixed-replace; boundary=frame")
        response.headers["X-Accel-Buffering"] = "no"
        return response

    @app.post('/api/runs/<run_id>/camera')
    def camera(run_id):
        try:
            return jsonify(manager.camera(run_id, request.get_json(silent=True))), 202
        except KeyError:
            abort(404)
        except ValueError as error:
            return jsonify(error=str(error)), 400
        except RunBusyError as error:
            return jsonify(error=str(error)), 409

    @app.post("/api/runs/<run_id>/cancel")
    def cancel(run_id):
        try:
            return jsonify(manager.cancel(run_id))
        except KeyError:
            abort(404)
        except RunBusyError as error:
            return jsonify(error=str(error)), 409

    @app.get("/api/runs/<run_id>/artifacts/<path:filename>")
    def artifact(run_id, filename):
        try:
            folder = manager.resolve(run_id)
        except KeyError:
            abort(404)
        target = (folder / filename).resolve()
        allowed = {".mp4", ".png", ".jpg", ".json", ".jsonl", ".md", ".txt", ".xml", ".npz", ".log"}
        if not target.is_relative_to(folder) or not target.is_file() or target.suffix.lower() not in allowed:
            abort(404)
        # Flask/Werkzeug supports byte ranges for browser video seeking.
        return send_from_directory(folder, filename, conditional=True,
                                   as_attachment=request.args.get("download") == "1")

    return app
