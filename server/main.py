"""Loopback workbench with an explicit backend, port and session root.

Each worker receives the same root as its API; independent instances can run
side by side without reading or replacing another instance's live sessions.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

import uvicorn

from server.api.v1.app import create_app
from server.application.evaluation_service import EvaluationService
from server.application.session_service import SessionService
from server.infrastructure.process_worker import ProcessWorker

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "sessions"


def build_app(token=None, backend=None, data_root=None, frontend_dir=None):
    session_root = Path(data_root).resolve() if data_root else DATA_ROOT
    session_root.mkdir(parents=True, exist_ok=True)
    service_holder = {}
    env = dict(os.environ)
    if backend:
        env["PRISM_BACKEND"] = backend
    worker = ProcessWorker(
        session_root,
        env=env,
        max_sessions=max(0, int(os.environ.get("PRISM_MAX_SESSIONS", "0"))),
        on_loaded=lambda session: service_holder["service"].mark_ready(session),
        on_applied=lambda session, command, detail: service_holder["service"].observe_applied(session, command, detail),
        on_lost=lambda session, reason: service_holder["service"].crash_worker(session, reason),
        on_manual_stopped=lambda session, reason, steps, measured: service_holder["service"].manual_stopped(session, reason, steps, measured),
        on_manual_progress=lambda session, steps, detail: service_holder["service"].manual_progress(session, steps, detail),
        on_policy_progress=lambda session, detail: service_holder["service"].policy_progress(session, detail),
        on_policy_stopped=lambda session, reason, detail: service_holder["service"].policy_stopped(session, reason, detail),
    )
    service = SessionService(session_root, worker)
    service_holder["service"] = service
    evaluations = EvaluationService(session_root) if os.environ.get("PRISM_ENABLE_EVALUATIONS", "1") != "0" else None
    app = create_app(
        service,
        token or secrets.token_urlsafe(24),
        legacy_runs_dir=ROOT / "experiments" / "runs",
        legacy_output_dir=ROOT / "experiments" / "scripts" / "output",
        frontend_dir=Path(frontend_dir) if frontend_dir else ROOT / "web" / "dist",
        frames_root=session_root,
        evaluations=evaluations,
    )
    return app


def main():
    token = os.environ.get("PRISM_TOKEN") or secrets.token_urlsafe(24)
    backend = os.environ.get("PRISM_BACKEND", "state_only")
    port = int(os.environ.get("PRISM_PORT", "8766"))
    if not 1024 <= port <= 65535:
        raise ValueError("PRISM_PORT must be between 1024 and 65535")
    app = build_app(token, backend, data_root=os.environ.get("PRISM_DATA_ROOT"),
                    frontend_dir=os.environ.get("PRISM_FRONTEND_DIR"))
    print(f"PRISM workbench: http://127.0.0.1:{port}")
    print(f"worker backend: {backend}" + ("" if backend == "robocasa" else " (no physics loaded)"))
    print("console authentication: enabled (token supplied through PRISM_TOKEN)")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
