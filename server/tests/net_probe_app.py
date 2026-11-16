"""Entry point for tests that need a real Uvicorn process on a real socket.

读取环境变量 PRISM_RUNS / PRISM_OUT / PRISM_SESSION_DIR / PRISM_PORT / PRISM_TOKEN，
用只读历史适配器构建控制面。测试进程通过该脚本启动真实的 Uvicorn 服务。
"""

from __future__ import annotations

import os

import uvicorn

from server.api.v1.app import create_app
from server.application.session_service import SessionService


class NullWorker:
    """历史只读路由不会触碰仿真 worker，这里只满足构造要求。"""

    def start(self, session_id):
        pass

    def submit(self, session_id, command, command_id):
        pass

    def alive(self, session_id):
        return True


def main():
    service = SessionService(os.environ["PRISM_SESSION_DIR"], NullWorker(), clock=None)
    app = create_app(
        service,
        os.environ["PRISM_TOKEN"],
        legacy_runs_dir=os.environ["PRISM_RUNS"],
        legacy_output_dir=os.environ["PRISM_OUT"],
    )
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=int(os.environ["PRISM_PORT"]),
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":
    main()
