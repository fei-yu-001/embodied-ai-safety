"""Frame endpoint contract: byte snapshot, source headers, and liveness.

Guards the parts of the WinError 5 fix that live in the API process:

- the endpoint resolves the newest numbered frame without touching a mutable
  pointer file, and reports where the bytes came from,
- `?seq=` still works, and legacy `latest.jpg`-only sessions still resolve,
- a live worker whose last frame publish failed must not be reported as a live
  frame (that would serve an old JPEG under a live header),
- a live worker with a successful publish is reported live.
"""

from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from server.api.v1.app import create_app
from server.application.session_service import SessionService
from server.infrastructure.frame_store import frame_name
from server.tests.test_session_control import MemoryWorker, NOW


def jpeg(payload=b"body"):
    return b"\xff\xd8" + payload + b"\xff\xd9"


class FrameApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="prism-frame-api-")
        self.root = Path(self.temporary.name)
        self.worker = MemoryWorker()
        self.service = SessionService(self.root, self.worker, clock=lambda: NOW)
        self.session = self.service.create()
        self.worker.start(self.session["id"], {})
        self.frames = self.root / self.session["id"] / "frames"
        self.client = TestClient(
            create_app(self.service, "frame-token", frames_root=self.root),
            base_url="http://127.0.0.1",
        )
        self.headers = {"X-Console-Token": "frame-token"}

    def tearDown(self):
        self.temporary.cleanup()

    def report(self, telemetry):
        """Feed worker telemetry without an inflight command (absorb path)."""
        self.service.manual_progress(self.session["id"], 0, telemetry)

    def publish(self, payload, seq):
        self.frames.mkdir(parents=True, exist_ok=True)
        (self.frames / frame_name(seq)).write_bytes(jpeg(payload))
        return seq

    def url(self, seq=None):
        base = f"/api/v1/sessions/{self.session['id']}/frame"
        return f"{base}?seq={seq}" if seq is not None else base

    def test_latest_serves_newest_numbered_frame_with_source_header(self):
        self.publish(b"one", 1)
        self.publish(b"two", 2)
        response = self.client.get(self.url(), headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, jpeg(b"two"))
        self.assertEqual(response.headers["X-Prism-Frame-Source"], "numbered")
        self.assertEqual(response.headers["X-Prism-Frame-Seq"], "2")
        self.assertEqual(response.headers["Cache-Control"], "no-store")

    def test_seq_query_is_honoured(self):
        self.publish(b"one", 1)
        self.publish(b"two", 2)
        response = self.client.get(self.url(1), headers=self.headers)
        self.assertEqual(response.content, jpeg(b"one"))
        self.assertEqual(response.headers["X-Prism-Frame-Seq"], "1")

    def test_missing_seq_is_404(self):
        self.publish(b"one", 1)
        response = self.client.get(self.url(9), headers=self.headers)
        self.assertEqual(response.status_code, 404)

    def test_legacy_latest_only_session_still_resolves(self):
        self.frames.mkdir(parents=True, exist_ok=True)
        (self.frames / "latest.jpg").write_bytes(jpeg(b"legacy"))
        response = self.client.get(self.url(), headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, jpeg(b"legacy"))
        self.assertEqual(response.headers["X-Prism-Frame-Source"], "legacy_latest")

    def test_invalid_seq_is_rejected(self):
        response = self.client.get(self.url(-1), headers=self.headers)
        self.assertEqual(response.status_code, 400)

    def test_live_worker_with_successful_publish_reports_live(self):
        self.publish(b"one", 1)
        self.report({"state": "PAUSED", "step": 1, "frame_seq": 1, "published_frame_seq": 1, "frame_error": None})
        response = self.client.get(self.url(), headers=self.headers)
        self.assertEqual(response.headers["X-Prism-Frame-Live"], "1")

    def test_live_worker_with_failed_publish_is_not_reported_live(self):
        """旧帧仍在磁盘上，但最近一次发布失败：不得标实时。"""
        self.publish(b"one", 1)
        self.report({"state": "PAUSED", "step": 2, "frame_seq": 2, "published_frame_seq": 1,
                     "frame_error": {"code": "frame_write_failed", "reason": "PermissionError:5", "seq": 2}})
        response = self.client.get(self.url(), headers=self.headers)
        # 仍能拿到上一张成功帧（历史），但必须非实时
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Prism-Frame-Live"], "0")

    def test_failure_superseded_by_a_later_success_is_live_again(self):
        self.publish(b"one", 1)
        self.publish(b"two", 2)
        self.report({"state": "PAUSED", "frame_seq": 2, "published_frame_seq": 2, "frame_error": None})
        response = self.client.get(self.url(), headers=self.headers)
        self.assertEqual(response.headers["X-Prism-Frame-Live"], "1")

    def test_works_while_a_reader_holds_an_old_frame(self):
        """API 读旧帧时新帧已存在：端点取最新的，且不因持有而失败。"""
        self.publish(b"one", 1)
        holder = open(self.frames / frame_name(1), "rb")
        try:
            self.publish(b"two", 2)
            response = self.client.get(self.url(), headers=self.headers)
        finally:
            holder.close()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, jpeg(b"two"))


if __name__ == "__main__":
    unittest.main()
