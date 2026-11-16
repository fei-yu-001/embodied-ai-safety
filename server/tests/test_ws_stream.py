"""WebSocket push: state snapshots plus incremental events."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from server.api.v1.app import create_app
from server.application.session_service import SessionService
from server.domain.session_control import CommandEnvelope
from server.tests.test_session_control import MemoryWorker, NOW


class StreamTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="prism-ws-")
        self.worker = MemoryWorker()
        self.service = SessionService(Path(self.temporary.name), self.worker, clock=lambda: NOW)
        self.client = TestClient(create_app(self.service, "ws-token"), base_url="http://127.0.0.1")
        self.session = self.service.create()

    def tearDown(self):
        self.temporary.cleanup()

    def test_token_is_required(self):
        with self.assertRaises(Exception):
            with self.client.websocket_connect(f"/api/v1/sessions/{self.session['id']}/ws?token=wrong") as socket:
                socket.receive_json()
        with self.assertRaises(Exception):
            with self.client.websocket_connect(f"/api/v1/sessions/{self.session['id']}/ws") as socket:
                socket.receive_json()

    def test_first_frame_carries_state_and_then_only_new_events(self):
        with self.client.websocket_connect(
            f"/api/v1/sessions/{self.session['id']}/ws?token=ws-token"
        ) as socket:
            first = socket.receive_json()
            self.assertEqual(first["type"], "snapshot")
            self.assertEqual(first["session"]["state"], "LOADING")
            self.assertEqual([e["type"] for e in first["events"]], ["session_created", "worker_loading"])

            self.service.mark_ready(self.session["id"])
            second = socket.receive_json()
            self.assertEqual(second["session"]["state"], "READY")
            self.assertEqual([e["type"] for e in second["events"]], ["worker_ready"])

            body = CommandEnvelope(
                command_id="ws-step",
                session_id=self.session["id"],
                fencing_token=second["session"]["fencing_token"],
                command="step",
                issued_at=NOW,
                expires_at=NOW + timedelta(seconds=30),
            )
            self.service.submit(body)
            third = socket.receive_json()
            self.assertEqual(third["events"][-1]["type"], "command_accepted")
            self.assertEqual(third["events"][-1]["command_id"], "ws-step")

    def test_unknown_session_closes_with_an_error_frame(self):
        with self.client.websocket_connect("/api/v1/sessions/sess_missing/ws?token=ws-token") as socket:
            message = socket.receive_json()
            self.assertEqual(message["type"], "error")
            self.assertEqual(message["code"], "session_not_found")


if __name__ == "__main__":
    unittest.main()
