"""HTTP contract for the M1 control plane."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from server.api.v1.app import create_app
from server.application.session_service import SessionService
from server.tests.test_session_control import MemoryWorker, NOW


class SessionApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="prism-api-")
        self.worker = MemoryWorker()
        self.service = SessionService(Path(self.temporary.name), self.worker, clock=lambda: NOW)
        self.client = TestClient(create_app(self.service, "test-token"), base_url="http://127.0.0.1")
        self.headers = {"X-Console-Token": "test-token"}

    def tearDown(self):
        self.temporary.cleanup()

    def test_health_does_not_claim_simulation_is_ready(self):
        response = self.client.get("/api/v1/health", headers={"Host": "127.0.0.1"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["api"], "ok")
        self.assertNotEqual(response.json()["simulation"], "ready")

    def test_allocation_failure_explains_that_environment_did_not_load(self):
        from unittest.mock import patch
        with patch.object(self.worker, "start", side_effect=MemoryError("bad allocation")):
            response = self.client.post("/api/v1/sessions", headers=self.headers)
        self.assertEqual(response.status_code, 503)
        self.assertIn("内存不足", response.json()["message"])
        self.assertIn("未载入", response.json()["message"])
        self.assertIn("sess_", response.json()["message"])

    def test_write_requires_token_and_loopback(self):
        missing = self.client.post("/api/v1/sessions")
        self.assertEqual(missing.status_code, 403)
        foreign = self.client.post("/api/v1/sessions", headers={**self.headers, "Host": "example.test"})
        self.assertEqual(foreign.status_code, 403)
        self.assertEqual(self.worker.submissions, [])

    def test_duplicate_command_returns_original_acknowledgement(self):
        created = self.client.post("/api/v1/sessions", headers=self.headers)
        self.assertEqual(created.status_code, 201)
        session = created.json()
        self.service.mark_ready(session["id"])
        body = self._body(session, "step", "step-http")
        first = self.client.post(f"/api/v1/sessions/{session['id']}/commands", headers=self.headers, json=body)
        second = self.client.post(f"/api/v1/sessions/{session['id']}/commands", headers=self.headers, json={**body, "command": "reset"})

        self.assertEqual(first.status_code, 202)
        self.assertEqual(first.json()["acknowledgement"], "accepted")
        self.assertEqual(second.status_code, 202)
        self.assertEqual(second.json(), first.json())
        self.assertEqual(self.worker.submissions, [(session["id"], "step", "step-http")])

    def _body(self, session, command, command_id):
        return {
            "command_id": command_id,
            "fencing_token": session["fencing_token"],
            "command": command,
            "issued_at": NOW.isoformat(),
            "expires_at": (NOW + timedelta(seconds=30)).isoformat(),
            "expected_state_version": self.service.get(session["id"])["state_version"],
        }


if __name__ == "__main__":
    unittest.main()
