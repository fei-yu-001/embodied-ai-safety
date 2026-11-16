"""Console login probes must never allocate sessions or expose credentials."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from server.api.v1.app import create_app
from server.application.session_service import SessionService
from server.tests.test_session_control import MemoryWorker, NOW


class ConsoleAuthTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="prism-auth-")
        self.addCleanup(self.temporary.cleanup)
        self.worker = MemoryWorker()
        self.service = SessionService(Path(self.temporary.name), self.worker, clock=lambda: NOW)
        self.client = TestClient(create_app(self.service, "test-secret"), base_url="http://127.0.0.1")
        self.addCleanup(self.client.close)
        self.headers = {"X-Console-Token": "test-secret"}

    def test_auth_requires_a_valid_token_without_allocating(self):
        with patch.object(self.service, "create") as create:
            for headers in ({}, {"X-Console-Token": "wrong"}):
                with self.subTest(headers=headers):
                    response = self.client.get("/api/v1/auth", headers=headers)
                    self.assertEqual(response.status_code, 403)
                    self.assertEqual(response.json()["code"], "forbidden_token")
                    self.assertIn("重新登录", response.json()["message"])
            response = self.client.get("/api/v1/auth", headers=self.headers)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"authenticated": True})
            self.assertEqual(response.headers["cache-control"], "no-store")
            self.assertNotIn("test-secret", response.text)
            create.assert_not_called()

    def test_auth_preserves_host_restriction(self):
        response = self.client.get("/api/v1/auth", headers={**self.headers, "Host": "example.test"})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["code"], "forbidden_host")

    def test_protected_routes_stay_protected(self):
        for method, path in [("POST", "/api/v1/sessions"),
                             ("GET", "/api/v1/sessions/sess_missing"),
                             ("GET", "/api/v1/sessions/sess_missing/events"),
                             ("GET", "/api/v1/sessions/sess_missing/scene"),
                             ("GET", "/api/v1/sessions/sess_missing/frame"),
                             ("GET", "/api/v1/evaluations")]:
            with self.subTest(path=path):
                response = self.client.request(method, path)
                self.assertEqual(response.status_code, 403)
                self.assertEqual(response.json()["code"], "forbidden_token")
        self.assertEqual(self.client.get("/api/v1/health").status_code, 200)

    def test_invalid_config_with_valid_auth_stays_a_config_error(self):
        with patch.object(self.service, "create", side_effect=ValueError("env_id 无效")):
            response = self.client.post("/api/v1/sessions", headers=self.headers,
                                        json={"config": {"env_id": "invalid-env"}})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["code"], "invalid_config")
        self.assertEqual(self.client.get("/api/v1/auth", headers=self.headers).status_code, 200)

    def test_valid_auth_still_allows_session_creation(self):
        self.assertEqual(self.client.get("/api/v1/auth", headers=self.headers).status_code, 200)
        response = self.client.post("/api/v1/sessions", headers=self.headers)
        self.assertEqual(response.status_code, 201)
        self.assertTrue(response.json()["id"].startswith("sess_"))


if __name__ == "__main__":
    unittest.main()
