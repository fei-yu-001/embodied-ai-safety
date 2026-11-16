"""World-frame scene query used by the 3D view."""

from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from server.api.v1.app import create_app
from server.application.session_service import SessionService, WorkerUnavailable
from server.tests.test_session_control import MemoryWorker, NOW


class BrokenWorker(MemoryWorker):
    def query(self, session_id, name, timeout=3.0, args=None):
        raise WorkerUnavailable("查询超时")


class SceneApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="prism-scene-")
        self.worker = MemoryWorker()
        self.service = SessionService(Path(self.temporary.name), self.worker, clock=lambda: NOW)
        self.session = self.service.create()
        self.client = TestClient(
            create_app(self.service, "scene-token", frames_root=Path(self.temporary.name)),
            base_url="http://127.0.0.1",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_scene_declares_its_frame_and_units(self):
        response = self.client.get(
            f"/api/v1/sessions/{self.session['id']}/scene", headers={"X-Console-Token": "scene-token"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["frame"]["units"], "m")
        self.assertEqual(data["frame"]["up"], "+z")
        self.assertEqual(data["frame"]["quaternion"], "wxyz")
        self.assertEqual(len(data["eef"]["quaternion"]), 4)

    def test_geometry_key_is_forwarded_and_omits_geometry(self):
        seen = {}

        class RecordingWorker(MemoryWorker):
            def query(self, session_id, name, timeout=3.0, args=None):
                seen["args"] = args
                payload = super().query(session_id, name, timeout)
                payload["geometry_key"] = "g_test"
                if (args or {}).get("have_geometry_key") == "g_test":
                    payload["geometry"] = None
                    payload["geometry_included"] = False
                else:
                    payload["geometry"] = {"key": "g_test"}
                    payload["geometry_included"] = True
                return payload

        service = SessionService(Path(self.temporary.name) / "keys", RecordingWorker(), clock=lambda: NOW)
        session = service.create()
        client = TestClient(create_app(service, "scene-token"), base_url="http://127.0.0.1")
        url = f"/api/v1/sessions/{session['id']}/scene"
        headers = {"X-Console-Token": "scene-token"}

        first = client.get(url, headers=headers).json()
        self.assertTrue(first["geometry_included"])
        self.assertIsNotNone(first["geometry"])
        self.assertIsNone(seen["args"]["have_geometry_key"])

        second = client.get(url, params={"geometry_key": "g_test"}, headers=headers).json()
        self.assertFalse(second["geometry_included"])
        self.assertIsNone(second["geometry"])
        self.assertEqual(second["geometry_key"], "g_test")
        self.assertEqual(seen["args"]["have_geometry_key"], "g_test")

    def test_unknown_session_and_unavailable_worker_are_distinct(self):
        missing = self.client.get("/api/v1/sessions/sess_nope/scene", headers={"X-Console-Token": "scene-token"})
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.json()["code"], "session_not_found")

        service = SessionService(Path(self.temporary.name) / "broken", BrokenWorker(), clock=lambda: NOW)
        session = service.create()
        client = TestClient(create_app(service, "scene-token"), base_url="http://127.0.0.1")
        unavailable = client.get(
            f"/api/v1/sessions/{session['id']}/scene", headers={"X-Console-Token": "scene-token"}
        )
        self.assertEqual(unavailable.status_code, 503)
        self.assertEqual(unavailable.json()["code"], "scene_unavailable")


if __name__ == "__main__":
    unittest.main()
