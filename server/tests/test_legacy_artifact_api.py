"""HTTP contract for the read-only historical run routes."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from fastapi.testclient import TestClient

from server.api.v1.app import create_app
from server.application.session_service import SessionService
from server.domain.legacy_contract import REQUIRED_VIEW_KEYS
from server.infrastructure.legacy_runs import get_artifact_path
from server.tests.test_session_control import MemoryWorker, NOW


RUN_ID = "run_20261109_145512_5b6ce9"
LEGACY_NAME = "patch_attack_seed0"
LEGACY_ID = f"legacy_{LEGACY_NAME}"
ARTIFACT = "clean.mp4"
ARTIFACT_BYTES = b"\x00\x01\x02\x03" * 8


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def snapshot(root):
    state = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            path = Path(dirpath) / name
            stat = path.lstat()
            state[str(path.relative_to(root))] = (stat.st_size, stat.st_mtime_ns)
    return state


class LegacyArtifactApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="prism-legacy-api-")
        self.root = Path(self.temporary.name)
        self.runs_dir = self.root / "runs"
        self.output_dir = self.root / "output"
        self.runs_dir.mkdir()
        self.output_dir.mkdir()

        self.run_dir = self.runs_dir / RUN_ID
        write_json(
            self.run_dir / "job.json",
            {"id": RUN_ID, "status": "completed", "created_at": "2026-11-09T14:55:12.747022+08:00"},
        )
        write_json(
            self.run_dir / "summary.json",
            {
                "experiment": "digital_noise_patch_pipeline",
                "created_at": "2026-11-09T14:55:12.747022+08:00",
                "steps_requested": 100,
                "steps_completed": 100,
            },
        )
        (self.run_dir / ARTIFACT).write_bytes(ARTIFACT_BYTES)
        (self.run_dir / "report.md").write_text("# report", encoding="utf-8")

        self.legacy_dir = self.output_dir / LEGACY_NAME
        write_json(
            self.legacy_dir / "summary.json",
            {"experiment": "pipeline_smoke", "created_at": "2026-11-08T21:44:19.591226+08:00"},
        )
        (self.legacy_dir / "comparison.mp4").write_bytes(b"\x00" * 16)

        self.secret = self.root / "secret.json"
        self.secret.write_text("{\"leaked\": true}", encoding="utf-8")

        worker = MemoryWorker()
        service = SessionService(self.root, worker, clock=lambda: NOW)
        app = create_app(
            service,
            "test-token",
            legacy_runs_dir=str(self.runs_dir),
            legacy_output_dir=str(self.output_dir),
        )
        self.app = app
        self.client = TestClient(app, base_url="http://127.0.0.1")
        self.headers = {"X-Console-Token": "test-token"}

    def tearDown(self):
        self.temporary.cleanup()

    def test_record_returns_200_with_the_full_contract(self):
        response = self.client.get(f"/api/v1/legacy-runs/{RUN_ID}", headers=self.headers)

        self.assertEqual(response.status_code, 200)
        record = response.json()
        for key in REQUIRED_VIEW_KEYS:
            self.assertIn(key, record)
        self.assertEqual(record["id"], RUN_ID)
        self.assertIs(record["legacy"], False)
        self.assertIs(record["writable"], False)
        self.assertEqual(record["status"], "CLOSED")
        self.assertEqual(record["legacy_status"], "completed")
        self.assertEqual(record["steps_completed"], 100)
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")

    def test_legacy_record_is_served_under_the_prefixed_id(self):
        response = self.client.get(f"/api/v1/legacy-runs/{LEGACY_ID}", headers=self.headers)

        self.assertEqual(response.status_code, 200)
        record = response.json()
        self.assertEqual(record["id"], LEGACY_ID)
        self.assertIs(record["legacy"], True)
        self.assertIs(record["writable"], False)

    def test_unknown_run_id_returns_404_with_a_code(self):
        response = self.client.get("/api/v1/legacy-runs/run_missing", headers=self.headers)

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["code"], "legacy_run_not_found")
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")

    def test_artifact_download_matches_the_bytes_on_disk(self):
        response = self.client.get(
            f"/api/v1/legacy-runs/{RUN_ID}/artifacts/{ARTIFACT}", headers=self.headers
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, (self.run_dir / ARTIFACT).read_bytes())
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        self.assertNotIn("attachment", response.headers.get("content-disposition", ""))

    def test_download_flag_turns_the_artifact_into_an_attachment(self):
        response = self.client.get(
            f"/api/v1/legacy-runs/{RUN_ID}/artifacts/{ARTIFACT}?download=1", headers=self.headers
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, ARTIFACT_BYTES)
        self.assertIn("attachment", response.headers["content-disposition"])

    def test_unknown_artifact_returns_404_with_a_code(self):
        response = self.client.get(
            f"/api/v1/legacy-runs/{RUN_ID}/artifacts/missing.mp4", headers=self.headers
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["code"], "legacy_artifact_not_found")
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")

    def test_non_whitelisted_suffix_is_not_served(self):
        (self.run_dir / "blob.exe").write_bytes(b"MZ")

        response = self.client.get(
            f"/api/v1/legacy-runs/{RUN_ID}/artifacts/blob.exe", headers=self.headers
        )

        self.assertEqual(response.status_code, 404)

    def test_path_traversal_names_never_escape_the_record_directory(self):
        # sub/../secret.json 会被 httpx/ASGI 在路由前归一化，测不到适配器；
        # 分隔符与 dot-segment 的拒绝在 test_legacy_runs 里直接覆盖。
        attempts = [
            "../secret.json",
            "%2e%2e%2fsecret.json",
            "..%5Csecret.json",
            "..%2F..%2Fsecret.json",
            "frames/leak.png",
            "C:%5Csecret.json",
            f"..%5C..%5C{self.secret}",
        ]
        for attempt in attempts:
            response = self.client.get(
                f"/api/v1/legacy-runs/{RUN_ID}/artifacts/{attempt}", headers=self.headers
            )
            with self.subTest(name=attempt):
                self.assertEqual(response.status_code, 404)
                self.assertNotIn(b"leaked", response.content)

    def test_traversal_run_id_is_not_served(self):
        attempts = ["..", "legacy_..%5C..", "run_missing"]
        for attempt in attempts:
            response = self.client.get(f"/api/v1/legacy-runs/{attempt}", headers=self.headers)
            with self.subTest(run_id=attempt):
                self.assertEqual(response.status_code, 404)

    @unittest.skipUnless(os.name == "nt", "Windows directory junctions")
    def test_junction_escape_is_rejected(self):
        outside = self.root / "junction_target"
        write_json(outside / "job.json", {"id": "junction_target", "status": "completed"})
        (outside / "leak.json").write_text("{\"leaked\": true}", encoding="utf-8")
        link = self.runs_dir / "run_junction"
        created = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(outside)],
            capture_output=True,
        )
        if created.returncode != 0 or not os.path.lexists(link):
            self.skipTest("当前环境无法创建目录联接")
        try:
            record = self.client.get("/api/v1/legacy-runs/run_junction", headers=self.headers)
            artifact = self.client.get(
                "/api/v1/legacy-runs/run_junction/artifacts/leak.json", headers=self.headers
            )
            self.assertEqual(record.status_code, 404)
            self.assertEqual(artifact.status_code, 404)
            self.assertNotIn(b"leaked", artifact.content)
        finally:
            os.rmdir(link)

    def test_symlinked_artifact_escape_is_rejected(self):
        link = self.run_dir / "leak.json"
        try:
            os.symlink(self.secret, link)
        except (OSError, NotImplementedError, AttributeError) as error:
            self.skipTest(f"当前环境无法创建符号链接: {error}")
        try:
            response = self.client.get(
                f"/api/v1/legacy-runs/{RUN_ID}/artifacts/leak.json", headers=self.headers
            )
            self.assertEqual(response.status_code, 404)
            self.assertNotIn(b"leaked", response.content)
        finally:
            os.unlink(link)

    def test_gets_artifact_path_for_a_published_file_only(self):
        response = self.client.get(
            f"/api/v1/legacy-runs/{RUN_ID}/artifacts/{ARTIFACT}", headers=self.headers
        )

        self.assertEqual(response.status_code, 200)
        path = get_artifact_path(RUN_ID, ARTIFACT, self.runs_dir, self.output_dir)
        self.assertEqual(Path(path), (self.run_dir / ARTIFACT).resolve())
        self.assertTrue(Path(path).is_absolute())

    def test_artifact_names_with_separators_are_refused_by_the_adapter(self):
        attempts = [
            "../secret.json",
            "sub/../secret.json",
            "sub/secret.json",
            "..\\secret.json",
            "C:\\secret.json",
            "/etc/secret.json",
            "..",
            "",
            "blob.exe",
            "missing.mp4",
            str(self.secret),
        ]
        for attempt in attempts:
            with self.subTest(name=attempt):
                self.assertIsNone(
                    get_artifact_path(RUN_ID, attempt, self.runs_dir, self.output_dir)
                )

    def test_serving_records_and_artifacts_leaves_the_directories_untouched(self):
        before = {**snapshot(self.runs_dir), **snapshot(self.output_dir)}

        self.client.get("/api/v1/legacy-runs", headers=self.headers)
        self.client.get(f"/api/v1/legacy-runs/{RUN_ID}", headers=self.headers)
        self.client.get(f"/api/v1/legacy-runs/{LEGACY_ID}", headers=self.headers)
        self.client.get(
            f"/api/v1/legacy-runs/{RUN_ID}/artifacts/{ARTIFACT}", headers=self.headers
        )
        self.client.get("/api/v1/legacy-runs/run_missing", headers=self.headers)

        self.assertEqual(before, {**snapshot(self.runs_dir), **snapshot(self.output_dir)})

    def test_token_check_still_applies_to_the_new_routes(self):
        record = self.client.get(f"/api/v1/legacy-runs/{RUN_ID}")
        artifact = self.client.get(
            f"/api/v1/legacy-runs/{RUN_ID}/artifacts/{ARTIFACT}",
            headers={"Host": "127.0.0.1"},
        )

        self.assertEqual(record.status_code, 403)
        self.assertEqual(artifact.status_code, 403)

    def test_legacy_routes_stay_read_only(self):
        legacy_routes = [route for route in self.app.routes if getattr(route, "path", "").startswith("/api/v1/legacy-runs")]
        self.assertTrue(legacy_routes)
        for route in legacy_routes:
            with self.subTest(path=route.path):
                self.assertEqual(route.methods, {"GET"})


if __name__ == "__main__":
    unittest.main()
