"""Read-only behavior of the historical run adapter, without Flask or MuJoCo."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from server.domain.legacy_contract import REQUIRED_VIEW_KEYS, STATUS_MAP
from server.infrastructure import legacy_runs
from server.infrastructure.legacy_runs import get_legacy_run, list_legacy_runs


LEGAL_STATUSES = set(STATUS_MAP.values()) | {"UNKNOWN"}
NEW_RUN = "run_20261109_145512_5b6ce9"
LEGACY_NAME = "patch_attack_seed0"


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def snapshot(root):
    state = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames:
            path = Path(dirpath) / name
            state[str(path.relative_to(root)) + "/"] = path.stat().st_mtime_ns
        for name in filenames:
            path = Path(dirpath) / name
            if path.is_symlink():
                state[str(path.relative_to(root))] = ("link", os.readlink(path))
                continue
            data = path.read_bytes()
            state[str(path.relative_to(root))] = (
                hashlib.sha256(data).hexdigest(),
                path.stat().st_mtime_ns,
            )
    return state


class LegacyRunsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="prism-legacy-")
        self.root = Path(self.temporary.name)
        self.runs_dir = self.root / "runs"
        self.output_dir = self.root / "output"
        self.runs_dir.mkdir()
        self.output_dir.mkdir()
        self.make_new_run(NEW_RUN)
        self.make_legacy_run(LEGACY_NAME)

    def tearDown(self):
        self.temporary.cleanup()

    def make_new_run(self, name, status="completed"):
        directory = self.runs_dir / name
        write_json(
            directory / "job.json",
            {
                "id": name,
                "status": status,
                "created_at": "2026-11-09T14:55:12.747022+08:00",
                "returncode": 0,
            },
        )
        write_json(
            directory / "summary.json",
            {
                "experiment": "digital_noise_patch_pipeline",
                "created_at": "2026-11-09T14:55:12.747022+08:00",
                "env_id": "robocasa/PickPlaceCounterToCabinet",
                "policy": "random",
                "seed": 0,
                "steps_requested": 100,
                "steps_completed": 100,
                "stop_reason": "step_limit",
                "success_observed": False,
                "elapsed_seconds": 55.96,
            },
        )
        write_json(
            directory / "progress.json",
            {"status": status, "phase": "实验完成", "step": 100, "total": 100},
        )
        (directory / "clean.mp4").write_bytes(b"\x00" * 32)
        (directory / "report.md").write_text("# report", encoding="utf-8")
        (directory / "blob.exe").write_bytes(b"MZ")
        return directory

    def make_legacy_run(self, name):
        directory = self.output_dir / name
        write_json(
            directory / "summary.json",
            {
                "experiment": "digital_noise_patch_pipeline_smoke",
                "created_at": "2026-11-08T21:44:19.591226+08:00",
                "env_id": "robocasa/PickPlaceCounterToCabinet",
                "steps_requested": 100,
                "steps_completed": 100,
                "stop_reason": "step_limit",
            },
        )
        (directory / "comparison.mp4").write_bytes(b"\x00" * 16)
        return directory

    def ids(self):
        return [record["id"] for record in list_legacy_runs(self.runs_dir, self.output_dir)]

    def test_every_record_carries_the_required_contract_keys(self):
        records = list_legacy_runs(self.runs_dir, self.output_dir)
        self.assertEqual(len(records), 2)
        for record in records:
            for key in REQUIRED_VIEW_KEYS:
                self.assertIn(key, record)
            self.assertIs(record["writable"], False)

    def test_new_run_directory_is_read_as_one_record(self):
        record = get_legacy_run(NEW_RUN, self.runs_dir, self.output_dir)
        self.assertIsNotNone(record)
        self.assertEqual(record["id"], NEW_RUN)
        self.assertIs(record["legacy"], False)
        self.assertIs(record["writable"], False)
        self.assertEqual(record["legacy_status"], "completed")
        self.assertEqual(record["status"], "CLOSED")
        self.assertEqual(record["created_at"], "2026-11-09T14:55:12.747022+08:00")
        self.assertEqual(record["steps_completed"], 100)
        self.assertEqual(record["steps_requested"], 100)
        self.assertEqual(record["phase"], "实验完成")
        self.assertIsNone(record["error"])
        self.assertEqual(record["summary"]["env_id"], "robocasa/PickPlaceCounterToCabinet")
        self.assertIs(record["summary"]["success_observed"], False)

    def test_artifacts_are_whitelisted_and_carry_name_bytes_url(self):
        record = get_legacy_run(NEW_RUN, self.runs_dir, self.output_dir)
        names = [item["name"] for item in record["artifacts"]]
        self.assertIn("clean.mp4", names)
        self.assertIn("report.md", names)
        self.assertIn("job.json", names)
        self.assertNotIn("blob.exe", names)
        for item in record["artifacts"]:
            self.assertEqual(set(item), {"name", "bytes", "url"})
            self.assertIsInstance(item["bytes"], int)
            self.assertTrue(item["url"].startswith(f"/api/v1/legacy-runs/{NEW_RUN}/artifacts/"))
        clean = [item for item in record["artifacts"] if item["name"] == "clean.mp4"][0]
        self.assertEqual(clean["bytes"], 32)

    def test_legacy_directory_gets_the_prefixed_id(self):
        record = get_legacy_run(f"legacy_{LEGACY_NAME}", self.runs_dir, self.output_dir)
        self.assertIsNotNone(record)
        self.assertEqual(record["id"], f"legacy_{LEGACY_NAME}")
        self.assertIs(record["legacy"], True)
        self.assertIs(record["writable"], False)
        self.assertEqual(record["created_at"], "2026-11-08T21:44:19.591226+08:00")
        self.assertIn(f"legacy_{LEGACY_NAME}", self.ids())
        self.assertNotIn(LEGACY_NAME, self.ids())

    def test_corrupt_job_json_is_skipped_without_breaking_the_list(self):
        broken = self.runs_dir / "run_broken"
        broken.mkdir()
        (broken / "job.json").write_text("{not json", encoding="utf-8")
        empty = self.runs_dir / "run_empty"
        empty.mkdir()
        write_json(empty / "job.json", {})
        (empty / "summary.json").write_text("[]", encoding="utf-8")

        ids = self.ids()

        self.assertNotIn("run_broken", ids)
        self.assertNotIn("run_empty", ids)
        self.assertIn(NEW_RUN, ids)
        self.assertIsNone(get_legacy_run("run_broken", self.runs_dir, self.output_dir))

    def test_directory_without_marker_file_is_not_a_record(self):
        stray = self.runs_dir / "run_stray"
        stray.mkdir()
        (stray / "actions.jsonl").write_text("{}\n", encoding="utf-8")

        self.assertNotIn("run_stray", self.ids())
        self.assertIsNone(get_legacy_run("run_stray", self.runs_dir, self.output_dir))

    def test_path_traversal_is_rejected(self):
        secret = self.root / "secret"
        write_json(secret / "job.json", {"id": "secret", "status": "completed"})

        attempts = [
            "../secret",
            "..",
            "",
            "a/b",
            "runs/../../secret",
            str(secret),
            f"{NEW_RUN}/../../secret",
            f"{LEGACY_NAME}/../{LEGACY_NAME}",
            "legacy_../secret",
            "legacy_",
            "~",
            "C:secret",
        ]
        for attempt in attempts:
            with self.subTest(run_id=attempt):
                self.assertIsNone(get_legacy_run(attempt, self.runs_dir, self.output_dir))

        self.assertNotIn("secret", self.ids())

    def test_symlink_escape_is_rejected(self):
        outside = self.root / "outside"
        write_json(outside / "job.json", {"id": "outside", "status": "completed"})
        link = self.runs_dir / "run_linked"
        try:
            os.symlink(outside, link, target_is_directory=True)
        except (OSError, NotImplementedError, AttributeError) as error:
            self.skipTest(f"当前环境无法创建符号链接: {error}")

        self.assertNotIn("run_linked", self.ids())
        self.assertIsNone(get_legacy_run("run_linked", self.runs_dir, self.output_dir))
        self.assertFalse(link.is_symlink() and not link.exists())

    @unittest.skipUnless(os.name == "nt", "Windows directory junctions")
    def test_junction_escape_is_rejected(self):
        outside = self.root / "junction_target"
        write_json(outside / "job.json", {"id": "junction_target", "status": "completed"})
        link = self.runs_dir / "run_junction"
        created = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(outside)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if created.returncode != 0 or not os.path.lexists(link):
            self.skipTest("当前环境无法创建目录联接")
        try:
            self.assertNotIn("run_junction", self.ids())
            self.assertIsNone(get_legacy_run("run_junction", self.runs_dir, self.output_dir))
        finally:
            os.rmdir(link)

    def test_symlinked_artifact_is_not_published(self):
        outside_file = self.root / "outside.png"
        outside_file.write_bytes(b"\x89PNG")
        target = self.runs_dir / NEW_RUN / "leak.png"
        try:
            os.symlink(outside_file, target)
        except (OSError, NotImplementedError, AttributeError) as error:
            self.skipTest(f"当前环境无法创建符号链接: {error}")

        record = get_legacy_run(NEW_RUN, self.runs_dir, self.output_dir)
        self.assertNotIn("leak.png", [item["name"] for item in record["artifacts"]])

    def test_reading_leaves_the_directories_untouched(self):
        before = snapshot(self.root)

        list_legacy_runs(self.runs_dir, self.output_dir)
        get_legacy_run(NEW_RUN, self.runs_dir, self.output_dir)
        get_legacy_run(f"legacy_{LEGACY_NAME}", self.runs_dir, self.output_dir)
        self.assertIsNone(get_legacy_run("run_missing", self.runs_dir, self.output_dir))

        self.assertEqual(before, snapshot(self.root))

    def test_reading_a_missing_root_is_not_an_exception(self):
        self.assertEqual(list_legacy_runs(None, None), [])
        self.assertEqual(list_legacy_runs(self.runs_dir / "nope", self.output_dir / "nope"), [])
        self.assertIsNone(get_legacy_run(NEW_RUN, self.runs_dir / "nope", None))

    def test_status_mapping_stays_inside_the_contract(self):
        expected = {
            "running": "RUNNING",
            "completed": "CLOSED",
            "failed": "FAILED",
            "cancelled": "CLOSED",
            "queued": "QUEUED",
            "interrupted": "INTERRUPTED",
        }
        for raw, mapped in expected.items():
            name = f"run_{raw}"
            self.make_new_run(name, status=raw)
            record = get_legacy_run(name, self.runs_dir, self.output_dir)
            with self.subTest(status=raw):
                self.assertEqual(record["legacy_status"], raw)
                self.assertEqual(record["status"], mapped)

        self.make_new_run("run_weird", status="does_not_exist")
        weird = get_legacy_run("run_weird", self.runs_dir, self.output_dir)
        self.assertEqual(weird["legacy_status"], "does_not_exist")
        self.assertEqual(weird["status"], "UNKNOWN")

        for record in list_legacy_runs(self.runs_dir, self.output_dir):
            self.assertIn(record["status"], LEGAL_STATUSES)

    def test_adapter_imports_stay_free_of_frameworks_and_experiment_packages(self):
        source = Path(legacy_runs.__file__).read_text(encoding="utf-8")
        imports = [
            line.strip()
            for line in source.splitlines()
            if line.strip().startswith(("import ", "from "))
        ]
        joined = " ".join(imports).lower()
        for banned in ("flask", "fastapi", "experiments"):
            with self.subTest(banned=banned):
                self.assertNotIn(banned, joined)


if __name__ == "__main__":
    unittest.main()
