"""Local console contracts, using tiny worker processes and no simulator assets."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import gymnasium as gym
import numpy as np

from experiments.defenses.filters import DefenseWrapper, ImageDefense
from experiments.safety_monitor.runtime import ActionGuard, TemporalImageMonitor
from experiments.system.app import create_app
from experiments.system.config import CAMERAS, ExperimentConfig
from experiments.system.jobs import RunBusyError, RunManager, read_json, write_json


# This worker deliberately has no RoboCasa, OpenGL or network dependencies.
WORKER = r'''
import json
from pathlib import Path
import sys
import time
folder, mode = Path(sys.argv[1]), sys.argv[2]
def progress(status, **extra):
    temp = folder / 'progress.json.tmp'
    temp.write_text(json.dumps(dict(status=status, step=3, **extra)), encoding='utf-8')
    temp.replace(folder / 'progress.json')
print('fake worker started', flush=True)
(folder / 'ready').touch()
if mode == 'hold':
    deadline = time.monotonic() + 10
    while not (folder / 'cancel.request').exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    progress('cancelled' if (folder / 'cancel.request').exists() else 'failed')
elif mode == 'stubborn':
    time.sleep(20)
elif mode == 'fail':
    progress('failed', error='intentional worker failure')
    sys.exit(7)
elif mode == 'completed_then_fail':
    progress('completed', error=None)
    sys.exit(9)
elif mode == 'silent':
    pass
else:
    (folder / 'comparison.mp4').write_bytes(bytes(range(128)))
    (folder / 'metrics.jsonl').write_text('{"step":0}\n{"step":3}\n{"step":', encoding='utf-8')
    (folder / 'summary.json').write_text('{"steps_completed":3}', encoding='utf-8')
    progress('completed')
'''


def invalid_configs():
    for key, low, high in (("steps", 1, 2000), ("seed", 0, 2**32 - 1),
                           ("patch_seed", 0, 2**32 - 1), ("patch_size", 1, 256),
                           ("save_every", 1, 2000)):
        for value in (None, True, False, 1.0, "1", [], {}, low - 1, high + 1, float("nan")):
            yield {key: value}
    for key in ("env_id", "split", "camera_key", "attack_mode", "location", "defense", "policy"):
        for value in (None, True, 1, [], {}, "unknown"):
            yield {key: value}
    for value in (None, True, 3.0, 5.0, 7.0, "5", [], {}, 0, 4, 9):
        yield {"median_kernel": value}
    for key, high in (("alpha", 1), ("temporal_threshold", 255)):
        for value in (None, True, False, "0.5", [], {}, -0.1, high + 0.1,
                      float("nan"), float("inf"), -float("inf"), 10**500):
            yield {key: value}
    for value in (None, 0, "0,0", [], [0], [0, 0, 0], [True, 0], [0, 1.0], [0, {}], [-200, 0], [0, 200]):
        yield {"offset": value}
    yield {"unknown_option": 1}


class ConfigTests(unittest.TestCase):
    def test_valid_config_roundtrip_and_boundary_placements(self):
        for camera in CAMERAS:
            value = ExperimentConfig.from_dict({"camera_key": camera, "median_kernel": 7})
            self.assertEqual(ExperimentConfig.from_dict(json.loads(json.dumps(value.to_dict()))), value)
        for location in ("center", "top_left", "top_right", "bottom_left", "bottom_right"):
            value = ExperimentConfig.from_dict({"patch_size": 256, "location": location})
            self.assertEqual(value.offset, (0, 0))
            with self.assertRaises(ValueError):
                ExperimentConfig.from_dict({"patch_size": 256, "location": location, "offset": [1, 0]})
        for alpha, threshold in ((0, 0), (1, 255), (0.125, 12.5)):
            ExperimentConfig.from_dict({"alpha": alpha, "temporal_threshold": threshold,
                                        "seed": 2**32 - 1, "steps": 2000})
        ExperimentConfig.from_dict({"patch_size": 50, "location": "top_left", "offset": [206, 206]})

    def test_invalid_json_values_raise_validation_errors(self):
        for payload in (None, [], "bad", 1, True, *invalid_configs()):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                ExperimentConfig.from_dict(payload)

    def test_configuration_does_not_mutate_caller_or_defaults(self):
        payload = {"offset": [1, 2], "steps": 3}
        config = ExperimentConfig.from_dict(payload)
        self.assertEqual(payload["offset"], [1, 2])
        payload["offset"][0] = 9
        self.assertEqual(config.offset, (1, 2))
        self.assertEqual(ExperimentConfig().offset, (0, 0))


class ManagerFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="safety-console-tests-")
        self.root = Path(self.temporary.name)
        self.managers = []
        self.manager = self.make_manager()
        self.addCleanup(self.cleanup_workers)

    def make_manager(self, mode="hold", **options):
        manager = RunManager(self.root / "runs", self.root / "legacy", python=sys.executable,
                             command_factory=lambda folder: [sys.executable, "-u", "-c", WORKER, str(folder), mode],
                             cancel_timeout=options.pop("cancel_timeout", 0.2), **options)
        self.managers.append(manager)
        return manager

    def cleanup_workers(self):
        for manager in self.managers:
            manager.shutdown()
            process = manager.process
            if process is not None:
                try:
                    process.wait(timeout=2)
                except Exception:
                    process.kill()
                    process.wait(timeout=5)
            self.wait_until(lambda: manager.active_id is None)
        self.temporary.cleanup()

    def wait_until(self, condition, timeout=8):
        deadline = time.monotonic() + timeout
        while not condition():
            if time.monotonic() >= deadline:
                self.fail("timed out waiting for worker state")
            time.sleep(0.01)

    def wait_finished(self, manager, run_id):
        self.wait_until(lambda: manager.active_id is None)
        return manager.describe(run_id, detail=True)

    def record(self, run_id="saved", status="completed", legacy=False, progress=None):
        base = self.manager.legacy_dir if legacy else self.manager.runs_dir
        folder = base / run_id
        folder.mkdir(parents=True)
        if legacy:
            write_json(folder / "summary.json", {"steps_completed": 10, "steps_requested": 20})
        else:
            write_json(folder / "job.json", {"status": status, "created_at": "2026-11-09T12:00:00+08:00"})
        if progress is not None:
            write_json(folder / "progress.json", progress)
        return folder

    def symlink(self, link, target):
        try:
            link.symlink_to(target, target_is_directory=target.is_dir())
        except OSError as error:
            self.skipTest(f"symlink creation unavailable on this host: {error}")


class ManagerTests(ManagerFixture):
    def test_concurrent_start_allows_exactly_one_worker(self):
        barrier = threading.Barrier(2)

        def start():
            barrier.wait(timeout=3)
            try:
                return self.manager.start(ExperimentConfig.from_dict({"steps": 3}))
            except RunBusyError:
                return None

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: start(), range(2)))
        self.assertEqual(sum(value is not None for value in results), 1)
        run = next(value for value in results if value)
        self.assertEqual(len(self.manager.list_runs()), 1)
        self.assertEqual(self.manager.active_id, run["id"])
        self.manager.cancel(run["id"])
        self.assertEqual(self.wait_finished(self.manager, run["id"])["status"], "cancelled")

    def test_completed_worker_retains_artifacts_and_partial_jsonl(self):
        manager = self.make_manager("complete")
        run = manager.start(ExperimentConfig.from_dict({"steps": 3}))
        result = self.wait_finished(manager, run["id"])
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["live"])
        self.assertEqual(result["steps_completed"], 3)
        self.assertEqual(result["metrics"], [{"step": 0}, {"step": 3}])
        self.assertIn("fake worker started", result["log"])
        self.assertIn("comparison.mp4", [entry["name"] for entry in result["artifacts"]])
        self.assertEqual(read_json(manager.resolve(run["id"]) / "job.json")["returncode"], 0)
        with self.assertRaises(RunBusyError):
            manager.cancel(run["id"])
        next_run = manager.start(ExperimentConfig())
        self.assertNotEqual(next_run["id"], run["id"])
        self.wait_finished(manager, next_run["id"])

    def test_worker_failure_overrides_false_completed_progress(self):
        for mode, code in (("fail", 7), ("completed_then_fail", 9), ("silent", 0)):
            with self.subTest(mode=mode):
                manager = self.make_manager(mode)
                run = manager.start(ExperimentConfig())
                result = self.wait_finished(manager, run["id"])
                self.assertEqual(result["status"], "failed")
                self.assertTrue(result["error"])
                self.assertFalse(result["live"])
                job = read_json(manager.resolve(run["id"]) / "job.json")
                self.assertEqual(job["returncode"], code)
                self.assertIn("finished_at", job)

    def test_failed_spawn_preserves_record_and_releases_slot(self):
        self.manager.command_factory = lambda folder: [str(self.root / "missing-python.exe")]
        with self.assertRaises(OSError):
            self.manager.start(ExperimentConfig())
        self.assertIsNone(self.manager.active_id)
        result = self.manager.list_runs()[0]
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["error"])
        self.manager.command_factory = self.make_manager("complete").command_factory
        run = self.manager.start(ExperimentConfig())
        self.assertEqual(self.wait_finished(self.manager, run["id"])["status"], "completed")

    def test_cancel_is_idempotent_and_timeout_stops_stuck_worker(self):
        manager = self.make_manager("stubborn", cancel_timeout=0.15)
        run = manager.start(ExperimentConfig())
        folder = manager.resolve(run["id"])
        self.wait_until(lambda: (folder / "ready").exists())
        with mock.patch.object(manager, "_cancel_timeout", wraps=manager._cancel_timeout) as timer:
            self.assertEqual(manager.cancel(run["id"])["status"], "cancelling")
            self.assertEqual(manager.cancel(run["id"])["status"], "cancelling")
            result = self.wait_finished(manager, run["id"])
            self.assertEqual(timer.call_count, 1)
        self.assertEqual(result["status"], "cancelled")
        self.assertTrue((folder / "cancel.request").exists())
        self.assertTrue((folder / "worker.log").exists())
        self.assertNotEqual(read_json(folder / "job.json")["returncode"], 0)

    def test_shutdown_requests_cancellation_and_rejects_new_work(self):
        run = self.manager.start(ExperimentConfig())
        self.manager.shutdown()
        with self.assertRaises(RunBusyError):
            self.manager.start(ExperimentConfig())
        self.assertEqual(self.wait_finished(self.manager, run["id"])["status"], "cancelled")
        self.manager.shutdown()

    def test_second_manager_construction_never_changes_live_job(self):
        run = self.manager.start(ExperimentConfig())
        folder = self.manager.resolve(run["id"])
        before = (folder / "job.json").read_bytes()
        self.make_manager()
        self.assertEqual((folder / "job.json").read_bytes(), before)
        # Explicit recovery is also safe for this manager's own active worker.
        self.assertEqual(self.manager.recover_interrupted(), [])
        self.assertEqual((folder / "job.json").read_bytes(), before)

    def test_recovery_is_explicit_and_preserves_completed_and_legacy_results(self):
        stale = self.record("stale", "running", progress={"status": "running", "step": 4})
        done = self.record("done", "running", progress={"status": "completed", "step": 8})
        failed = self.record("failed", "running", progress={"status": "failed", "error": "boom"})
        self.record("stopped", "cancelling", progress={"status": "cancelled"})
        terminal = self.record("terminal", "failed")
        legacy = self.record("patch_attack_seed0", legacy=True)
        legacy_before, terminal_before = (legacy / "summary.json").read_bytes(), (terminal / "job.json").read_bytes()
        other = self.make_manager()
        self.assertEqual(read_json(stale / "job.json")["status"], "running")
        self.assertEqual(set(other.recover_interrupted()), {"stale", "done", "failed", "stopped"})
        self.assertEqual(read_json(stale / "job.json")["status"], "interrupted")
        self.assertEqual(read_json(done / "job.json")["status"], "completed")
        self.assertEqual(read_json(failed / "job.json")["error"], "boom")
        self.assertEqual((terminal / "job.json").read_bytes(), terminal_before)
        self.assertEqual((legacy / "summary.json").read_bytes(), legacy_before)
        self.assertEqual(other.describe("legacy_patch_attack_seed0")["steps_completed"], 10)
        self.assertEqual(other.recover_interrupted(), [])

    def test_broken_record_and_incomplete_event_does_not_break_history(self):
        folder = self.record()
        (folder / "job.json").write_text('[]', encoding="utf-8")
        (folder / "progress.json").write_text('null', encoding="utf-8")
        (folder / "summary.json").write_text('{broken', encoding="utf-8")
        (folder / "config.json").write_bytes(b'\xff\xfe\xff')
        (folder / "events.jsonl").write_text('{"type":"start"}\nnull\n[]\n{"unfinished":', encoding="utf-8")
        self.assertEqual(len(self.manager.list_runs()), 1)
        detail = self.manager.describe("saved", detail=True)
        self.assertEqual(detail["status"], "interrupted")
        self.assertEqual(detail["events"], [{"type": "start"}])
        self.assertEqual(detail["config"], {})
        self.assertEqual(self.manager.recover_interrupted(), [])

    def test_run_ids_cannot_escape_or_resolve_base_directory(self):
        self.record()
        self.manager.legacy_dir.mkdir(exist_ok=True)
        write_json(self.manager.legacy_dir / "summary.json", {"private": True})
        for value in ("../saved", "..", "", "saved/..", "saved\\..", "C:\\private", "legacy_", "x" * 101, None, 1):
            with self.subTest(run_id=value), self.assertRaises(KeyError):
                self.manager.resolve(value)
        self.assertEqual(self.manager.resolve("saved").parent, self.manager.runs_dir)

    def test_external_linked_run_is_not_read_or_recovered(self):
        outside = self.root / "outside"
        outside.mkdir()
        write_json(outside / "job.json", {"status": "running"})
        before = (outside / "job.json").read_bytes()
        self.symlink(self.manager.runs_dir / "escape", outside)
        with self.assertRaises(KeyError):
            self.manager.resolve("escape")
        self.assertEqual(self.manager.list_runs(), [])
        self.assertEqual(self.manager.recover_interrupted(), [])
        self.assertEqual((outside / "job.json").read_bytes(), before)

    def test_external_linked_artifacts_are_not_exposed_in_detail(self):
        folder = self.record()
        outside = self.root / "private.jsonl"
        outside.write_text('{"secret":true}\n', encoding="utf-8")
        self.symlink(folder / "metrics.jsonl", outside)
        self.symlink(folder / "worker.log", outside)
        self.symlink(folder / "config.json", outside)
        result = self.manager.describe("saved", detail=True)
        self.assertEqual(result["metrics"], [])
        self.assertEqual(result["config"], {})
        self.assertNotIn("secret", result["log"])
        self.assertNotIn("metrics.jsonl", [entry["name"] for entry in result["artifacts"]])


class ApiTests(ManagerFixture):
    def setUp(self):
        super().setUp()
        self.app = create_app(self.manager)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()
        self.headers = {"X-Console-Token": self.app.extensions["console_token"]}

    def test_health_bootstrap_and_frontend_build(self):
        self.assertEqual(self.client.get("/api/health").json["status"], "ok")
        bootstrap = self.client.get("/api/bootstrap")
        self.assertEqual(bootstrap.status_code, 200)
        self.assertEqual(bootstrap.json["token"], self.headers["X-Console-Token"])
        self.assertEqual(bootstrap.headers["Cache-Control"], "no-store")
        self.assertEqual(bootstrap.headers["X-Content-Type-Options"], "nosniff")
        self.assertIn("frame-ancestors 'none'", bootstrap.headers["Content-Security-Policy"])
        frontend = Path(__file__).resolve().parents[2] / "frontend/dist/index.html"
        self.assertTrue(frontend.is_file(), "frontend/dist/index.html must exist; run npm run build")
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('id="app"', html)
        asset = re.search(r'src="(/assets/[^\"]+\.js)"', html)
        self.assertIsNotNone(asset)
        asset_response = self.client.get(asset.group(1))
        self.assertEqual(asset_response.status_code, 200)
        asset_response.close()
        response.close()

    def test_invalid_configs_return_400_without_starting_process(self):
        with mock.patch.object(self.manager, "start") as start:
            for payload in (None, [], "bad", 1, True, *invalid_configs()):
                with self.subTest(payload=payload):
                    response = self.client.post("/api/runs", data=json.dumps(payload),
                                                content_type="application/json", headers=self.headers)
                    self.assertEqual(response.status_code, 400, response.get_data(as_text=True))
                    self.assertTrue(response.json["error"])
            malformed = self.client.post("/api/runs", data='{"steps":', content_type="application/json", headers=self.headers)
            self.assertEqual(malformed.status_code, 400)
            start.assert_not_called()
        self.assertEqual(self.manager.list_runs(), [])

    def test_token_origin_and_host_checks_prevent_worker_launch(self):
        cases = (
            ({}, None),
            ({"X-Console-Token": "wrong"}, None),
            (self.headers | {"Origin": "https://outside.example"}, None),
            (self.headers | {"Origin": "null"}, None),
            (self.headers | {"Origin": "http://localhost:9999"}, None),
            (self.headers, "http://outside.example"),
        )
        with mock.patch.object(self.manager, "start") as start:
            for headers, base in cases:
                with self.subTest(headers=headers, base=base):
                    response = self.client.post("/api/runs", json={}, headers=headers,
                                                **({"base_url": base} if base else {}))
                    self.assertEqual(response.status_code, 403)
            start.assert_not_called()
        self.assertEqual(self.client.get("/api/bootstrap", base_url="http://outside.example").status_code, 403)
        self.assertEqual(self.client.get("/api/health", base_url="http://127.0.0.1:8765").status_code, 200)
        self.assertEqual(self.client.get("/api/health", base_url="http://[::1]:8765").status_code, 200)

    def test_start_conflict_cancel_and_history_http_contract(self):
        response = self.client.post("/api/runs", json={"steps": 3}, headers=self.headers | {"Origin": "http://localhost"})
        self.assertEqual(response.status_code, 201)
        run_id = response.json["id"]
        self.assertEqual(self.client.post("/api/runs", json={}, headers=self.headers).status_code, 409)
        self.assertEqual(self.client.get("/api/runs").json["active_id"], run_id)
        self.assertEqual(self.client.get(f"/api/runs/{run_id}").json["config"]["steps"], 3)
        cancel = self.client.post(f"/api/runs/{run_id}/cancel", headers=self.headers)
        self.assertEqual(cancel.status_code, 200)
        self.wait_finished(self.manager, run_id)
        self.assertEqual(self.client.get(f"/api/runs/{run_id}").json["status"], "cancelled")
        self.assertEqual(self.client.post(f"/api/runs/{run_id}/cancel", headers=self.headers).status_code, 409)
        self.assertEqual(self.client.get("/api/runs/unknown").status_code, 404)
        self.assertEqual(self.client.post("/api/runs/unknown/cancel", headers=self.headers).status_code, 404)

    def test_large_payload_rejected_before_start(self):
        response = self.client.post("/api/runs", json={"payload": "x" * 20000}, headers=self.headers)
        self.assertEqual(response.status_code, 413)
        self.assertIsNone(self.manager.active_id)

    def test_video_range_seek_download_and_nested_frame(self):
        folder = self.record()
        video = bytes(range(256)) * 4
        (folder / "comparison.mp4").write_bytes(video)
        (folder / "frames").mkdir()
        (folder / "frames/0000_comparison.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        url = "/api/runs/saved/artifacts/comparison.mp4"
        response = self.client.get(url, headers={"Range": "bytes=20-29"})
        self.assertEqual(response.status_code, 206)
        self.assertEqual(response.data, video[20:30])
        self.assertEqual(response.headers["Content-Range"], "bytes 20-29/1024")
        self.assertEqual(response.mimetype, "video/mp4")
        response.close()
        response = self.client.get(url, headers={"Range": "bytes=-8"})
        self.assertEqual(response.status_code, 206)
        self.assertEqual(response.data, video[-8:])
        response.close()
        self.assertEqual(self.client.get(url, headers={"Range": "bytes=9999-"}).status_code, 416)
        response = self.client.get(url + "?download=1")
        self.assertIn("attachment", response.headers["Content-Disposition"])
        self.assertEqual(response.data, video)
        response.close()
        response = self.client.get("/api/runs/saved/artifacts/frames/0000_comparison.png")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "image/png")
        response.close()

    def test_path_traversal_and_disallowed_artifacts_return_404(self):
        folder = self.record()
        (self.manager.runs_dir / "secret.json").write_text('{"secret":true}', encoding="utf-8")
        (folder / "payload.html").write_text("unsafe", encoding="utf-8")
        (folder / "cancel.request").touch()
        for filename in ("../secret.json", "%2e%2e/secret.json", "..%5Csecret.json", "payload.html",
                         "cancel.request", "missing.mp4", "C:%5CWindows%5Cwin.ini"):
            with self.subTest(filename=filename):
                response = self.client.get("/api/runs/saved/artifacts/" + filename)
                self.assertEqual(response.status_code, 404)
                self.assertNotIn(b'"secret":true', response.data)
        for path in ("/api/runs/%2e%2e", "/api/runs/saved%5c..", "/api/runs/legacy_"):
            self.assertEqual(self.client.get(path).status_code, 404)

    def test_artifact_symlink_cannot_read_outside_run(self):
        folder = self.record()
        outside = self.root / "secret.json"
        outside.write_text('{"secret":true}', encoding="utf-8")
        self.symlink(folder / "escape.json", outside)
        self.assertEqual(self.client.get("/api/runs/saved/artifacts/escape.json").status_code, 404)


class DefenseMonitorTests(unittest.TestCase):
    def test_defense_filters_only_supplied_pixels_without_mutation(self):
        image = np.full((9, 9, 3), 40, np.uint8)
        image[4, 4] = 255
        original = image.copy()
        filtered = ImageDefense("median", 3)(image)
        self.assertTrue(np.all(filtered == 40))
        np.testing.assert_array_equal(image, original)
        self.assertFalse(np.shares_memory(filtered, image))
        untouched = ImageDefense("none")(image)
        np.testing.assert_array_equal(untouched, image)
        self.assertFalse(np.shares_memory(untouched, image))
        # No reference image or attack mask is passed to this interface.
        altered = original.copy()
        altered[:] = 100
        self.assertTrue(np.all(ImageDefense("median", 3)(altered) == 100))
        with self.assertRaises(TypeError):
            ImageDefense()(image.astype(np.float32))

    def test_defense_wrapper_never_reads_clean_reference(self):
        class CameraEnv(gym.Env):
            observation_space = gym.spaces.Dict({"camera": gym.spaces.Box(0, 255, (9, 9, 3), np.uint8)})
            action_space = gym.spaces.Box(-1, 1, (1,), np.float32)

            @property
            def last_clean_image(self):
                raise AssertionError("defense accessed clean reference")

            def reset(self, *, seed=None, options=None):
                self.image = np.full((9, 9, 3), 20, np.uint8)
                self.image[4, 4] = 255
                return {"camera": self.image, "state": "preserved"}, {"success": False}

            def step(self, action):
                observation, info = self.reset()
                return observation, 0.5, False, True, info

        env = CameraEnv()
        wrapped = DefenseWrapper(env, ImageDefense("median", 3), "camera")
        observation, info = wrapped.reset()
        self.assertEqual(observation["state"], "preserved")
        self.assertEqual(info, {"success": False})
        self.assertTrue(np.all(observation["camera"] == 20))
        self.assertTrue(np.all(env.image[4, 4] == 255))
        self.assertFalse(np.shares_memory(wrapped.last_input_image, env.image))
        observation, reward, terminated, truncated, info = wrapped.step(np.zeros(1, np.float32))
        self.assertEqual((reward, terminated, truncated), (0.5, False, True))
        self.assertTrue(np.all(observation["camera"] == 20))
        self.assertGreaterEqual(wrapped.last_latency_ms, 0)

    def test_temporal_monitor_owns_frame_copy_and_avoids_uint8_wraparound(self):
        monitor = TemporalImageMonitor(12)
        frame = np.zeros((4, 4, 3), np.uint8)
        self.assertEqual(monitor.update(frame), {"temporal_mean_abs_delta": None, "temporal_alert": False})
        frame[:] = 255  # The previously observed frame must be owned by monitor.
        result = monitor.update(frame)
        self.assertEqual(result["temporal_mean_abs_delta"], 255)
        self.assertTrue(result["temporal_alert"])
        self.assertTrue(np.all(frame == 255))
        self.assertFalse(monitor.update(frame)["temporal_alert"])
        result = monitor.update(np.full_like(frame, 243))
        self.assertEqual(result["temporal_mean_abs_delta"], 12)
        self.assertFalse(result["temporal_alert"])

    def test_action_guard_rejects_invalid_actions_without_mutation(self):
        space = gym.spaces.Dict({"arm": gym.spaces.Box(-1, 1, (2,), np.float32)})
        guard = ActionGuard(space)
        valid = {"arm": np.array([0.25, -0.5], np.float32)}
        before = valid["arm"].copy()
        self.assertEqual(guard.check(valid), (True, None))
        np.testing.assert_array_equal(valid["arm"], before)
        for action in (None, {}, {"other": before}, {"arm": np.zeros(3, np.float32)},
                       {"arm": np.array([float("nan"), 0], np.float32)},
                       {"arm": np.array([float("inf"), 0], np.float32)},
                       {"arm": np.array([2, 0], np.float32)},
                       {"arm": np.array(["x", "y"])}, {"arm": np.array([None, None])},
                       {"arm": [[0], [0, 1]]}):
            with self.subTest(action=action):
                accepted, reason = guard.check(action)
                self.assertFalse(accepted)
                self.assertIsInstance(reason, str)


if __name__ == "__main__":
    unittest.main()
