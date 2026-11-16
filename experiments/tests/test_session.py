"""Resident-session API contracts without MuJoCo, subprocesses or a live server."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from experiments.system.app import create_app
from experiments.system.config import ExperimentConfig
from experiments.system.jobs import RunBusyError, RunManager, read_json, write_json
from experiments.system.streaming import stream_frames


class SessionFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="safety-session-tests-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manager = RunManager(self.root / "runs", self.root / "legacy")
        self.run_id = "session_fixture"
        self.folder = self.manager.runs_dir / self.run_id
        self.folder.mkdir()
        self.command_path = self.folder / "control.request.json"
        self.set_job("loading")
        write_json(self.folder / "config.json", ExperimentConfig.from_dict({}).to_dict())
        self.set_progress("ready")
        # Only liveness is needed: control() must not spawn or step a simulator.
        self.process = mock.Mock(spec=["poll"])
        self.process.poll.return_value = None
        self.manager.process = self.process
        self.manager.active_id = self.run_id
        self.app = create_app(self.manager)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()
        self.headers = {"X-Console-Token": self.app.extensions["console_token"]}
        self.control_url = f"/api/runs/{self.run_id}/control"
        self.stream_url = f"/api/runs/{self.run_id}/stream"

    def set_job(self, status):
        write_json(self.folder / "job.json", {"id": self.run_id, "status": status})

    def set_progress(self, status, seq=0, **extra):
        write_json(self.folder / "progress.json", {
            "status": status, "step": 0, "episode_step": 0, "episode": 1,
            "control_seq": seq, "episode_done": False, **extra,
        })

    def control(self, command, **options):
        return self.client.post(self.control_url, json={"command": command},
                                headers=options.pop("headers", self.headers), **options)

    def assert_rejected(self, command, code=409, **options):
        before = self.command_path.read_bytes() if self.command_path.exists() else None
        response = self.control(command, **options)
        self.assertEqual(response.status_code, code, response.get_data(as_text=True))
        self.assertTrue(response.json["error"])
        after = self.command_path.read_bytes() if self.command_path.exists() else None
        self.assertEqual(after, before, "a rejected command must not replace a pending request")


class SessionConfigTests(SessionFixture):
    def test_defaults_load_resident_session_without_recording(self):
        config = ExperimentConfig.from_dict({})
        self.assertEqual((config.run_mode, config.record_video, config.steps), ("session", False, 1200))
        defaults = self.client.get("/api/bootstrap").json["defaults"]
        self.assertEqual(defaults["run_mode"], "session")
        self.assertIs(defaults["record_video"], False)
        self.assertEqual(defaults["steps"], 1200)
        for mode in ("session", "batch"):
            for recording in (False, True):
                value = ExperimentConfig.from_dict({"run_mode": mode, "record_video": recording})
                self.assertEqual(ExperimentConfig.from_dict(json.loads(json.dumps(value.to_dict()))), value)

    def test_invalid_mode_and_recording_flag_return_400_before_launch(self):
        invalid = {
            "run_mode": (None, True, 0, [], {}, "", "live", "SESSION"),
            "record_video": (None, 0, 1, 0.0, "false", "true", [], {}),
        }
        with mock.patch.object(self.manager, "start") as start:
            for key, values in invalid.items():
                for value in values:
                    with self.subTest(key=key, value=value):
                        with self.assertRaises(ValueError):
                            ExperimentConfig.from_dict({key: value})
                        response = self.client.post("/api/runs", json={key: value}, headers=self.headers)
                        self.assertEqual(response.status_code, 400)
                        self.assertTrue(response.json["error"])
            start.assert_not_called()


class SessionControlTests(SessionFixture):
    def test_loading_rejects_controls_until_worker_is_ready(self):
        self.set_progress("loading")
        for command in ("resume", "pause", "step", "reset"):
            with self.subTest(command=command):
                self.assert_rejected(command)
        self.assertFalse(self.command_path.exists())
        self.set_progress("ready")
        self.assertEqual(self.control("step").status_code, 202)

    def test_batch_and_legacy_configuration_reject_session_controls(self):
        for config in (ExperimentConfig.from_dict({"run_mode": "batch"}).to_dict(), {"steps": 100}):
            write_json(self.folder / "config.json", config)
            for status, command in (("ready", "resume"), ("ready", "step"),
                                    ("running", "pause"), ("paused", "reset")):
                with self.subTest(config=config, status=status, command=command):
                    self.set_progress(status)
                    self.assert_rejected(command)
        self.assertFalse(self.command_path.exists())

    def test_single_step_is_pending_until_worker_acknowledges_sequence(self):
        self.set_progress("ready", seq=7)
        original_progress = (self.folder / "progress.json").read_bytes()
        response = self.control("step")
        self.assertEqual(response.status_code, 202)
        self.assertTrue(response.json["live"])
        self.assertEqual(response.json["status"], "ready")
        self.assertTrue(response.json["command_pending"])
        self.assertEqual(read_json(self.command_path), {"seq": 8, "command": "step"})
        self.assertEqual((self.folder / "progress.json").read_bytes(), original_progress)
        self.assert_rejected("step")
        self.assert_rejected("reset")
        # Simulate a worker acknowledgement, not a physical step in this test.
        self.set_progress("paused", seq=8, step=1, episode_step=1)
        detail = self.client.get(f"/api/runs/{self.run_id}").json
        self.assertFalse(detail["command_pending"])
        self.assertEqual(detail["steps_completed"], 1)
        self.assertEqual(self.control("step").status_code, 202)
        self.assertEqual(read_json(self.command_path), {"seq": 9, "command": "step"})

    def test_resume_pause_reset_and_resume_follow_acknowledged_states(self):
        for seq, command, current, acknowledged in (
            (1, "resume", "ready", "running"),
            (2, "pause", "running", "paused"),
            (3, "reset", "paused", "paused"),
            (4, "resume", "paused", "running"),
        ):
            with self.subTest(command=command, state=current):
                self.assertEqual(self.manager.describe(self.run_id)["status"], current)
                response = self.control(command)
                self.assertEqual(response.status_code, 202, response.get_data(as_text=True))
                self.assertEqual(response.json["status"], current)
                self.assertTrue(response.json["command_pending"])
                self.assertEqual(read_json(self.command_path), {"seq": seq, "command": command})
                self.set_progress(acknowledged, seq=seq)
                self.assertFalse(self.manager.describe(self.run_id)["command_pending"])

    def test_incompatible_controls_cannot_implicitly_change_running_state(self):
        for status, commands in (("ready", ("pause",)), ("paused", ("pause",)),
                                 ("running", ("resume", "step", "reset"))):
            self.set_progress(status)
            for command in commands:
                with self.subTest(status=status, command=command):
                    self.assert_rejected(command)
                    self.assertEqual(self.manager.describe(self.run_id)["status"], status)

    def test_concurrent_controls_queue_only_one_unacknowledged_command(self):
        barrier = threading.Barrier(2)

        def send(command):
            barrier.wait(timeout=3)
            try:
                return self.manager.control(self.run_id, {"command": command})
            except RunBusyError:
                return None

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(send, ("step", "reset")))
        self.assertEqual(sum(result is not None for result in results), 1)
        self.assertTrue(next(result for result in results if result)["command_pending"])
        queued = read_json(self.command_path)
        self.assertEqual(queued["seq"], 1)
        self.assertIn(queued["command"], ("step", "reset"))

    def test_terminated_or_cancelling_jobs_reject_all_controls(self):
        # Stale ready progress must never override the authoritative job status.
        for status in ("completed", "cancelled", "failed", "interrupted", "cancelling"):
            self.set_job(status)
            for command in ("resume", "pause", "step", "reset"):
                with self.subTest(status=status, command=command):
                    self.assert_rejected(command)

    def test_exited_or_unowned_worker_rejects_controls_and_has_no_pending_command(self):
        write_json(self.command_path, {"seq": 1, "command": "step"})
        for state in ("exited", "unowned", "missing"):
            with self.subTest(state=state):
                self.manager.process = None if state == "missing" else self.process
                self.process.poll.return_value = 0 if state == "exited" else None
                self.manager.active_id = "another_run" if state == "unowned" else self.run_id
                detail = self.manager.describe(self.run_id)
                self.assertFalse(detail["live"])
                self.assertFalse(detail["command_pending"])
                for command in ("resume", "pause", "step", "reset"):
                    self.assert_rejected(command)

    def test_finished_episode_requires_reset_before_step_or_resume(self):
        self.set_progress("paused", episode_done=True, episode_step=20, step=20)
        self.assert_rejected("step")
        self.assert_rejected("resume")
        response = self.control("reset")
        self.assertEqual(response.status_code, 202)
        self.assertTrue(response.json["episode_done"])
        self.assertTrue(response.json["command_pending"])
        self.set_progress("paused", seq=1, episode=2, step=20)
        self.assertEqual(self.control("resume").status_code, 202)

    def test_invalid_control_payloads_return_400_without_writing_requests(self):
        for payload in (None, [], {}, "step", {"command": "run"}, {"command": None},
                        {"command": True}, {"command": []}, {"command": {}},
                        {"command": "step", "seq": 99}):
            with self.subTest(payload=payload):
                response = self.client.post(self.control_url, data=json.dumps(payload),
                                            content_type="application/json", headers=self.headers)
                self.assertEqual(response.status_code, 400)
                self.assertTrue(response.json["error"])
        response = self.client.post(self.control_url, data='{"command":',
                                    content_type="application/json", headers=self.headers)
        self.assertEqual(response.status_code, 400)
        self.assertFalse(self.command_path.exists())
        response = self.client.post("/api/runs/missing/control", json={"command": "step"}, headers=self.headers)
        self.assertEqual(response.status_code, 404)

    def test_control_requires_valid_token_and_same_origin(self):
        for headers in ({}, {"X-Console-Token": "wrong"},
                        self.headers | {"Origin": "https://outside.example"},
                        self.headers | {"Origin": "null"},
                        self.headers | {"Origin": "http://localhost:9999"}):
            with self.subTest(headers=headers):
                self.assert_rejected("step", code=403, headers=headers)
        self.assert_rejected("step", code=403, base_url="http://outside.example")
        response = self.control("step", headers=self.headers | {"Origin": "http://localhost"})
        self.assertEqual(response.status_code, 202)


class SessionStreamTests(SessionFixture):
    @staticmethod
    def multipart(data):
        return (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                + str(len(data)).encode("ascii") + b"\r\n\r\n" + data + b"\r\n")

    def test_all_views_and_default_scene_return_matching_multipart_bytes(self):
        views = {"scene": "scene.jpg", "comparison": "preview.jpg", "clean": "clean.jpg",
                 "attacked": "attacked.jpg", "defended": "defended.jpg", "left": "left.jpg",
                 "right": "right.jpg", "wrist": "wrist.jpg"}
        self.set_job("completed")  # A saved run yields one frame and then closes.
        for filename in views.values():
            # Test byte-preserving transport; JPEG rendering is the worker's job.
            (self.folder / filename).write_bytes(b"\xff\xd8" + filename.encode("ascii") + b"\xff\xd9")
        for view, filename in [(None, "scene.jpg"), *views.items()]:
            with self.subTest(view=view):
                response = self.client.get(self.stream_url, query_string={} if view is None else {"view": view},
                                           buffered=True)
                try:
                    self.assertEqual(response.status_code, 200)
                    self.assertIn("multipart/x-mixed-replace; boundary=frame", response.content_type)
                    self.assertEqual(response.headers["X-Accel-Buffering"], "no")
                    self.assertEqual(response.headers["Cache-Control"], "no-store")
                    self.assertEqual(response.data, self.multipart((self.folder / filename).read_bytes()))
                finally:
                    response.close()

    def test_invalid_views_cannot_select_arbitrary_files(self):
        for view in ("", "SCENE", "scene.jpg", "../secret.jpg", "..\\secret.jpg", "unknown"):
            with self.subTest(view=view):
                response = self.client.get(self.stream_url, query_string={"view": view})
                self.assertEqual(response.status_code, 400)
                self.assertTrue(response.json["error"])
        self.assertEqual(self.client.get("/api/runs/missing/stream").status_code, 404)

    def test_stream_observes_atomic_frame_replacement_without_advancing_session(self):
        self.set_progress("paused", seq=3, step=5)
        progress_before = (self.folder / "progress.json").read_bytes()
        frame = self.folder / "scene.jpg"
        first, second = b"first JPEG snapshot", b"second JPEG snapshot with a different size"
        frame.write_bytes(first)
        frames = stream_frames(self.folder, "scene.jpg")
        self.addCleanup(frames.close)
        with mock.patch("experiments.system.streaming.sleep"), \
                mock.patch("experiments.system.streaming.monotonic", return_value=10.0):
            self.assertEqual(next(frames), self.multipart(first))
            temporary = self.folder / "scene.tmp.jpg"
            temporary.write_bytes(second)
            temporary.replace(frame)
            self.assertEqual(next(frames), self.multipart(second))
            self.set_job("cancelled")
            with self.assertRaises(StopIteration):
                next(frames)
        self.assertEqual((self.folder / "progress.json").read_bytes(), progress_before)
        self.assertFalse(self.command_path.exists())

    def test_finished_run_without_snapshot_closes_empty_stream(self):
        self.set_job("failed")
        response = self.client.get(self.stream_url, buffered=True)
        try:
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.data, b"")
        finally:
            response.close()

    def test_stream_never_reads_a_snapshot_resolving_outside_run(self):
        self.set_job("completed")
        external = self.root / "private.jpg"
        external.write_bytes(b"private external image")
        candidate = self.folder / "scene.jpg"
        candidate.write_bytes(b"placeholder for a linked snapshot")
        original_resolve = Path.resolve

        def resolve(path, *args, **kwargs):
            # Model an external link target without Windows symlink privileges.
            if path == candidate:
                return original_resolve(external)
            return original_resolve(path, *args, **kwargs)

        with mock.patch.object(Path, "resolve", autospec=True, side_effect=resolve), \
                mock.patch.object(Path, "read_bytes", side_effect=AssertionError("external snapshot was read")) as read:
            response = self.client.get(self.stream_url, query_string={"view": "scene"}, buffered=True)
            try:
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.data, b"")
                read.assert_not_called()
            finally:
                response.close()


if __name__ == "__main__":
    unittest.main()
