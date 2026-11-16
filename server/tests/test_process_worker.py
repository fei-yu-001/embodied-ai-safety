"""Real subprocess worker: boot, command confirmation, and crash handling."""

import io
import tempfile
import threading
import unittest
from unittest.mock import Mock

from server.infrastructure import simulation_worker
from server.infrastructure.process_worker import ProcessWorker, WorkerStartError
from server.infrastructure.worker_protocol import ProtocolError, decode, encode


class ProtocolTests(unittest.TestCase):
    def test_roundtrip_and_version_check(self):
        line = encode({"type": "command", "command": "step", "command_id": "c-1"})
        self.assertEqual(decode(line)["command_id"], "c-1")
        with self.assertRaises(ProtocolError):
            decode('{"type":"command","protocol":999}')

    def test_oversized_and_non_object_messages_are_rejected(self):
        with self.assertRaises(ProtocolError):
            encode(["not", "an", "object"])
        with self.assertRaises(ProtocolError):
            encode({"type": "command", "command_id": "x" * 1_000_001})


class StateOnlyBackendTests(unittest.TestCase):
    def test_step_advances_and_reset_starts_a_new_episode(self):
        source = io.StringIO(
            '{"type":"load","session_id":"sess_a","protocol":1}\n'
            '{"type":"command","command":"step","command_id":"c-1","protocol":1}\n'
            '{"type":"command","command":"reset","command_id":"c-2","protocol":1}\n'
        )
        sink = io.StringIO()
        self.assertEqual(simulation_worker.run(source, sink), 0)
        messages = [decode(line) for line in sink.getvalue().splitlines()]
        self.assertEqual(messages[0]["type"], "worker_booted")
        self.assertFalse(messages[0]["loads_physics"])
        self.assertEqual([m["type"] for m in messages[1:]], ["loaded", "applied", "applied"])
        self.assertEqual(messages[2]["state"], "PAUSED")
        self.assertEqual(messages[3]["state"], "PAUSED")

    def test_unknown_command_is_reported_without_crashing(self):
        source = io.StringIO('{"type":"command","command":"jump","command_id":"c-9","protocol":1}\n')
        sink = io.StringIO()
        simulation_worker.run(source, sink)
        messages = [decode(line) for line in sink.getvalue().splitlines()]
        self.assertEqual(messages[-1]["type"], "error")


class WorkerLifecycleTests(unittest.TestCase):
    def test_stop_shutdown_eof_and_failed_load_release_backend(self):
        for ending in ("stop", "shutdown", "eof", "load_failure"):
            with self.subTest(ending=ending):
                backend = Mock(name="backend")
                backend.name = "test"
                backend.loads_physics = True
                backend.load.return_value = {}
                backend.apply.return_value = {"state": "CLOSED"}
                if ending == "load_failure":
                    backend.load.side_effect = MemoryError("bad allocation")
                messages = [{"type": "load", "session_id": "sess_lifecycle"}]
                if ending == "stop":
                    messages += [{"type": "command", "command": "stop", "command_id": "end"},
                                 {"type": "command", "command": "step", "command_id": "late"}]
                elif ending == "shutdown":
                    messages += [{"type": "shutdown"}]
                sink = io.StringIO()
                result = simulation_worker.run(io.StringIO("".join(encode(m) for m in messages)), sink,
                                               backend_factory=lambda *_: backend)
                backend.close.assert_called_once_with()
                self.assertEqual(result, 1 if ending == "load_failure" else 0)
                if ending == "stop":
                    backend.apply.assert_called_once_with("stop", {})
                    self.assertEqual(decode(sink.getvalue().splitlines()[-1])["state"], "CLOSED")


class ProcessWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="prism-worker-")
        self.lost = []
        self.applied = []
        self.applied_event = threading.Event()
        self.worker = ProcessWorker(
            self.temporary.name,
            on_applied=lambda session, command, detail: (self.applied.append((session, command)), self.applied_event.set()),
            on_lost=lambda session, reason: self.lost.append((session, reason)),
        )

    def tearDown(self):
        for session in ("sess_proc", "sess_crash"):
            self.worker.crash(session)
        self.temporary.cleanup()

    def test_command_is_confirmed_by_the_child_process(self):
        self.worker.start("sess_proc")
        self.assertTrue(self.worker.alive("sess_proc"))
        self.worker.submit("sess_proc", "step", "cmd-1")
        self.assertTrue(self.applied_event.wait(15), "子进程必须回传 applied")
        self.assertEqual(self.applied, [("sess_proc", "cmd-1")])

    def test_stop_exits_the_real_child_and_preserves_closed_session(self):
        import time
        from datetime import datetime, timedelta, timezone
        from server.application.session_service import SessionService
        from server.domain.session_control import CommandEnvelope
        service = SessionService(self.temporary.name, self.worker)
        self.worker.on_loaded = service.mark_ready
        self.worker.on_applied = service.observe_applied
        self.worker.on_lost = service.crash_worker
        created = service.create()
        process = self.worker._processes[created["id"]]
        now = datetime.now(timezone.utc)
        try:
            service.submit(CommandEnvelope(command_id="finish", session_id=created["id"],
                fencing_token=created["fencing_token"], command="stop", issued_at=now,
                expires_at=now + timedelta(seconds=30), expected_state_version=created["state_version"]))
            self.assertEqual(process.wait(timeout=15), 0)
            for _ in range(100):
                if service.get(created["id"])["state"] == "CLOSED":
                    break
                time.sleep(.01)
            self.assertFalse(self.worker.alive(created["id"]))
            self.assertEqual(service.get(created["id"])["state"], "CLOSED")
        finally:
            self.worker.crash(created["id"])

    def test_capacity_limit_refuses_second_live_worker_and_releases_after_exit(self):
        self.worker.max_sessions = 1
        self.worker.start("sess_proc")
        with self.assertRaisesRegex(WorkerStartError, "会话上限"):
            self.worker.start("sess_crash")
        self.assertNotIn("sess_crash", self.worker._processes)
        self.worker.crash("sess_proc")
        self.worker.start("sess_crash")
        self.assertTrue(self.worker.alive("sess_crash"))

    def test_killed_worker_reports_loss_and_rejects_new_commands(self):
        self.worker.start("sess_crash")
        process = self.worker._processes["sess_crash"]
        process.kill()
        process.wait(timeout=10)
        deadline = threading.Event()
        for _ in range(50):
            if self.lost:
                break
            deadline.wait(0.1)
        self.assertTrue(self.lost, "子进程死亡后必须触发 on_lost")
        self.assertFalse(self.worker.alive("sess_crash"))
        with self.assertRaises(WorkerStartError):
            self.worker.submit("sess_crash", "step", "cmd-2")


if __name__ == "__main__":
    unittest.main()
