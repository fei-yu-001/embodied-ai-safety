"""Command identity, reconnect, and crash recovery without MuJoCo."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from server.application.session_service import SessionService, WorkerUnavailable
from server.domain.session_control import CommandEnvelope, ControlError, SessionControl


NOW = datetime(2026, 11, 13, 1, 0, tzinfo=timezone.utc)


class MemoryWorker:
    def __init__(self):
        self.submissions = []
        self.alive_ids = set()
        self.info = {}

    def start(self, session_id, config=None):
        self.alive_ids.add(session_id)
        self.info[session_id] = {"backend": "fake", "loads_physics": False, "state": {"episode": 1}}

    def submit(self, session_id, command, command_id, args=None):
        if session_id not in self.alive_ids:
            raise WorkerUnavailable(session_id)
        self.submissions.append((session_id, command, command_id))

    def alive(self, session_id):
        return session_id in self.alive_ids

    def keepalive(self, session_id, lease_id):
        return session_id in self.alive_ids

    def query(self, session_id, name, timeout=3.0, args=None):
        if session_id not in self.alive_ids:
            raise WorkerUnavailable(session_id)
        return {
            "frame": {"world": "mujoco_world", "units": "m", "up": "+z", "quaternion": "wxyz"},
            "available": {"eef": True, "base": True, "joints": True, "objects": True},
            "eef": {"position": [0.0, 0.0, 1.0], "quaternion": [1.0, 0.0, 0.0, 0.0]},
            "base": {"position": [0.0, 0.0, 0.0]},
            "joints": [0.0] * 7,
            "objects": [{"name": "cup", "position": [0.1, 0.2, 0.9]}],
            "step": 0,
            "episode": 1,
            "sim_seconds": 0.0,
        }

    def crash(self, session_id):
        self.alive_ids.discard(session_id)
        self.info.pop(session_id, None)


class SessionControlTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="prism-session-")
        self.worker = MemoryWorker()
        self.service = SessionService(Path(self.temporary.name), self.worker, clock=lambda: NOW)

    def tearDown(self):
        self.temporary.cleanup()

    def test_duplicate_step_is_not_sent_to_worker_again(self):
        session = self._ready_session()
        first = self.service.submit(self._envelope(session, "step", "step-1"))
        self.service.observe_applied(session["id"], "step-1")
        second = self.service.submit(self._envelope(session, "resume", "step-1"))

        self.assertEqual(first.acknowledgement, "accepted")
        self.assertEqual(second.acknowledgement, "applied")
        self.assertEqual(second.command_id, "step-1")
        self.assertEqual(self.worker.submissions, [(session["id"], "step", "step-1")])
        self.assertEqual(self.service.get(session["id"])["applied_steps"], 1)

    def test_reconnect_replays_only_events_after_cursor(self):
        session = self._ready_session()
        seen = self.service.get(session["id"])["last_event_seq"]
        event = self.service.record_safety(session["id"], "动作被拒绝，未推进物理")
        recovered = SessionService(Path(self.temporary.name), MemoryWorker(), clock=lambda: NOW)

        replay = recovered.events_after(session["id"], seen)
        self.assertEqual([item["seq"] for item in replay], [event["seq"]])
        self.assertEqual(replay[0]["type"], "safety_event")
        self.assertEqual(replay[0]["message"], "动作被拒绝，未推进物理")

    def test_worker_crash_keeps_events_and_does_not_replay_step(self):
        session = self._ready_session()
        accepted = self.service.submit(self._envelope(session, "step", "step-crash"))
        self.assertEqual(accepted.acknowledgement, "accepted")
        crashed = self.service.crash_worker(session["id"])
        recovered = SessionService(Path(self.temporary.name), MemoryWorker(), clock=lambda: NOW)
        replay = recovered.submit(self._envelope(session, "step", "step-crash"))

        self.assertEqual(crashed["state"], "INTERRUPTED")
        self.assertEqual(replay.acknowledgement, "rejected")
        self.assertEqual(replay.code, "worker_lost")
        self.assertEqual(self.worker.submissions, [(session["id"], "step", "step-crash")])
        self.assertTrue(recovered.events_after(session["id"], 0))

    def test_expired_and_stale_commands_do_not_change_state(self):
        session = self._ready_session()
        version = session["state_version"]
        expired = self.service.submit(self._envelope(session, "step", "old", expires_at=NOW - timedelta(seconds=1)))
        stale = self.service.submit(self._envelope(session, "step", "stale", expected_state_version=version + 9))

        self.assertEqual(expired.code, "expired")
        self.assertEqual(stale.code, "state_version_conflict")
        self.assertEqual(self.worker.submissions, [])
        self.assertEqual(self.service.get(session["id"])["state"], "READY")

    def test_jog_is_allowed_when_paused_and_applies_without_state_change(self):
        session = self._ready_session()
        ack = self.service.submit(self._envelope(session, "jog", "jog-1", args={"delta_m": [0.01, 0, 0]}))
        self.service.observe_applied(session["id"], "jog-1")
        self.assertEqual(ack.acknowledgement, "accepted")
        self.assertEqual(self.service.get(session["id"])["state"], "PAUSED")
        self.assertEqual(self.worker.submissions, [(session["id"], "jog", "jog-1")])

    def test_args_are_refused_for_commands_that_do_not_take_them(self):
        session = self._ready_session()
        with self.assertRaises(ControlError) as raised:
            self.service.submit(self._envelope(session, "step", "step-args", args={"delta_m": [0.01, 0, 0]}))
        self.assertEqual(raised.exception.code, "invalid_args")
        self.assertEqual(self.worker.submissions, [])

    def test_edit_is_refused_while_the_session_is_running(self):
        session = self._ready_session()
        lease = self.service.acquire_lease(session["id"], "browser-a", 1500)
        self.service.submit(self._envelope(session, "jog_start", "manual-1", args={"lease_id": lease["lease_id"]}))
        self.service.observe_applied(session["id"], "manual-1")
        ack = self.service.submit(
            self._envelope(session, "edit", "edit-1", args={"target": "cup", "kind": "mass", "values": 1.0})
        )
        self.assertEqual(ack.acknowledgement, "rejected")
        self.assertEqual(ack.code, "state_conflict")
        self.assertNotIn((session["id"], "edit", "edit-1"), self.worker.submissions)

    def test_edit_is_allowed_when_paused(self):
        session = self._ready_session()
        ack = self.service.submit(self._envelope(session, "edit", "edit-2", args={"target": "cup", "kind": "mass", "values": 1.0}))
        self.service.observe_applied(session["id"], "edit-2")
        self.assertEqual(ack.acknowledgement, "accepted")
        self.assertEqual(self.service.get(session["id"])["state"], "PAUSED")
        self.assertEqual(self.worker.submissions[-1], (session["id"], "edit", "edit-2"))

    def test_restore_does_not_execute_folded_commands(self):
        control = SessionControl("sess_restore", "token")
        control.create_event()
        restored = SessionControl.from_events(control.events)
        self.assertEqual(restored.snapshot()["state"], "QUEUED")
        self.assertEqual(restored.applied_steps, 0)

    def _ready_session(self):
        session = self.service.create()
        return self.service.mark_ready(session["id"])

    def _envelope(self, session, command, command_id, expires_at=None, expected_state_version=None, args=None):
        return CommandEnvelope(
            command_id=command_id,
            session_id=session["id"],
            fencing_token=session["fencing_token"],
            command=command,
            issued_at=NOW,
            expires_at=expires_at or (NOW + timedelta(seconds=30)),
            expected_state_version=expected_state_version,
            args=args or {},
        )


if __name__ == "__main__":
    unittest.main()
