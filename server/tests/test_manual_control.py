"""Control lease and worker-owned manual motion."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import threading
import time
import unittest

from server.application.session_service import SessionService
from server.domain.control_lease import ControlLease, LeaseError
from server.domain.session_control import CommandEnvelope, ControlError
from server.infrastructure.manual_motion import ManualMotion
from server.tests.test_session_control import MemoryWorker, NOW


class LeaseTests(unittest.TestCase):
    def setUp(self):
        self.now = NOW
        self.lease = ControlLease(clock=lambda: self.now)

    def test_second_holder_is_refused_while_the_first_is_live(self):
        first = self.lease.acquire("browser-a", 1500)
        with self.assertRaises(LeaseError) as raised:
            self.lease.acquire("browser-b", 1500)
        self.assertEqual(raised.exception.code, "lease_held")
        self.assertEqual(self.lease.acquire("browser-a", 1500).lease_id, first.lease_id)

    def test_expiry_then_reacquire_by_another_holder(self):
        first = self.lease.acquire("browser-a", 500)
        self.now = NOW + timedelta(milliseconds=600)
        second = self.lease.acquire("browser-b", 500)
        self.assertNotEqual(first.lease_id, second.lease_id)
        with self.assertRaises(LeaseError):
            self.lease.check(first.lease_id)

    def test_renew_extends_and_bumps_revision(self):
        view = self.lease.acquire("browser-a", 500)
        self.now = NOW + timedelta(milliseconds=400)
        renewed = self.lease.renew(view.lease_id, 500)
        self.assertGreater(renewed.revision, view.revision)
        self.now = NOW + timedelta(milliseconds=800)
        self.assertTrue(self.lease.check(view.lease_id))

    def test_release_frees_the_slot(self):
        view = self.lease.acquire("browser-a", 1500)
        self.lease.release(view.lease_id)
        with self.assertRaises(LeaseError):
            self.lease.check(view.lease_id)
        self.assertEqual(self.lease.acquire("browser-b", 1500).holder, "browser-b")

    def test_bad_ttl_is_refused(self):
        for ttl in (0, 199, 20000, "1000"):
            with self.subTest(ttl=ttl), self.assertRaises(LeaseError):
                self.lease.acquire("browser-a", ttl)


class ManualCommandTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="prism-lease-")
        self.worker = MemoryWorker()
        self.clock = [NOW]
        self.service = SessionService(Path(self.temporary.name), self.worker, clock=lambda: self.clock[0])
        session = self.service.create()
        self.session = self.service.mark_ready(session["id"])

    def tearDown(self):
        self.temporary.cleanup()

    def test_jog_start_requires_a_live_lease(self):
        body = self._envelope("jog_start", "jog-1", {"lease_id": "lease_fake"})
        with self.assertRaises(ControlError) as raised:
            self.service.submit(body)
        self.assertEqual(raised.exception.code, "lease_missing")
        self.assertEqual(self.worker.submissions, [])

    def test_manual_session_moves_to_running_and_back(self):
        lease = self.service.acquire_lease(self.session["id"], "browser-a", 1500)
        started = self.service.submit(
            self._envelope("jog_start", "jog-start-1", {"lease_id": lease["lease_id"], "delta_m": [0.01, 0, 0]})
        )
        self.service.observe_applied(self.session["id"], "jog-start-1")
        running = self.service.get(self.session["id"])
        self.assertEqual(started.acknowledgement, "accepted")
        self.assertEqual(running["state"], "RUNNING")
        self.assertEqual(running["mode"], "manual")

        stopped = self.service.manual_stopped(self.session["id"], "lease_expired", 12)
        self.assertEqual(stopped["state"], "PAUSED")
        self.assertIsNone(stopped["mode"])
        events = self.service.events_after(self.session["id"], 0)
        self.assertEqual(events[-1]["type"], "manual_stopped")
        self.assertEqual(events[-1]["reason"], "lease_expired")
        self.assertEqual(events[-1]["steps"], 12)

    def test_lease_expiry_blocks_a_late_jog_start(self):
        lease = self.service.acquire_lease(self.session["id"], "browser-a", 500)
        self.clock[0] = NOW + timedelta(milliseconds=900)
        with self.assertRaises(ControlError) as raised:
            self.service.submit(self._envelope("jog_start", "jog-late", {"lease_id": lease["lease_id"]}))
        self.assertEqual(raised.exception.code, "lease_expired")

    def _envelope(self, command, command_id, args):
        return CommandEnvelope(
            command_id=command_id,
            session_id=self.session["id"],
            fencing_token=self.session["fencing_token"],
            command=command,
            issued_at=self.clock[0],
            expires_at=self.clock[0] + timedelta(seconds=30),
            args=args,
        )


class FakeBackend:
    def __init__(self):
        self.steps = 0
        self.reject_after = None

    def apply(self, command, args=None):
        self.steps += 1
        if self.reject_after is not None and self.steps >= self.reject_after:
            return {"state": "PAUSED", "rejected": "outside_action_space"}
        return {"state": "RUNNING", "step": self.steps}


class ManualMotionTests(unittest.TestCase):
    def test_loop_stops_when_keepalives_stop(self):
        backend = FakeBackend()
        stopped = threading.Event()
        reasons = []
        motion = ManualMotion(
            backend, {"delta_m": [0.01, 0, 0]}, period_s=0.01, keepalive_ms=300,
            emit=lambda reason, steps: (reasons.append((reason, steps)), stopped.set()),
        )
        motion.start()
        time.sleep(0.05)
        motion.keepalive()
        self.assertTrue(stopped.wait(3), "keepalive 停止后循环必须自行结束")
        self.assertEqual(reasons[0][0], "lease_expired")
        self.assertGreater(reasons[0][1], 0)
        self.assertLessEqual(backend.steps, reasons[0][1] + 1)

    def test_operator_stop_is_reported(self):
        backend = FakeBackend()
        stopped = threading.Event()
        reasons = []
        motion = ManualMotion(
            backend, {}, period_s=0.01, keepalive_ms=5000,
            emit=lambda reason, steps: (reasons.append(reason), stopped.set()),
        )
        motion.start()
        motion.stop("operator_stop")
        self.assertTrue(stopped.wait(3))
        self.assertEqual(reasons[0], "operator_stop")

    def test_stop_waits_for_inflight_step_and_its_callback(self):
        entered, release, stopped = threading.Event(), threading.Event(), threading.Event()
        backend = FakeBackend()
        original = backend.apply
        def slow_step(*args):
            entered.set()
            release.wait(3)
            return original(*args)
        backend.apply = slow_step
        callbacks = []
        motion = ManualMotion(backend, {}, .01, 5000,
                              emit=lambda *args: callbacks.append(args), lock=threading.Lock())
        motion.start()
        self.assertTrue(entered.wait(2))
        stopper = threading.Thread(target=lambda: (motion.stop("operator_stop"), stopped.set()))
        stopper.start()
        try:
            self.assertFalse(stopped.wait(.05), "stop must await the in-flight step")
        finally:
            release.set()
            stopper.join(3)
        self.assertTrue(stopped.is_set())
        self.assertEqual(callbacks, [("operator_stop", 1)])
        self.assertEqual(backend.steps, 1)

    def test_rejected_action_ends_the_loop_instead_of_spinning(self):
        backend = FakeBackend()
        backend.reject_after = 2
        stopped = threading.Event()
        reasons = []
        motion = ManualMotion(
            backend, {}, period_s=0.01, keepalive_ms=5000,
            emit=lambda reason, steps: (reasons.append(reason), stopped.set()),
        )
        motion.start()
        self.assertTrue(stopped.wait(3))
        self.assertEqual(reasons[0], "rejected")


if __name__ == "__main__":
    unittest.main()
