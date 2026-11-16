"""The run button must advance physics in the worker, independently of the UI."""
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from server.infrastructure.policy_motion import PolicyMotion
from server.infrastructure.process_worker import ProcessWorker
from server.application.session_service import SessionService
from server.domain.session_control import SessionControl

class FakeBackend:
    def __init__(self, terminal=False, fail=False):
        self.steps=0;self.terminal=terminal;self.fail=fail
    def apply(self, command):
        if self.fail:raise RuntimeError('physics failed')
        assert command=='step'
        self.steps+=1
        return {'step':self.steps,'sim_seconds':self.steps*.05,'terminated':self.terminal and self.steps==2}

class PolicyMotionTests(unittest.TestCase):
    def test_run_advances_without_browser_and_pause_is_quiescent(self):
        backend=FakeBackend();progress=threading.Event();stops=[]
        loop=PolicyMotion(backend,.01,threading.Lock(),lambda _:progress.set(),lambda *v:stops.append(v))
        loop.start()
        try:self.assertTrue(progress.wait(2))
        finally:loop.stop()
        steps=backend.steps;time.sleep(.03)
        self.assertGreaterEqual(steps,1);self.assertEqual(backend.steps,steps);self.assertEqual(stops,[])
    def test_episode_end_stops_after_the_terminal_step(self):
        backend=FakeBackend(terminal=True);done=threading.Event();stops=[]
        loop=PolicyMotion(backend,.01,threading.Lock(),lambda _:None,lambda *v:(stops.append(v),done.set()))
        loop.start()
        try:self.assertTrue(done.wait(2))
        finally:loop.stop()
        self.assertEqual(backend.steps,2);self.assertEqual(stops[0][0],'episode_done')
    def test_physics_error_is_reported_and_loop_stops(self):
        backend=FakeBackend(fail=True);done=threading.Event();stops=[]
        loop=PolicyMotion(backend,.01,threading.Lock(),lambda _:None,lambda *v:(stops.append(v),done.set()))
        loop.start()
        try:self.assertTrue(done.wait(2))
        finally:loop.stop()
        self.assertEqual(stops[0][0],'policy_error');self.assertIn('physics failed',stops[0][1]['last_error'])
    def test_policy_stop_state_can_be_replayed(self):
        control=SessionControl('sess_policy','fence');control.create_event();control.mark_loading();control.mark_ready()
        control.state='RUNNING';control.mark_policy_stopped('episode_done')
        restored=SessionControl.from_events(control.events)
        self.assertEqual(restored.state,'PAUSED');self.assertEqual(restored.events[-1]['reason'],'episode_done')
    def test_child_process_runs_then_pause_stops_progress(self):
        with tempfile.TemporaryDirectory(prefix='prism-policy-') as directory:
            progress=[];applied=[];changed=threading.Event();paused=threading.Event()
            def acknowledge(session,command,detail):
                applied.append((command,detail))
                if command=='pause':paused.set()
            worker=ProcessWorker(Path(directory),env={**os.environ,'PRISM_BACKEND':'state_only'},on_applied=acknowledge,on_policy_progress=lambda session,detail:(progress.append(detail),changed.set()))
            try:
                worker.start('sess_policy');worker.submit('sess_policy','resume','resume')
                self.assertTrue(changed.wait(10));self.assertGreaterEqual(progress[-1]['step'],1)
                worker.submit('sess_policy','pause','pause');self.assertTrue(paused.wait(10))
                count=len(progress);time.sleep(.35);self.assertEqual(len(progress),count)
                self.assertEqual(applied[-1][1]['state'],'PAUSED')
            finally:worker.crash('sess_policy')

if __name__=='__main__':unittest.main()
