"""Backend wiring for the tool safety observer, verified without MuJoCo.

The real `_tool_safety_scene` needs a live MuJoCo model, so these tests inject a
fake snapshot into that one seam and leave everything else real: config
validation, observer lifetime, tick placement, overlays, and error handling.
Regressions guarded here:

- a disabled spec must not create an observer or call the scene builder;
- every `_advance` evaluates, including jog repeats and reset;
- `load` evaluates step 0 once (no double counting, no missed step);
- thinning the browser geometry payload still leaves the observer with static
  geometry (it reads the cache, not the polled payload);
- an unknown tool body is reported as unknown and never stops the robot;
- overlays are explicitly empty (not "safe") when the feature is off or errored.
"""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from server.domain.tool_safety import ToolSafetySpecError, normalize_tool_safety_spec
from server.runtime.adapters.robocasa_backend import (
    ConfigError,
    RoboCasaBackend,
    normalize_config,
)


class FakeSpace:
    def __init__(self, shape, dtype=np.float32):
        self.shape = shape
        self.dtype = dtype


class FakeActionSpace:
    def __init__(self):
        self.spaces = {
            "action.end_effector_position": FakeSpace((3,)),
            "action.end_effector_rotation": FakeSpace((3,)),
            "action.gripper_close": FakeSpace((1,)),
        }

    def sample(self):
        return {key: np.zeros(space.shape, dtype=space.dtype) for key, space in self.spaces.items()}


class FakeSim:
    def __init__(self):
        self.data = type("Data", (), {"time": 0.0})()


class FakeUnwrappedEnv:
    def __init__(self):
        self.control_freq = 20
        self.sim = FakeSim()


class FakeEnv:
    def __init__(self, steps_to_terminate=100):
        self.action_space = FakeActionSpace()
        self.unwrapped = FakeUnwrappedEnv()
        self.steps_to_terminate = steps_to_terminate
        self.steps = 0
        self.resets = []
        self.applied = []

    def reset(self, seed=None):
        self.resets.append(seed)
        self.steps = 0
        self.unwrapped.sim.data.time = 0.0
        return self._observation(), {"annotation.human.task_description": "取刀"}

    def step(self, action):
        self.applied.append(action)
        self.steps += 1
        self.unwrapped.sim.data.time += 0.05
        terminated = self.steps >= self.steps_to_terminate
        return self._observation(), 1.0, terminated, False, {"success": terminated}

    def _observation(self):
        return {
            "video.robot0_agentview_left": np.zeros((16, 16, 3), dtype=np.uint8),
            "state.end_effector_position_relative": np.array([0.0, 0.0, 0.0], dtype=np.float64),
            "annotation.human.task_description": "取刀",
        }


class FakeGuard:
    def check(self, action):
        return True, None


class RejectingGuard:
    def check(self, action):
        return False, "outside_action_space"


class FakePolicy:
    name = "fake"
    uses_vision = False

    def __init__(self, action_space):
        self.action_space = action_space

    def __call__(self, observation):
        return self.action_space.sample()


TOOL_BODY_ID = "8"
TOOL_BODY_NAME = "knife_1_main"
INSIDE_TOOL_POSE = {TOOL_BODY_ID: {"p": [0.0, 0.0, 0.9], "q": [1.0, 0.0, 0.0, 0.0]}}


def geometry_snapshot(key="g_test_key"):
    return {
        "key": key,
        "frame": {"world": "mujoco_world", "units": "m", "up": "+z", "quaternion": "wxyz"},
        "robot": {"bodies": [{"id": "1", "name": "robot0_base"}]},
        "objects": [{"id": TOOL_BODY_ID, "name": TOOL_BODY_NAME, "body": TOOL_BODY_NAME}],
    }


def blade_spec(enabled=True, body=TOOL_BODY_NAME):
    return {
        "version": "tool_safety_spec/1",
        "enabled": enabled,
        "risk_zones": [{"id": "prep_zone", "kind": "sphere", "center": [0.0, 0.0, 0.9], "radius": 0.05, "clearance": 0.0}],
        "tool_annotations": [{
            "tool_id": "knife",
            "body": body,
            "approximation": "blade_edge_segment",
            "segments": [{"id": "blade", "kind": "segment", "start": [-0.09, 0.0, 0.0], "end": [0.09, 0.0, 0.0], "radius": 0.002}],
        }],
    }


class WiringTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="prism-tool-backend-")
        self.frames = Path(self.temporary.name) / "frames"
        self.env = FakeEnv()

    def tearDown(self):
        self.temporary.cleanup()

    def build(self, config=None, guard=None, stub=True, transforms=None):
        """Construct, optionally stub the scene seam, then load."""
        backend = RoboCasaBackend(
            self.frames,
            env_factory=lambda cfg: self.env,
            policy_factory=lambda env, cfg: FakePolicy(env.action_space),
            guard_factory=lambda env: (guard or FakeGuard()),
        )
        if stub:
            self.stub_scene(backend, transforms)
        backend.load("sess_tool", {"seed": 0, **(config or {})})
        return backend

    def stub_scene(self, backend, transforms=None):
        def scene():
            return {
                "session_id": backend.session_id,
                "episode": backend.episode,
                "step": backend.step,
                "sim_seconds": backend.sim_seconds,
                "body_transforms": transforms if transforms is not None else dict(INSIDE_TOOL_POSE),
                "geometry": geometry_snapshot(),
            }
        backend._tool_safety_scene = scene
        return backend

    # ---------------------------------------------------------------- config
    def test_normalize_config_accepts_spec_and_defaults_disabled(self):
        config = normalize_config({"tool_safety": blade_spec()})
        self.assertEqual(config["tool_safety"]["version"], "tool_safety_spec/1")
        self.assertTrue(config["tool_safety"]["enabled"])
        default = normalize_config(None)["tool_safety"]
        self.assertFalse(default["enabled"])
        self.assertEqual(default["risk_zones"], [])

    def test_invalid_spec_fails_closed_at_config_time(self):
        with self.assertRaises(ConfigError):
            normalize_config({"tool_safety": {"enabled": True, "risk_zones": [{"id": "z", "kind": "sphere", "center": [0, 0, 0], "radius": -1}]}})
        with self.assertRaises(ConfigError):
            normalize_config({"tool_safety": {"surprise": 1}})
        with self.assertRaises(ToolSafetySpecError):
            normalize_tool_safety_spec({"enabled": True, "tool_annotations": [{"tool_id": "k", "body": "b", "segments": [{"id": "s", "start": [0, 0, 0], "end": [float("nan"), 0, 0]}]}]})

    # ---------------------------------------------------------------- lifetime
    def test_disabled_spec_creates_no_observer_and_no_scene_work(self):
        built = {"count": 0}

        def counted():
            built["count"] += 1
            return {}

        backend = self.build(guard=None, stub=False)
        backend._tool_safety_scene = counted
        self.assertIsNone(backend.tool_safety)
        backend.apply("step")
        self.assertEqual(built["count"], 0, "关闭时不得构造场景快照")
        self.assertEqual(backend._summary()["tool_safety"]["status"], "disabled")

    def test_enabled_observer_assesses_load_step_zero_once(self):
        backend = self.build({"tool_safety": blade_spec()})
        self.assertIsNotNone(backend.tool_safety)
        self.assertEqual(backend.tool_safety.metrics["steps_assessed"], 1)
        self.assertEqual(backend.tool_safety.metrics["step"], 0)

    def test_every_advance_evaluates_not_browser_polling(self):
        backend = self.build({"tool_safety": blade_spec()})
        backend.apply("step")
        backend.apply("step")
        self.assertEqual(backend.tool_safety.metrics["steps_assessed"], 3)
        # 重复查询 scene_state 不会推进评估
        before = backend.tool_safety.metrics["steps_assessed"]
        backend.scene_state({})
        self.assertEqual(backend.tool_safety.metrics["steps_assessed"], before)

    def test_jog_repeat_evaluates_each_physical_step(self):
        backend = self.build({"tool_safety": blade_spec()})
        backend.apply("jog", {"delta_m": [0.01, 0.0, 0.0], "repeat": 4})
        self.assertEqual(backend.tool_safety.metrics["steps_assessed"], 5)  # load + 4

    def test_reset_clears_evidence_and_keeps_config(self):
        backend = self.build({"tool_safety": blade_spec()})
        backend.apply("step")
        self.assertEqual(backend.tool_safety.metrics["episode_violations"], 2)  # load step0 + 1 step
        result = backend.apply("reset")
        self.assertEqual(result["episode"], 2)
        metrics = backend.tool_safety.metrics
        self.assertEqual(metrics["steps_assessed"], 1)
        self.assertTrue(backend.tool_safety.enabled)
        self.assertTrue(metrics["reset_reason"] in {"session_changed", "first_observation"})

    def test_rejected_action_does_not_evaluate(self):
        backend = self.build({"tool_safety": blade_spec()}, guard=RejectingGuard())
        before = backend.tool_safety.metrics["steps_assessed"]
        result = backend.apply("step")
        self.assertIn("rejected", result)
        self.assertEqual(backend.tool_safety.metrics["steps_assessed"], before)

    def test_hold_and_random_control_still_work_with_tool_safety_on(self):
        backend = self.build({"tool_safety": blade_spec(), "policy": "random"})
        stepped = backend.apply("step")
        self.assertEqual(stepped["step"], 1)
        jogged = backend.apply("jog", {"delta_m": [0.01, 0.0, 0.0], "gripper": "hold"})
        self.assertEqual(jogged["jog"]["gripper"], "hold")
        self.assertEqual(jogged["step"], 2)

    # ---------------------------------------------------------------- payload
    def test_payload_reports_violation_and_drives_overlays(self):
        backend = self.build({"tool_safety": blade_spec()})
        payload = backend.tool_safety_payload
        self.assertEqual(payload["status"], "violation")
        overlays = backend._overlays()
        self.assertEqual(overlays["risk_zones"][0]["id"], "prep_zone")
        self.assertEqual(len(overlays["world_tool_segments"]), 1)
        self.assertEqual(overlays["source"], "user_defined_annotations")
        self.assertEqual(overlays["status"], "violation")
        self.assertEqual(overlays["risk_zones_source"], "tool_safety_observer")

    def test_clear_tool_pose_is_ok_without_violation(self):
        backend = self.build({"tool_safety": blade_spec()}, transforms={
            TOOL_BODY_ID: {"p": [2.0, 0.0, 0.9], "q": [1.0, 0.0, 0.0, 0.0]},
        })
        self.assertEqual(backend.tool_safety_payload["status"], "ok")
        self.assertEqual(backend.tool_safety_payload["assessment"]["violations"], [])
        self.assertEqual(backend._overlays()["status"], "ok")

    def test_observer_uses_cached_geometry_when_payload_is_thinned(self):
        """Task-object geometry is omitted in the Raycast init overlay snapshot."""
        backend = self.build({"tool_safety": blade_spec()})
        built = {"count": 0}
        original_build = None
        import server.runtime.adapters.scene_geometry as scene_geometry

        original_build = scene_geometry.build_geometry

        def counting_build(model, task, robot):
            built["count"] += 1
            return geometry_snapshot("g_built")

        def fake_transforms(model, data):
            return dict(INSIDE_TOOL_POSE)

        original_transforms = scene_geometry.build_transforms
        scene_geometry.build_geometry = counting_build
        scene_geometry.build_transforms = fake_transforms
        backend.geometry = geometry_snapshot("g_cached")
        backend._geometry_key = lambda model, task: "g_cached"
        backend.env.unwrapped.env = type("Task", (), {
            "sim": type("Sim", (), {"model": object(), "data": object()})(),
            "robots": [object()],
        })()
        # 恢复真实场景构造：这条测试要验证的就是“读缓存”的那段代码
        backend._tool_safety_scene = RoboCasaBackend._tool_safety_scene.__get__(backend)
        try:
            backend._tool_safety_tick()
        finally:
            scene_geometry.build_geometry = original_build
            scene_geometry.build_transforms = original_transforms
        self.assertEqual(built["count"], 0, "已有缓存时不应重建几何")
        self.assertEqual(backend.tool_safety_payload["status"], "violation")
        self.assertEqual(backend._tool_body_index[TOOL_BODY_NAME], TOOL_BODY_ID)

    def test_unknown_tool_body_is_unknown_and_never_stops_the_robot(self):
        backend = self.build({"tool_safety": blade_spec(body="nonexistent_tool_body")})
        result = backend.apply("step")
        payload = backend.tool_safety_payload
        self.assertEqual(payload["status"], "unknown")
        self.assertIn("missing_body_pose:nonexistent_tool_body:knife", payload["assessment"]["unknown_reasons"])
        self.assertEqual(payload["assessment"]["violations"], [])
        self.assertEqual(result["step"], 1)
        self.assertEqual(result["state"], "PAUSED")
        self.assertNotIn("stopped", result)
        self.assertEqual(result["tool_safety"]["status"], "unknown")

    def test_overlays_are_explicit_when_disabled_or_errored(self):
        backend = self.build()
        off = backend._overlays()
        self.assertEqual(off["source"], "tool_safety_disabled")
        self.assertEqual(off["risk_zones"], [])
        self.assertEqual(off["world_tool_segments"], [])

        backend = self.build({"tool_safety": blade_spec()})
        backend.tool_safety_error = "RuntimeError: boom"
        backend.tool_safety_payload = None
        errored = backend._overlays()
        self.assertEqual(errored["source"], "tool_safety_error")
        self.assertIn("boom", errored["reason"])

    def test_tick_failure_is_recorded_and_does_not_stop_the_robot(self):
        backend = self.build({"tool_safety": blade_spec()})

        def explode():
            raise RuntimeError("geometry exploded")

        backend._tool_safety_scene = explode
        result = backend.apply("step")
        self.assertEqual(result["step"], 1)
        self.assertIn("geometry exploded", backend.tool_safety_error)
        self.assertEqual(backend._summary()["tool_safety"]["status"], "unknown")
        self.assertEqual(backend._overlays()["source"], "tool_safety_error")


if __name__ == "__main__":
    unittest.main()
