"""RoboCasa adapter contracts verified with a fake environment.

These tests never import MuJoCo. They pin the rules that matter: config is
validated fail-closed, custom scenes bypass `split`, a rejected action does not
advance the environment, and a frame is published per step.
"""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from server.runtime.adapters.robocasa_backend import (
    ConfigError,
    RoboCasaBackend,
    environment_kwargs,
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
    def __init__(self, steps_to_terminate=3):
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
        return self._observation(), {"annotation.human.task_description": "把杯子放进柜子"}

    def step(self, action):
        self.applied.append(action)
        self.steps += 1
        self.unwrapped.sim.data.time += 0.05
        terminated = self.steps >= self.steps_to_terminate
        return self._observation(), 1.0, terminated, False, {"success": terminated}

    def _observation(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        return {
            "video.robot0_agentview_left": image,
            "state.end_effector_position_relative": np.array([0.0, 0.0, 0.0], dtype=np.float64),
            "annotation.human.task_description": "把杯子放进柜子",
        }


class FakeGuard:
    def __init__(self, accept=True):
        self.accept = accept

    def check(self, action):
        return (True, None) if self.accept else (False, "outside_action_space")


class FakePolicy:
    name = "fake"
    uses_vision = False

    def __init__(self, action_space):
        self.action_space = action_space

    def __call__(self, observation):
        return self.action_space.sample()


class NormalizeConfigTests(unittest.TestCase):
    def test_defaults_are_explicit_and_custom_scene_bypasses_split(self):
        config = normalize_config(None)
        self.assertEqual(config["env_id"], "robocasa/PickPlaceCounterToCabinet")
        self.assertEqual(config["split"], "pretrain")
        custom = normalize_config({"scene_mode": "custom", "layout_id": 3, "style_id": 4})
        kwargs = environment_kwargs(custom)
        self.assertIsNone(kwargs["split"])
        self.assertEqual(kwargs["layout_ids"], [3])
        self.assertEqual(kwargs["style_ids"], [4])
        self.assertEqual(kwargs["obj_instance_split"], "pretrain")

    def test_vision_policy_is_selectable_and_declares_vision(self):
        config = normalize_config({"policy": "vision_centroid", "vision_target": "green", "vision_gain": 0.7})
        self.assertEqual(config["policy"], "vision_centroid")
        self.assertEqual(config["vision_target"], "green")
        self.assertAlmostEqual(config["vision_gain"], 0.7)

    def test_vision_parameters_are_validated(self):
        for payload in ({"vision_target": "purple"}, {"vision_gain": 5}, {"vision_gain": "fast"}):
            with self.subTest(payload=payload), self.assertRaises(ConfigError):
                normalize_config(payload)

    def test_unknown_and_out_of_range_values_fail_closed(self):
        for payload in (
            {"surprise": 1},
            {"env_id": "libero/PickCup"},
            {"control_freq": 60},
            {"seed": -1},
            {"policy": "vla"},
            {"camera_key": "video.nope"},
            {"scene_mode": "sample", "layout_id": 2},
            {"scene_mode": "custom"},
        ):
            with self.subTest(payload=payload), self.assertRaises(ConfigError):
                normalize_config(payload)


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="prism-backend-")
        self.frames = Path(self.temporary.name) / "frames"
        self.env = FakeEnv()
        self.backend = RoboCasaBackend(
            self.frames,
            env_factory=lambda config: self.env,
            policy_factory=lambda env, config: FakePolicy(env.action_space),
            guard_factory=lambda env: FakeGuard(),
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_load_does_not_advance_physics_and_publishes_one_frame(self):
        info = self.backend.load("sess_backend", {"seed": 7})
        self.assertEqual(self.env.resets, [7])
        self.assertEqual(self.env.steps, 0)
        self.assertEqual(info["step"], 0)
        self.assertEqual(info["policy_uses_vision"], False)
        self.assertTrue((self.frames / "latest.jpg").is_file())
        self.assertEqual(info["frame_seq"], 1)

    def test_step_advances_once_and_reports_real_physics_seconds(self):
        self.backend.load("sess_backend", {"seed": 0})
        result = self.backend.apply("step")
        self.assertEqual(result["step"], 1)
        self.assertAlmostEqual(result["sim_seconds"], 0.05, places=6)
        self.assertEqual(result["state"], "PAUSED")
        self.assertEqual(len(self.env.applied), 1)

    def test_rejected_action_does_not_advance_the_environment(self):
        self.backend = RoboCasaBackend(
            self.frames,
            env_factory=lambda config: self.env,
            policy_factory=lambda env, config: FakePolicy(env.action_space),
            guard_factory=lambda env: FakeGuard(accept=False),
        )
        self.backend.load("sess_backend", {"seed": 0})
        result = self.backend.apply("step")
        self.assertEqual(result["step"], 0)
        self.assertEqual(self.env.applied, [])
        self.assertEqual(result["rejected"], "outside_action_space")
        self.assertEqual(result["last_error"]["code"], "action_rejected")

    def test_reset_starts_a_new_episode_without_losing_the_counter(self):
        self.backend.load("sess_backend", {"seed": 0})
        self.backend.apply("step")
        previous_frame = self.backend.frame_seq
        result = self.backend.apply("reset")
        self.assertGreater(result["frame_seq"], previous_frame)
        self.assertEqual(result["published_frame_seq"], result["frame_seq"])
        self.assertIsNone(result["frame_error"])
        self.assertEqual(result["episode"], 2)
        self.assertEqual(result["step"], 0)
        self.assertEqual(result["state"], "PAUSED")

    def test_stop_closes_without_claiming_a_measurement(self):
        self.backend.load("sess_backend", {"seed": 0})
        self.assertEqual(self.backend.apply("stop")["state"], "CLOSED")


class JogTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="prism-jog-")
        self.env = FakeEnv()
        self.backend = RoboCasaBackend(
            Path(self.temporary.name) / "frames",
            env_factory=lambda config: self.env,
            policy_factory=lambda env, config: FakePolicy(env.action_space),
            guard_factory=lambda env: FakeGuard(),
        )
        self.backend.load("sess_jog", {"seed": 0})

    def tearDown(self):
        self.temporary.cleanup()

    def test_jog_maps_meters_to_controller_input_and_advances_once(self):
        result = self.backend.apply("jog", {"delta_m": [0.05, -0.025, 0.0], "gripper": "close"})
        action = self.env.applied[-1]
        self.assertEqual(result["step"], 1)
        self.assertEqual(result["jog"]["gripper"], "close")
        self.assertAlmostEqual(float(action["action.end_effector_position"][0]), 1.0, places=6)
        self.assertAlmostEqual(float(action["action.end_effector_position"][1]), -0.5, places=6)
        self.assertAlmostEqual(float(action["action.gripper_close"][0]), 1.0, places=6)
        self.assertListEqual(np.round(action["action.end_effector_rotation"], 6).tolist(), [0.0, 0.0, 0.0])

    def test_gripper_open_maps_to_minus_one(self):
        self.backend.apply("jog", {"gripper": "open"})
        self.assertAlmostEqual(float(self.env.applied[-1]["action.gripper_close"][0]), -1.0, places=6)

    def test_hold_repeats_the_last_gripper_command_instead_of_inventing_a_neutral(self):
        self.backend.apply("jog", {"gripper": "close"})
        result = self.backend.apply("jog", {"gripper": "hold"})
        self.assertEqual(result["jog"]["gripper"], "hold")
        self.assertEqual(result["jog"]["gripper_state"], "close")
        self.assertAlmostEqual(float(self.env.applied[-1]["action.gripper_close"][0]), 1.0, places=6)

    def test_oversized_step_is_refused_and_does_not_advance(self):
        result = self.backend.apply("jog", {"delta_m": [0.051, 0, 0]})
        self.assertEqual(result["step"], 0)
        self.assertEqual(self.env.applied, [])
        self.assertEqual(result["last_error"]["code"], "jog_rejected")

    def test_unknown_argument_and_bad_gripper_are_refused(self):
        for args in ({"delta_m": [0, 0, 0, 0]}, {"delta_m": ["a", 0, 0]}, {"gripper": "half"}, {"jump": 1}):
            with self.subTest(args=args):
                result = self.backend.apply("jog", args)
                self.assertEqual(result["state"], "PAUSED")
                self.assertIn("rejected", result)
        self.assertEqual(self.env.applied, [])

    def test_repeat_runs_several_steps_for_one_acknowledged_command(self):
        result = self.backend.apply("jog", {"delta_m": [0.01, 0, 0], "repeat": 3})
        self.assertEqual(result["step"], 3)
        self.assertEqual(result["jog"]["repeat"], 3)
        self.assertEqual(len(self.env.applied), 3)
        self.assertIn("measured_m", result["jog"])

    def test_manual_span_reports_the_measured_displacement(self):
        env = self._moving_env()
        backend = RoboCasaBackend(
            Path(self.temporary.name) / "manual_frames",
            env_factory=lambda config: env,
            policy_factory=lambda env, config: FakePolicy(env.action_space),
            guard_factory=lambda env: FakeGuard(),
        )
        backend.load("sess_manual", {"seed": 0})
        backend.begin_manual()
        for _ in range(3):
            backend.apply("jog", {"delta_m": [0.02, 0.0, 0.0]})
        measured = backend.end_manual()
        self.assertIsNotNone(measured)
        self.assertGreater(abs(measured[0]), 0.0)
        self.assertIsNone(backend.end_manual())

    def _moving_env(self):
        env = FakeEnv()
        position = {"x": 0.0}

        def step(action):
            position["x"] += float(action["action.end_effector_position"][0]) * 0.01
            return env._step_inner(action)

        env._step_inner = env.step
        env.step = step
        env._observation = lambda: {
            "video.robot0_agentview_left": np.zeros((16, 16, 3), dtype=np.uint8),
            "state.end_effector_position_relative": np.array([position["x"], 0.0, 0.0]),
            "annotation.human.task_description": "移动",
        }
        return env

    def test_repeat_is_bounded(self):
        for repeat in (0, 11, "3", 2.0):
            with self.subTest(repeat=repeat):
                result = self.backend.apply("jog", {"repeat": repeat})
                self.assertIn("rejected", result)
        self.assertEqual(self.env.applied, [])

    def test_capabilities_declare_what_is_missing(self):
        info = self.backend.load("sess_jog", {"seed": 0})
        self.assertTrue(info["capabilities"]["jog"])
        self.assertTrue(info["capabilities"]["gripper"])
        self.assertFalse(info["capabilities"]["joint_position"])
        self.assertTrue(info["capabilities"]["scene_edit"])


class EditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="prism-edit-")
        self.env = FakeEnv()
        self.backend = RoboCasaBackend(
            Path(self.temporary.name) / "frames",
            env_factory=lambda config: self.env,
            policy_factory=lambda env, config: FakePolicy(env.action_space),
            guard_factory=lambda env: FakeGuard(),
        )
        self.backend.load("sess_edit", {"seed": 0})

    def tearDown(self):
        self.temporary.cleanup()

    def test_edit_refuses_instead_of_crashing_when_scene_is_unavailable(self):
        result = self.backend.apply("edit", {"target": "cup", "kind": "mass", "values": 1.0})
        self.assertEqual(result["state"], "PAUSED")
        self.assertIn("rejected", result)
        self.assertEqual(result["last_error"]["code"], "edit_rejected")
        self.assertEqual(self.backend.step, 0)

    def test_edit_requires_target_and_known_kind(self):
        for args in ({}, {"target": "cup"}, {"target": "cup", "kind": "colour", "values": 1}, "nope"):
            with self.subTest(args=args):
                self.assertIn("rejected", self.backend.apply("edit", args))

    def test_mass_bounds_are_checked_before_touching_the_model(self):
        result = self.backend.apply("edit", {"target": "cup", "kind": "mass", "values": 999.0})
        self.assertIn("rejected", result)
        self.assertIn("质量", result["rejected"])

    def test_position_requires_three_finite_numbers(self):
        for values in ([1, 2], ["a", 1, 2], [1, 2, 3, 4]):
            with self.subTest(values=values):
                result = self.backend.apply("edit", {"target": "cup", "kind": "position", "values": values})
                self.assertIn("rejected", result)


if __name__ == "__main__":
    unittest.main()
