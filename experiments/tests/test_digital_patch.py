"""Invariants needed for clean/attacked comparisons, without MuJoCo assets."""

import unittest

import gymnasium as gym
import numpy as np

from experiments.attacks.digital_patch import (
    apply_patch,
    image_delta_metrics,
    make_noise_patch,
)
from experiments.attacks.observation import DigitalPatchWrapper


class FakeCameraEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Dict({
            "camera": gym.spaces.Box(0, 255, (8, 10, 3), np.uint8),
            "other_camera": gym.spaces.Box(0, 255, (8, 10, 3), np.uint8),
            "state": gym.spaces.Box(-1, 1, (2,), np.float32),
        })
        self.action_space = gym.spaces.Box(-1, 1, (1,), np.float32)
        self.last_observation = None

    def _observation(self, value):
        self.last_observation = {
            "camera": np.full((8, 10, 3), value, np.uint8),
            "other_camera": np.full((8, 10, 3), 33, np.uint8),
            "state": np.zeros(2, np.float32),
        }
        return self.last_observation

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._observation(0), {"success": False}

    def step(self, action):
        return self._observation(12), 0.75, True, False, {"success": True}


class DigitalPatchTests(unittest.TestCase):
    def test_only_bbox_changes_and_inputs_remain_intact(self):
        clean = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
        patch = np.full((2, 4, 3), 255, dtype=np.uint8)
        original = clean.copy()
        attacked, placement = apply_patch(clean, patch, location="center")
        self.assertEqual((placement.top, placement.left, placement.bottom, placement.right), (3, 3, 5, 7))
        mask = np.zeros(clean.shape[:2], dtype=bool)
        mask[3:5, 3:7] = True
        np.testing.assert_array_equal(attacked[~mask], clean[~mask])
        np.testing.assert_array_equal(attacked[mask], patch.reshape(-1, 3))
        np.testing.assert_array_equal(clean, original)
        self.assertFalse(np.shares_memory(attacked, clean))
        self.assertFalse(np.shares_memory(attacked, patch))

    def test_uint8_alpha_blending_no_overflow_or_source_mutation(self):
        clean = np.full((4, 4, 3), 200, dtype=np.uint8)
        patch = np.full((2, 2, 3), 100, dtype=np.uint8)
        attacked, _ = apply_patch(clean, patch, location=(1, 1), alpha=0.5)
        self.assertTrue(np.all(attacked[1:3, 1:3] == 150))
        self.assertTrue(np.all(patch == 100))
        transparent, _ = apply_patch(clean, patch, alpha=0)
        np.testing.assert_array_equal(transparent, clean)
        self.assertEqual(image_delta_metrics(clean, transparent)["changed_pixels"], 0)

    def test_pixel_metrics_count_pixels_not_channels(self):
        clean = np.zeros((4, 4, 3), dtype=np.uint8)
        patch = np.full((2, 2, 3), 255, dtype=np.uint8)
        attacked, _ = apply_patch(clean, patch)
        metrics = image_delta_metrics(clean, attacked)
        self.assertEqual(metrics["changed_pixels"], 4)
        self.assertEqual(metrics["changed_fraction"], 0.25)
        self.assertEqual(metrics["max_abs_delta"], 255)
        self.assertEqual(metrics["mean_abs_delta"], 63.75)

    def test_noise_seed_does_not_consume_global_rng(self):
        np.random.seed(123)
        expected = np.random.random(5)
        np.random.seed(123)
        first = make_noise_patch((4, 5), seed=42)
        np.testing.assert_array_equal(np.random.random(5), expected)
        np.testing.assert_array_equal(first, make_noise_patch((4, 5), seed=42))
        self.assertFalse(np.array_equal(first, make_noise_patch((4, 5), seed=43)))
        self.assertEqual(first.shape, (4, 5, 3))

    def test_out_of_bounds_and_invalid_alpha_are_rejected(self):
        image = np.zeros((8, 10, 3), dtype=np.uint8)
        patch = make_noise_patch(3)
        for kwargs in (
            {"location": (-1, 0)}, {"location": (6, 8)},
            {"location": "bottom_right", "offset": (1, 0)},
            {"location": "unknown"}, {"alpha": -0.1}, {"alpha": 1.1},
            {"alpha": float("nan")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                apply_patch(image, patch, **kwargs)
        with self.assertRaises(ValueError):
            apply_patch(image, make_noise_patch(11))

    def test_normalized_float_pixels_cannot_be_silently_corrupted(self):
        with self.assertRaises(TypeError):
            apply_patch(np.zeros((4, 4, 3), np.float32), make_noise_patch(2))
        with self.assertRaises(ValueError):
            apply_patch(np.zeros((4, 4), np.uint8), make_noise_patch(2))

    def test_wrapper_patches_reset_and_step_and_preserves_other_observations(self):
        raw = FakeCameraEnv()
        env = DigitalPatchWrapper(raw, np.full((2, 2, 3), 255, np.uint8), camera_key="camera")
        observation, info = env.reset(seed=0)
        self.assertEqual(info, {"success": False})
        self.assertEqual(env.last_metrics["changed_pixels"], 4)
        self.assertTrue(np.all(raw.last_observation["camera"] == 0))
        self.assertTrue(np.all(env.last_clean_image == 0))
        self.assertTrue(env.observation_space.contains(observation))
        np.testing.assert_array_equal(observation["other_camera"], raw.last_observation["other_camera"])
        np.testing.assert_array_equal(observation["state"], raw.last_observation["state"])
        observation, reward, terminated, truncated, info = env.step(np.zeros(1, np.float32))
        self.assertEqual((reward, terminated, truncated, info), (0.75, True, False, {"success": True}))
        self.assertTrue(np.all(env.last_clean_image == 12))
        self.assertTrue(np.all(raw.last_observation["camera"] == 12))
        self.assertEqual(env.last_metrics["changed_pixels"], 4)
        self.assertTrue(env.observation_space.contains(observation))

    def test_wrapper_rejects_wrong_camera_and_owns_patch_copy(self):
        raw = FakeCameraEnv()
        patch = np.full((2, 2, 3), 255, np.uint8)
        with self.assertRaises(KeyError):
            DigitalPatchWrapper(raw, patch, camera_key="missing")
        env = DigitalPatchWrapper(raw, patch, camera_key="camera")
        patch[:] = 0
        observation, _ = env.reset()
        self.assertEqual(image_delta_metrics(env.last_clean_image, observation["camera"])["changed_pixels"], 4)


if __name__ == "__main__":
    unittest.main()
