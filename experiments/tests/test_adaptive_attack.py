"""Adaptive patch variants must evade the defence gate while staying visible.

The defence claims two thresholds (local variance < 12, saturation > 90). These
tests assert the adaptive patches actually break them, so a later "defence
fails" result cannot be blamed on a malformed attack.
"""

import unittest

import cv2
import numpy as np

from experiments.attacks.adaptive import (
    JitteredPatchWrapper,
    adaptive_low_sat,
    adaptive_patch,
    adaptive_texture,
)


class _Space:
    def __init__(self, shape):
        self.shape = shape
        self.dtype = np.float32


class _ActionSpace:
    def __init__(self):
        self.spaces = {
            "action.end_effector_position": _Space((3,)),
            "action.end_effector_rotation": _Space((3,)),
            "action.gripper_close": _Space((1,)),
            "action.base_motion": _Space((4,)),
        }


class _FakeEnv:
    def __init__(self):
        import gymnasium as gym

        self.action_space = _ActionSpace()
        self.observation_space = gym.spaces.Dict({
            "video.robot0_agentview_left": gym.spaces.Box(0, 255, (256, 256, 3), dtype=np.uint8),
        })
        self.steps = 0

    def reset(self, seed=None, options=None):
        return self._obs(), {}

    def step(self, action):
        self.steps += 1
        return self._obs(), 0.0, False, False, {}

    def _obs(self):
        return {"video.robot0_agentview_left": np.zeros((256, 256, 3), dtype=np.uint8)}


def _stats(patch):
    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (5, 5))
    variance = np.clip(cv2.blur(gray * gray, (5, 5)) - mean * mean, 0, None)
    return float(hsv[:, :, 1].mean()), float(variance.mean())


class AdaptivePatchTests(unittest.TestCase):
    def test_texture_patch_breaks_the_variance_gate(self):
        patch = adaptive_texture()
        _saturation, variance = _stats(patch)
        self.assertGreater(variance, 12.0, "纹理补丁必须让局部方差超过防御阈值")

    def test_low_saturation_patch_breaks_the_saturation_gate(self):
        patch = adaptive_low_sat()
        saturation, _variance = _stats(patch)
        self.assertLess(saturation, 90.0, "低饱和补丁必须低于防御的饱和度阈值")

    def test_adaptive_patch_breaks_both_gates(self):
        patch = adaptive_patch()
        saturation, variance = _stats(patch)
        self.assertLess(saturation, 90.0)
        self.assertGreater(variance, 12.0)

    def test_adaptive_patch_is_still_inside_the_victim_band(self):
        from experiments.policies.vision_centroid import VisionCentroidPolicy

        policy = VisionCentroidPolicy(_ActionSpace(), camera_key="cam")
        image = np.random.default_rng(0).integers(30, 200, (256, 256, 3), dtype=np.uint8)
        image[108:148, 108:148] = adaptive_patch()
        policy({"cam": image})
        self.assertIsNotNone(policy.last_centroid, "自适应补丁仍必须被受害者检测到")

    def test_jitter_moves_the_patch_between_frames(self):
        env = JitteredPatchWrapper(
            _FakeEnv(), adaptive_patch(), camera_key="video.robot0_agentview_left",
            location="center", alpha=1.0, jitter=4, seed=7,
        )
        env.reset(seed=0)
        first = env.placement.to_dict()
        for _ in range(5):
            env.step({})
        offsets = set(env._offsets)
        self.assertGreater(len(offsets), 1, "抖动攻击必须在帧间换位置")
        self.assertEqual(first["width"], env.placement.width)


if __name__ == "__main__":
    unittest.main()
