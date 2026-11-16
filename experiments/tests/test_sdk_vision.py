"""SDK contract, registry, and the vision victim policy.

No MuJoCo here: the colour-centroid policy is exercised on synthetic frames so
that the "pixels drive actions" claim is checked without a 30 s scene load.
"""

from pathlib import Path
import unittest

import numpy as np

from experiments.policies.vision_centroid import VisionCentroidPolicy
from experiments.sdk.contract import (
    PolicyContractError,
    require_manifest,
    validate_action,
)
from experiments.sdk.registry import build_policy, policy_names


class FakeSpace:
    def __init__(self, shape, dtype=np.float32):
        self.shape = shape
        self.dtype = dtype
        self.low = -np.ones(shape, dtype=dtype)
        self.high = np.ones(shape, dtype=dtype)

    def contains(self, value):
        array = np.asarray(value)
        return array.shape == self.shape and bool(np.all(array >= self.low)) and bool(np.all(array <= self.high))


class FakeActionSpace:
    def __init__(self):
        self.spaces = {
            "action.end_effector_position": FakeSpace((3,)),
            "action.end_effector_rotation": FakeSpace((3,)),
            "action.gripper_close": FakeSpace((1,)),
            "action.base_motion": FakeSpace((4,)),
        }


def frame_with_blob(cx, cy, radius=10):
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:128, 0:128]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    image[mask] = (40, 200, 60)
    return image


class ContractTests(unittest.TestCase):
    def test_action_shape_and_space_are_enforced(self):
        space = FakeActionSpace()
        good = {key: np.zeros(s.shape, dtype=np.float32) for key, s in space.spaces.items()}
        self.assertIs(validate_action(good, space), good)
        for broken in (
            {},
            {**good, "extra": np.zeros(1, dtype=np.float32)},
            {**good, "action.gripper_close": np.zeros(2, dtype=np.float32)},
            {**good, "action.end_effector_position": np.full(3, 2.0, dtype=np.float32)},
        ):
            with self.subTest(broken=list(broken)):
                with self.assertRaises(PolicyContractError):
                    validate_action(broken, space)

    def test_registry_refuses_unknown_names_and_lists_known_ones(self):
        self.assertIn("vision_centroid", policy_names())
        with self.assertRaises(PolicyContractError):
            build_policy("gpt-vla", None, {})


class VisionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.space = FakeActionSpace()
        self.policy = VisionCentroidPolicy(self.space, camera_key="camera")

    def test_blob_left_of_centre_pushes_the_arm_left(self):
        action = self.policy({"camera": frame_with_blob(30, 64)})
        self.assertLess(float(action["action.end_effector_position"][0]), 0.0)
        self.assertAlmostEqual(float(action["action.end_effector_position"][1]), 0.0, places=3)

    def test_blob_right_of_centre_pushes_the_arm_right(self):
        action = self.policy({"camera": frame_with_blob(100, 64)})
        self.assertGreater(float(action["action.end_effector_position"][0]), 0.0)

    def test_no_blob_stops_motion_and_closes_the_gripper(self):
        action = self.policy({"camera": np.zeros((128, 128, 3), dtype=np.uint8)})
        self.assertTrue(np.allclose(action["action.end_effector_position"], 0.0))
        self.assertEqual(float(action["action.gripper_close"][0]), 1.0)

    def test_same_frame_gives_the_same_action(self):
        frame = frame_with_blob(70, 40)
        first = self.policy({"camera": frame})
        second = self.policy({"camera": frame})
        for key in first:
            self.assertTrue(np.array_equal(first[key], second[key]), key)

    def test_manifest_declares_vision_and_names_the_limitation(self):
        manifest = require_manifest(self.policy)
        self.assertTrue(manifest.uses_vision)
        self.assertEqual(manifest.camera_key, "camera")
        self.assertIn("不是 VLA", manifest.notes)

    def test_action_is_inside_the_action_space(self):
        action = self.policy({"camera": frame_with_blob(5, 5)})
        validate_action(action, self.space)


if __name__ == "__main__":
    unittest.main()
