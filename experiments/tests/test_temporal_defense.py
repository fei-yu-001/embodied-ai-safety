"""Temporal-consistency defence on synthetic frames.

Checks the two claims that matter: a flat colour blob gets suppressed, and a
textured scene is left alone (no self-inflicted damage).
"""

import unittest

import numpy as np

from experiments.defenses.temporal import TemporalConsistencyDefense


def textured_scene(seed=0, size=128):
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 200, (size, size, 3), dtype=np.uint8)
    return base


def scene_with_blob(size=128, blob=(52, 40, 40, 40)):
    image = textured_scene(1, size)
    x, y, w, h = blob
    image[y:y + h, x:x + w] = (40, 200, 60)
    return image


class TemporalDefenceTests(unittest.TestCase):
    def test_flat_colour_blob_is_suppressed_after_persistence(self):
        defence = TemporalConsistencyDefense()
        frame = scene_with_blob()
        first = defence(frame)
        self.assertFalse(defence.last_flagged, "单帧不应立刻下结论")
        second = defence(frame)
        self.assertTrue(defence.last_flagged, "持续出现的纯色块应被标记")
        self.assertGreater(defence.last_mask_fraction, 0.0)
        # 被遮住的像素不再是最初那块纯绿：直接看补丁区域
        self.assertFalse(np.array_equal(second, frame))
        blob = second[40:80, 52:92].astype(int)
        original = frame[40:80, 52:92].astype(int)
        self.assertGreater(float(np.abs(blob - original).mean()), 20.0)
        # 纯绿像素占比明显下降
        green = (blob[:, :, 1] > 180) & (blob[:, :, 0] < 90)
        self.assertLess(float(green.mean()), 0.5)

    def test_textured_scene_is_left_untouched(self):
        defence = TemporalConsistencyDefense()
        frame = textured_scene(3)
        for _ in range(3):
            cleaned = defence(frame)
        self.assertFalse(defence.last_flagged)
        self.assertTrue(np.array_equal(cleaned, frame))

    def test_uniform_low_saturation_surface_is_not_flagged(self):
        defence = TemporalConsistencyDefense()
        wall = np.full((128, 128, 3), 210, dtype=np.uint8)  # 白墙：低方差、低饱和
        for _ in range(3):
            defence(wall)
        self.assertFalse(defence.last_flagged, "白墙不能被当成补丁")

    def test_reset_clears_temporal_state(self):
        defence = TemporalConsistencyDefense()
        frame = scene_with_blob()
        defence(frame)
        defence(frame)
        self.assertTrue(defence.last_flagged)
        defence.reset()
        defence(frame)
        self.assertFalse(defence.last_flagged)

    def test_bad_input_is_rejected(self):
        defence = TemporalConsistencyDefense()
        for bad in (np.zeros((8, 8), dtype=np.uint8), np.zeros((8, 8, 3), dtype=np.float32), "nope"):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    defence(bad)

    def test_factory_can_build_the_temporal_defence(self):
        from experiments.defenses.filters import ImageDefense

        defence = ImageDefense("temporal", 5)
        frame = scene_with_blob()
        defence(frame)
        cleaned = defence(frame)
        self.assertEqual(cleaned.shape, frame.shape)


if __name__ == "__main__":
    unittest.main()
