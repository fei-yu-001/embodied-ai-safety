"""Adaptive patches that target the defence's own assumptions.

The temporal defence assumes (a) a patch is a flat region (low local variance)
and (b) it is strongly saturated. These variants break exactly those two
assumptions while staying inside the victim's colour band:

  adaptive_texture  - noisy texture, so local variance is high
  adaptive_low_sat  - saturation just under the defence threshold, still inside
                      the victim's detection band
  adaptive_patch    - both, plus per-frame sub-pixel jitter

None of this is "cheating": an adaptive attacker is allowed to know the
defence it is attacking. That is precisely what §5.9 R1 requires us to test.
"""

from __future__ import annotations

import numpy as np

from experiments.attacks.observation import DigitalPatchWrapper


def _hsv_to_rgb(hsv):
    import cv2

    array = np.asarray(hsv, dtype=np.uint8).reshape(1, 1, 3)
    return cv2.cvtColor(array, cv2.COLOR_HSV2RGB).reshape(3)


def adaptive_texture(size=40, seed=0, noise=22):
    """Green-ish, strongly textured: high variance, still in the victim band."""
    rng = np.random.default_rng(seed)
    base = np.asarray(_hsv_to_rgb((55, 200, 190)), dtype=np.int16)
    noise_map = rng.integers(-noise, noise + 1, (size, size, 3), dtype=np.int16)
    patch = np.clip(base + noise_map, 0, 255).astype(np.uint8)
    return patch


def adaptive_low_sat(size=40, seed=0, saturation=70):
    """Saturation below the defence gate, above the victim's detection floor."""
    rng = np.random.default_rng(seed)
    hsv = np.zeros((size, size, 3), dtype=np.uint8)
    hsv[:, :, 0] = 55
    hsv[:, :, 1] = saturation + rng.integers(-6, 7, (size, size))
    hsv[:, :, 2] = 190 + rng.integers(-20, 21, (size, size))
    import cv2

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def adaptive_patch(size=40, seed=0):
    """Both tricks at once: low saturation plus texture."""
    rng = np.random.default_rng(seed)
    patch = adaptive_low_sat(size=size, seed=seed, saturation=72)
    texture = rng.integers(-18, 19, (size, size, 3), dtype=np.int16)
    return np.clip(patch.astype(np.int16) + texture, 0, 255).astype(np.uint8)


class JitteredPatchWrapper(DigitalPatchWrapper):
    """Apply the patch at a slightly different offset every frame."""

    def __init__(self, env, patch, *, jitter=4, seed=0, **kwargs):
        super().__init__(env, patch, **kwargs)
        self.jitter = int(jitter)
        self._rng = np.random.default_rng(seed)
        self._offsets = []

    def _transform(self, observation):
        if self.jitter > 0:
            dx = int(self._rng.integers(-self.jitter, self.jitter + 1))
            dy = int(self._rng.integers(-self.jitter, self.jitter + 1))
        else:
            dx = dy = 0
        self._offsets.append((dx, dy))
        self.offset = (dx, dy) if self.location == "center" else self.offset
        return super()._transform(observation)
