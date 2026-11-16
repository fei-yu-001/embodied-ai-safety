"""Camera-only filtering without access to a clean image or attack mask."""

from time import perf_counter

import cv2
import gymnasium as gym
import numpy as np


class ImageDefense:
    def __init__(self, name="none", kernel=5):
        if name not in ("none", "median", "temporal"):
            raise ValueError(f"unknown defense: {name}")
        if name == "temporal":
            from experiments.defenses.temporal import TemporalConsistencyDefense

            self.impl = TemporalConsistencyDefense()
            self.name = name
            self.kernel = kernel
            return
        if kernel not in (3, 5, 7):
            raise ValueError("median kernel must be 3, 5 or 7")
        self.name = name
        self.kernel = kernel

    def __call__(self, image):
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise TypeError("defense requires HxWx3 uint8 RGB")
        if self.name == "none":
            return image.copy()
        if self.name == "temporal":
            return self.impl(image)
        return cv2.medianBlur(image, self.kernel)


class DefenseWrapper(gym.Wrapper):
    """Apply a baseline filter to every observation before policy inference."""

    def __init__(self, env, defense, camera_key):
        super().__init__(env)
        self.defense = defense
        self.camera_key = camera_key
        self.last_input_image = None
        self.last_latency_ms = 0.0

    def _transform(self, observation):
        started = perf_counter()
        image = observation[self.camera_key]
        result = dict(observation)
        result[self.camera_key] = self.defense(image)
        self.last_latency_ms = (perf_counter() - started) * 1000
        self.last_input_image = image.copy()
        return result

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        return self._transform(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._transform(observation), reward, terminated, truncated, info
