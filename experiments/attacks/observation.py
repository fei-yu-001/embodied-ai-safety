"""Inject a fixed digital patch into the observation sent to a Gym policy."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from .digital_patch import Location, apply_patch, image_delta_metrics


class DigitalPatchWrapper(gym.Wrapper):
    """Patch reset/step observations, leaving rewards and simulator state intact.

    Only the selected RGB camera is replaced.  Clean images and perturbation
    metadata are kept on the wrapper for evaluation, outside the observation
    passed to a policy.  Other observation values are shared and must be treated
    as read-only, following the normal Gym observation convention.
    """

    def __init__(
        self,
        env: gym.Env,
        patch: np.ndarray,
        *,
        camera_key: str = "video.robot0_agentview_left",
        location: Location = "center",
        offset: tuple[int, int] = (0, 0),
        alpha: float = 1.0,
    ) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise TypeError("DigitalPatchWrapper requires a Dict observation space")
        if camera_key not in env.observation_space.spaces:
            available = ", ".join(env.observation_space.spaces)
            raise KeyError(f"camera {camera_key!r} is unavailable; keys: {available}")
        camera_space = env.observation_space[camera_key]
        if not isinstance(camera_space, gym.spaces.Box):
            raise TypeError("selected camera must have a Box observation space")
        # Validate placement and pixel format before running a costly rollout.
        patch = np.asarray(patch)
        apply_patch(
            np.zeros(camera_space.shape, dtype=camera_space.dtype),
            patch,
            location=location,
            offset=offset,
            alpha=alpha,
        )
        self.patch = patch.copy()
        self.camera_key = camera_key
        self.location = location
        self.offset = offset
        self.alpha = alpha
        self.last_clean_image: np.ndarray | None = None
        self.last_metrics: dict = {}
        self.placement = None

    def _transform(self, observation: dict) -> dict:
        self.last_clean_image = observation[self.camera_key].copy()
        attacked, self.placement = apply_patch(
            self.last_clean_image,
            self.patch,
            location=self.location,
            offset=self.offset,
            alpha=self.alpha,
        )
        self.last_metrics = image_delta_metrics(self.last_clean_image, attacked)
        result = dict(observation)
        result[self.camera_key] = attacked
        return result

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        return self._transform(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._transform(observation), reward, terminated, truncated, info
