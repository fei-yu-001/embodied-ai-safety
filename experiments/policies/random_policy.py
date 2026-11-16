"""Replace this adapter with a trained policy once a clean baseline is ready."""


class SeededRandomPolicy:
    name = "seeded_random_with_fixed_base"
    uses_vision = False

    def __init__(self, action_space, seed, action_scale=1.0):
        self.action_scale = action_scale
        self.action_space = action_space
        self.action_space.seed(seed)

    def __call__(self, observation):
        if not isinstance(observation, dict):
            raise TypeError("policy requires a dictionary observation")
        action = self.action_space.sample()
        action["action.base_motion"][:] = 0.0
        action["action.end_effector_position"] *= self.action_scale
        action["action.end_effector_rotation"] *= self.action_scale
        return action


class HoldPolicy:
    name = "zero_delta_hold"
    uses_vision = False

    def __init__(self, action_space):
        self.action_space = action_space

    def __call__(self, observation):
        import numpy as np
        return {key: np.zeros(space.shape, dtype=space.dtype) for key, space in self.action_space.spaces.items()}
