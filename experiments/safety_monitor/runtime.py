"""Small runtime monitors used by the integrated simulation pipeline."""

import numpy as np


class TemporalImageMonitor:
    """Flag large adjacent-frame changes, including ordinary camera motion.

    Only processed camera inputs are available here. This is a diagnostic
    threshold, not an adversarial detector or a physical harm assessment.
    """

    def __init__(self, threshold=12.0):
        self.threshold = float(threshold)
        self.previous = None

    def update(self, image):
        score = None
        if self.previous is not None:
            score = float(np.abs(image.astype(np.int16) - self.previous.astype(np.int16)).mean())
        self.previous = image.copy()
        return {
            "temporal_mean_abs_delta": score,
            "temporal_alert": score is not None and score > self.threshold,
        }


class ActionGuard:
    """Reject malformed, nonfinite or out-of-space actions before env.step."""

    def __init__(self, action_space):
        self.action_space = action_space

    def check(self, action):
        if not isinstance(action, dict) or set(action) != set(self.action_space.spaces):
            return False, "action_keys_mismatch"
        for key, space in self.action_space.spaces.items():
            try:
                value = np.asarray(action[key])
            except (TypeError, ValueError):
                return False, f"malformed_value:{key}"
            if value.shape != space.shape:
                return False, f"shape_mismatch:{key}"
            if not np.issubdtype(value.dtype, np.number) or not np.all(np.isfinite(value)):
                return False, f"nonfinite_or_nonnumeric:{key}"
            try:
                in_space = space.contains(action[key])
            except (TypeError, ValueError):
                return False, f"malformed_value:{key}"
            if not in_space:
                return False, f"outside_action_space:{key}"
        return True, None
