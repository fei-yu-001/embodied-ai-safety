"""Vision-driven endpoint policy used as the evaluation victim.

This is a colour-centred visual servo controller, not a VLA. It looks at the
selected camera, finds the largest blob of the target colour, and drives the
end effector to centre that blob in the image while lowering towards it. It is
deliberately simple: the point is that pixels are in the control loop, so a
patch on the image has a measurable effect on actions.

Honesty note: this policy does not solve RoboCasa pick-and-place. It is a
surrogate victim for measuring image-level attack and defence effects, and any
report must say so.
"""

from __future__ import annotations

import numpy as np

from experiments.sdk.contract import SDK_VERSION


class VisionCentroidPolicy:
    name = "vision_centroid"
    version = "1"
    sdk_version = SDK_VERSION
    uses_vision = True
    notes = (
        "颜色质心视觉伺服控制器，不是 VLA；用于验证像素进入控制回路，"
        "不能据此宣称 RoboCasa 任务成功率"
    )

    def __init__(self, action_space, camera_key="video.robot0_agentview_left", target="green", gain=0.6):
        self.action_space = action_space
        self.camera_key = camera_key
        self.target = target
        self.gain = float(gain)
        self.params = {"target": target, "gain": self.gain, "camera_key": camera_key}
        self.last_centroid = None
        self.last_area = 0

    def __call__(self, observation):
        image = observation.get(self.camera_key) if isinstance(observation, dict) else None
        action = {key: np.zeros(space.shape, dtype=space.dtype) for key, space in self.action_space.spaces.items()}
        if image is None:
            return action

        centroid = self._centroid(image)
        self.last_centroid = centroid
        self.last_area = 0 if centroid is None else centroid[2]
        if centroid is None:
            # Nothing visible: keep the gripper shut and stop moving, rather
            # than sweeping blindly.
            if "action.gripper_close" in action:
                action["action.gripper_close"][:] = 1.0
            return action

        cx, cy, area = centroid
        height, width = image.shape[:2]
        # Image error -> base-frame delta. Sign conventions verified against
        # the camera view: +X in base moves the arm to the right of the image.
        error_x = (cx - width / 2) / (width / 2)
        error_y = (cy - height / 2) / (height / 2)
        action["action.end_effector_position"][0] = float(np.clip(error_x * self.gain, -1, 1))
        action["action.end_effector_position"][1] = float(np.clip(-error_y * self.gain, -1, 1))
        action["action.end_effector_position"][2] = float(np.clip((0.35 - area) * self.gain, -0.8, 0.8))
        if "action.gripper_close" in action:
            action["action.gripper_close"][:] = 1.0
        return action

    def _centroid(self, image):
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3:
            return None
        import cv2

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lower, upper = _BANDS[self.target]
        mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count <= 1:
            return None
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = float(stats[best, cv2.CC_STAT_AREA])
        if area < _MIN_AREA:
            return None
        cx, cy = centroids[best]
        return float(cx), float(cy), area / float(image.shape[0] * image.shape[1])


_BANDS = {
    # OpenCV hue is 0-179. Green bottle, then two generic fallbacks.
    "green": ((35, 60, 40), (90, 255, 255)),
    "red": ((0, 80, 60), (10, 255, 255)),
    "blue": ((95, 80, 40), (130, 255, 255)),
}

_MIN_AREA = 30.0
