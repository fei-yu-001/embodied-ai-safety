"""Temporal-consistency defence for colour-flat adversarial patches.

Rationale: a printed patch is a flat, high-saturation region that persists in
the same image location across consecutive frames while the rest of the scene
moves with the robot. Natural surfaces are textured; a uniform blob is not.

The defence is stateful on purpose: the temporal requirement is the part that
separates this from a median filter. It reports what it did so a report can
show the mask rate instead of claiming "defended".
"""

from __future__ import annotations

import cv2
import numpy as np

SCALE = 64
WINDOW = 5


class TemporalConsistencyDefense:
    name = "temporal"

    def __init__(self, variance_threshold=12.0, saturation_threshold=90, persist_frames=2, dilate=3):
        self.variance_threshold = float(variance_threshold)
        self.saturation_threshold = int(saturation_threshold)
        self.persist_frames = max(1, int(persist_frames))
        self.dilate = max(1, int(dilate))
        self.previous = None
        self.persistence = None
        self.last_mask_fraction = 0.0
        self.last_flagged = False
        self.last_patch_mask = None

    def __call__(self, image):
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise TypeError("时序防御需要 HxWx3 uint8 RGB")

        small = cv2.resize(image, (SCALE, SCALE), interpolation=cv2.INTER_AREA)
        blob = self._flat_colour_mask(small)
        if self.persistence is None or self.persistence.shape != blob.shape:
            self.persistence = np.zeros_like(blob, dtype=np.float32)
        # Accumulate evidence instead of trusting one frame.
        self.persistence = 0.5 * self.persistence + blob.astype(np.float32) * 0.5
        flagged = self.persistence >= (0.5 if self.persist_frames <= 1 else 0.6)

        cleaned = image.copy()
        if flagged.any():
            mask = cv2.resize(flagged.astype(np.uint8) * 255, (image.shape[1], image.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
            if self.dilate > 1:
                mask = cv2.dilate(mask, np.ones((self.dilate, self.dilate), np.uint8))
            # 大块纯色补丁在上一帧里也在，不能用上一帧补自己；用掩码外围的
            # 中位色填回去，直接抹掉它作为“目标”的颜色特征。
            cleaned = self._fill_from_ring(image, mask)
            self.last_mask_fraction = float(np.count_nonzero(mask) / mask.size)
            self.last_patch_mask = mask
            self.last_flagged = True
        else:
            self.last_mask_fraction = 0.0
            self.last_patch_mask = None
            self.last_flagged = False

        self.previous = image.copy()
        return cleaned

    def _fill_from_ring(self, image, mask):
        # 掩码在 64×64 上算出来，边缘会欠覆盖；先外扩保证整块补丁在掩码内，
        # 再取更外一圈的真实纹理当参考色。
        mask = cv2.dilate(mask, np.ones((11, 11), np.uint8))
        ring = cv2.dilate(mask, np.ones((25, 25), np.uint8)) > 0
        ring &= mask == 0
        filled = image.copy()
        if np.any(ring):
            colour = np.median(image[ring], axis=0).astype(np.uint8)
            filled[mask > 0] = colour
        else:
            filled[mask > 0] = 128
        return filled

    def _flat_colour_mask(self, small):
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1].astype(np.float32)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
        mean = cv2.blur(gray, (WINDOW, WINDOW))
        mean_sq = cv2.blur(gray * gray, (WINDOW, WINDOW))
        variance = np.clip(mean_sq - mean * mean, 0, None)
        flat = (variance < self.variance_threshold) & (saturation > self.saturation_threshold)
        return flat.astype(np.uint8)

    def reset(self):
        self.previous = None
        self.persistence = None
        self.last_mask_fraction = 0.0
        self.last_flagged = False
        self.last_patch_mask = None
