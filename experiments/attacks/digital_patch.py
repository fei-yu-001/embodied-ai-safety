"""Reusable image-space digital patch primitives.

The attack prototype operates on the camera observation that a downstream policy
would consume.  It deliberately keeps the patch generation and placement logic
independent from RoboCasa so the same code can later be used with recorded
frames, a victim policy, or a defense module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Mapping, Sequence

import numpy as np


LocationName = Literal[
    "center",
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
]
Location = LocationName | Sequence[int]


@dataclass(frozen=True)
class PatchPlacement:
    """The resolved image-space bounding box for one patch application."""

    top: int
    left: int
    height: int
    width: int
    image_height: int
    image_width: int

    @property
    def bottom(self) -> int:
        """Exclusive bottom coordinate."""
        return self.top + self.height

    @property
    def right(self) -> int:
        """Exclusive right coordinate."""
        return self.left + self.width

    @property
    def area(self) -> int:
        return self.height * self.width

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-serializable representation."""
        return asdict(self) | {"bottom": self.bottom, "right": self.right}


def _as_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"expected an HxWx3 image, got shape={image.shape!r}"
        )
    if image.dtype != np.uint8:
        raise TypeError(f"expected uint8 RGB pixels in [0, 255], got {image.dtype}")
    if image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError(f"image dimensions must be positive, got {image.shape!r}")
    return image


def make_noise_patch(
    size: int | tuple[int, int],
    *,
    seed: int = 0,
    low: int = 0,
    high: int = 255,
) -> np.ndarray:
    """Create a deterministic uint8 RGB noise patch.

    ``high`` is inclusive, which makes ``low=0, high=255`` cover the complete
    uint8 range.  The generated patch is kept fixed across frames by the caller;
    a fixed patch is the digital analogue of one physical sticker.
    """
    if isinstance(size, int):
        height = width = size
    else:
        if len(size) != 2:
            raise ValueError(f"size must be an int or (height, width), got {size!r}")
        height, width = (int(size[0]), int(size[1]))

    if height < 1 or width < 1:
        raise ValueError(f"patch dimensions must be positive, got {(height, width)!r}")
    if not 0 <= low <= high <= 255:
        raise ValueError(
            f"noise bounds must satisfy 0 <= low <= high <= 255, got {low}, {high}"
        )

    rng = np.random.default_rng(seed)
    return rng.integers(
        low,
        high + 1,
        size=(height, width, 3),
        dtype=np.uint8,
    )


def resolve_placement(
    image_shape: Sequence[int],
    patch_shape: Sequence[int],
    location: Location = "center",
    *,
    offset: Sequence[int] = (0, 0),
) -> PatchPlacement:
    """Resolve a named or explicit patch location without clipping.

    Named locations use the image boundary as the anchor.  ``offset`` is added
    after the anchor and is useful for controlled sweeps in later experiments.
    An explicit location is interpreted as ``(top, left)``.
    """
    if len(image_shape) < 2 or len(patch_shape) < 2:
        raise ValueError("image_shape and patch_shape must contain height and width")

    image_height, image_width = int(image_shape[0]), int(image_shape[1])
    patch_height, patch_width = int(patch_shape[0]), int(patch_shape[1])
    if patch_height < 1 or patch_width < 1:
        raise ValueError("patch dimensions must be positive")
    if patch_height > image_height or patch_width > image_width:
        raise ValueError(
            "patch must fit inside the image: "
            f"image={(image_height, image_width)}, patch={(patch_height, patch_width)}"
        )
    if len(offset) != 2:
        raise ValueError(f"offset must be (dy, dx), got {offset!r}")
    offset_top, offset_left = int(offset[0]), int(offset[1])

    if isinstance(location, str):
        max_top = image_height - patch_height
        max_left = image_width - patch_width
        anchors: Mapping[str, tuple[int, int]] = {
            "center": (max_top // 2, max_left // 2),
            "top_left": (0, 0),
            "top_right": (0, max_left),
            "bottom_left": (max_top, 0),
            "bottom_right": (max_top, max_left),
        }
        if location not in anchors:
            choices = ", ".join(anchors)
            raise ValueError(f"unknown location {location!r}; choose one of {choices}")
        top, left = anchors[location]
    else:
        if len(location) != 2:
            raise ValueError(f"explicit location must be (top, left), got {location!r}")
        top, left = int(location[0]), int(location[1])

    top += offset_top
    left += offset_left
    if not (0 <= top <= image_height - patch_height):
        raise ValueError(
            f"patch top={top} is outside image bounds for height={image_height} "
            f"and patch height={patch_height}"
        )
    if not (0 <= left <= image_width - patch_width):
        raise ValueError(
            f"patch left={left} is outside image bounds for width={image_width} "
            f"and patch width={patch_width}"
        )

    return PatchPlacement(
        top=top,
        left=left,
        height=patch_height,
        width=patch_width,
        image_height=image_height,
        image_width=image_width,
    )


def apply_patch(
    image: np.ndarray,
    patch: np.ndarray,
    *,
    location: Location = "center",
    offset: Sequence[int] = (0, 0),
    alpha: float = 1.0,
) -> tuple[np.ndarray, PatchPlacement]:
    """Blend ``patch`` into ``image`` and return the modified image and bbox.

    The input arrays are never mutated.  ``alpha=1`` gives a hard replacement;
    smaller values model a partially transparent overlay while keeping all
    output pixels in the valid uint8 range.
    """
    image = _as_image(image)
    patch = _as_image(patch)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    placement = resolve_placement(
        image.shape,
        patch.shape,
        location=location,
        offset=offset,
    )

    output = image.copy()
    region = (slice(placement.top, placement.bottom),
              slice(placement.left, placement.right))
    blended = (
        (1.0 - alpha) * image[region].astype(np.float32)
        + alpha * patch.astype(np.float32)
    )
    output[region] = np.rint(np.clip(blended, 0.0, 255.0)).astype(np.uint8)
    return output, placement


def image_delta_metrics(
    clean: np.ndarray,
    attacked: np.ndarray,
) -> dict[str, float | int]:
    """Compute compact, JSON-friendly perturbation metrics."""
    clean = _as_image(clean)
    attacked = _as_image(attacked)
    if clean.shape != attacked.shape:
        raise ValueError(
            f"clean and attacked shapes differ: {clean.shape!r} vs {attacked.shape!r}"
        )

    delta = np.abs(
        attacked.astype(np.int16, copy=False) - clean.astype(np.int16, copy=False)
    )
    changed_pixels = np.any(delta != 0, axis=2)
    return {
        "mean_abs_delta": float(delta.mean()),
        "max_abs_delta": int(delta.max(initial=0)),
        "changed_pixels": int(changed_pixels.sum()),
        "changed_fraction": float(changed_pixels.mean()),
    }


__all__ = [
    "Location",
    "LocationName",
    "PatchPlacement",
    "apply_patch",
    "image_delta_metrics",
    "make_noise_patch",
    "resolve_placement",
]
