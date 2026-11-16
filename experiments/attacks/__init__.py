"""Attack injection components for the embodied AI safety experiments."""

from .digital_patch import (
    Location,
    LocationName,
    PatchPlacement,
    apply_patch,
    image_delta_metrics,
    make_noise_patch,
    resolve_placement,
)

__all__ = [
    "Location",
    "LocationName",
    "PatchPlacement",
    "apply_patch",
    "image_delta_metrics",
    "make_noise_patch",
    "resolve_placement",
]
