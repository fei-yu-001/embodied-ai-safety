"""Versioned algorithm contracts.

A policy only sees the observation it is handed and returns a raw action dict.
It must not touch the simulator, and it must declare whether it looks at pixels
so that reports cannot quietly present a blind controller as a vision model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SDK_VERSION = "1"


@dataclass(frozen=True)
class PolicyManifest:
    name: str
    version: str
    sdk_version: str
    uses_vision: bool
    camera_key: str | None
    notes: str
    params: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "name": self.name,
            "version": self.version,
            "sdk_version": self.sdk_version,
            "uses_vision": self.uses_vision,
            "camera_key": self.camera_key,
            "notes": self.notes,
            "params": dict(self.params),
        }


class PolicyContractError(Exception):
    pass


def validate_action(action, action_space):
    """Shape/space check independent of any particular policy."""
    import numpy as np

    if not isinstance(action, dict):
        raise PolicyContractError("动作必须是字典")
    if set(action) != set(action_space.spaces):
        raise PolicyContractError("动作键与动作空间不一致")
    for key, space in action_space.spaces.items():
        value = action[key]
        try:
            array = np.asarray(value)
        except (TypeError, ValueError) as error:
            raise PolicyContractError(f"{key} 不是数值") from error
        if array.shape != space.shape:
            raise PolicyContractError(f"{key} 形状应为 {space.shape}，实际 {array.shape}")
        if not np.all(np.isfinite(array)):
            raise PolicyContractError(f"{key} 含非有限值")
        if not space.contains(value):
            raise PolicyContractError(f"{key} 超出动作空间")
    return action


def require_manifest(policy):
    for attribute in ("name", "version", "uses_vision"):
        if not hasattr(policy, attribute):
            raise PolicyContractError(f"策略缺少 {attribute}")
    return PolicyManifest(
        name=policy.name,
        version=getattr(policy, "version", "0"),
        sdk_version=getattr(policy, "sdk_version", SDK_VERSION),
        uses_vision=bool(policy.uses_vision),
        camera_key=getattr(policy, "camera_key", None),
        notes=getattr(policy, "notes", ""),
        params=dict(getattr(policy, "params", {}) or {}),
    )
