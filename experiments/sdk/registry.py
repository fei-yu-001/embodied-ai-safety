"""Policy/attack/defense registry.

Plugins are declared here, not discovered by importing arbitrary modules: a
name in a config must map to code that was reviewed in the repository.
"""

from __future__ import annotations

from experiments.sdk.contract import PolicyContractError


def build_policy(name, env, config):
    """Return a policy instance for `name`."""
    if name == "random":
        from experiments.policies.random_policy import SeededRandomPolicy

        return SeededRandomPolicy(env.action_space, config.get("seed", 0))
    if name == "hold":
        from experiments.policies.random_policy import HoldPolicy

        return HoldPolicy(env.action_space)
    if name == "vision_centroid":
        from experiments.policies.vision_centroid import VisionCentroidPolicy

        return VisionCentroidPolicy(
            env.action_space,
            camera_key=config.get("camera_key", "video.robot0_agentview_left"),
            target=config.get("vision_target", "green"),
            gain=float(config.get("vision_gain", 0.6)),
        )
    raise PolicyContractError(f"未注册的策略：{name}")


def policy_names():
    return ["random", "hold", "vision_centroid"]
