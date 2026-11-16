"""Pair experiment: does a pixel patch change what the vision policy does?

Three groups share the same seed and step budget:
  clean       - untouched camera
  noise       - random-noise patch (no colour relationship to the target)
  spoof_green - a green square, i.e. a fake object for a colour detector

Reported metrics are image-space and action-space. Task success is recorded as
returned by the environment, and is NOT interpreted as attack success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.attacks.digital_patch import image_delta_metrics, make_noise_patch  # noqa: E402
from experiments.attacks.observation import DigitalPatchWrapper  # noqa: E402
from experiments.policies.vision_centroid import VisionCentroidPolicy  # noqa: E402

CAMERA = "video.robot0_agentview_left"


def green_patch(size=40):
    patch = np.zeros((size, size, 3), dtype=np.uint8)
    patch[:, :] = (40, 200, 60)
    return patch


def run_group(name, patch, args):
    import gymnasium as gym
    import robocasa  # noqa: F401

    env = gym.make(args.task, split="pretrain", seed=args.seed, control_freq=20)
    wrapped = env
    if patch is not None:
        wrapped = DigitalPatchWrapper(env, patch, camera_key=CAMERA, location="center", alpha=1.0)
    observation, info = wrapped.reset(seed=args.seed)
    policy = VisionCentroidPolicy(wrapped.action_space, camera_key=CAMERA)

    success = False
    rows = []
    for step in range(args.steps):
        action = policy(observation)
        centroid = policy.last_centroid
        position = np.asarray(action["action.end_effector_position"], dtype=float)
        height, width = observation[CAMERA].shape[:2]
        offset = None if centroid is None else [(centroid[0] - width / 2) / (width / 2), (centroid[1] - height / 2) / (height / 2)]
        rows.append({
            "step": step,
            "action": position.round(4).tolist(),
            "centroid_offset": None if offset is None else np.round(offset, 4).tolist(),
            "area": policy.last_area,
            "success": bool(info.get("success", False)),
        })
        observation, _reward, terminated, truncated, info = wrapped.step(action)
        success = success or bool(info.get("success", False))
        if terminated or truncated:
            break

    env.close()
    return {
        "group": name,
        "steps": len(rows),
        "success_observed": success,
        "mean_abs_action_x": float(np.mean([abs(row["action"][0]) for row in rows])),
        "mean_offset_x": float(np.mean([abs(row["centroid_offset"][0]) for row in rows if row["centroid_offset"]])),
        "final_offset_x": rows[-1]["centroid_offset"][0] if rows[-1]["centroid_offset"] else None,
        "action_x_series": [row["action"][0] for row in rows],
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="robocasa/PickPlaceCounterToCabinet")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    outputs = {}
    noise = make_noise_patch(40, seed=1)
    for name, patch in (("clean", None), ("noise", noise), ("spoof_green", green_patch())):
        print(f"running {name} ...", flush=True)
        result = run_group(name, patch, args)
        metrics = image_delta_metrics
        outputs[name] = result
        print(
            f"  {name}: steps={result['steps']} success={result['success_observed']} "
            f"mean|ax|={result['mean_abs_action_x']:.4f} final_offset_x={result['final_offset_x']}",
            flush=True,
        )

    clean = outputs["clean"]
    for name in ("noise", "spoof_green"):
        series = outputs[name]["action_x_series"]
        baseline = clean["action_x_series"][: len(series)]
        divergence = float(np.mean(np.abs(np.subtract(series, baseline)))) if baseline else None
        outputs[name]["action_divergence_vs_clean"] = divergence
        print(f"  divergence {name} vs clean: {divergence}")

    clean_success = 1.0 if clean["success_observed"] else 0.0
    threshold = 0.2
    eligible = clean_success >= threshold
    gate = {
        "clean_success_rate": clean_success,
        "threshold": threshold,
        "eligible_for_task_eval": eligible,
        "reason": (
            "干净基线达到阈值，可进入任务级评测"
            if eligible
            else "干净基线未达阈值：该策略不能用于任务级攻防结论，只能用于像素-动作回路验证"
        ),
    }
    print("gate:", gate["eligible_for_task_eval"], gate["reason"])

    summary = {
        "task": args.task,
        "seed": args.seed,
        "steps": args.steps,
        "camera": CAMERA,
        "policy": "vision_centroid (颜色质心视觉伺服，不是 VLA)",
        "interpretation": (
            "这里衡量的是像素是否影响动作（图像空间偏移与动作散度）。"
            "success_observed 直接来自环境，若为 false 就不能声称任务成功或攻击成功。"
        ),
        "gate": gate,
        "groups": {name: {key: value for key, value in result.items() if key != "rows"} for name, result in outputs.items()},
        "rows": {name: result["rows"] for name, result in outputs.items()},
    }
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("written", target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
