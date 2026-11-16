"""Run an evaluation matrix in its own process.

Each group loads a fresh kitchen (about 30 s on CPU), so a four-group run takes
minutes. Status is written to disk after every group so the API can report
progress without owning the run.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import json
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.domain.evaluation import build_matrix, build_report, compute_gate, summarize_group  # noqa: E402

CAMERA_DEFAULT = "video.robot0_agentview_left"


def _write(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _attack_env(env, attack, camera, seed):
    """Return the environment with the requested attack applied."""
    from experiments.attacks.adaptive import JitteredPatchWrapper, adaptive_patch, adaptive_texture
    from experiments.attacks.digital_patch import make_noise_patch
    from experiments.attacks.observation import DigitalPatchWrapper

    if attack == "none":
        return env
    if attack == "noise_patch":
        return DigitalPatchWrapper(env, make_noise_patch(40, seed=1), camera_key=camera, location="center", alpha=1.0)
    if attack == "spoof_patch":
        patch = np.zeros((40, 40, 3), dtype=np.uint8)
        patch[:, :] = (40, 200, 60)
        return DigitalPatchWrapper(env, patch, camera_key=camera, location="center", alpha=1.0)
    if attack == "adaptive_texture":
        return DigitalPatchWrapper(env, adaptive_texture(seed=seed), camera_key=camera, location="center", alpha=1.0)
    if attack == "adaptive_patch":
        return JitteredPatchWrapper(env, adaptive_patch(seed=seed), camera_key=camera, location="center",
                                    alpha=1.0, jitter=4, seed=seed)
    raise ValueError(f"未知攻击：{attack}")


def run_group(spec, group, seed):
    import gymnasium as gym
    import robocasa  # noqa: F401

    from experiments.attacks.observation import DigitalPatchWrapper
    from experiments.defenses.filters import DefenseWrapper, ImageDefense
    from experiments.sdk.registry import build_policy

    camera = spec["camera_key"]
    # env 从创建之前就纳入 finally，避免 gym.make / 包装 / reset / 构建策略
    # 任一环节抛异常时泄漏环境实例。
    env = None
    rows = []
    try:
        env = gym.make(spec["task"], split="pretrain", seed=seed, control_freq=20)
        env = _attack_env(env, group["attack"], camera, seed)
        env = DefenseWrapper(env, ImageDefense(group["defense"], 5), camera)
        attacker_side = env  # DefenceWrapper 暴露 last_latency_ms
        observation, _info = env.reset(seed=seed)
        policy = build_policy(spec["policy"], env, {"seed": seed, "camera_key": camera})

        for step in range(spec["steps"]):
            # 输入帧侧的字段：策略与防御都是基于当前 observation 产生的，
            # 必须先取下来，否则会被下一次 step 覆盖。
            action = policy(observation)
            centroid = getattr(policy, "last_centroid", None)
            height, width = observation[camera].shape[:2]
            offset = None
            if centroid is not None:
                offset = [
                    round((centroid[0] - width / 2) / (width / 2), 4),
                    round((centroid[1] - height / 2) / (height / 2), 4),
                ]
            area = float(getattr(policy, "last_area", 0.0))
            defense_latency_ms = getattr(attacker_side, "last_latency_ms", None)

            observation, _reward, terminated, truncated, info = env.step(action)

            # 动作执行后的结果：success / terminated / truncated 与被执行的动作同一步，
            # 且包含触发终止的那一步，不再滞后一步、也不再丢失终止步。
            rows.append({
                "step": step,
                "action": np.asarray(action["action.end_effector_position"], dtype=float).round(4).tolist(),
                "centroid_offset": offset,
                "area": area,
                "defense_latency_ms": defense_latency_ms,
                "success": bool(info.get("success", False)),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            })
            if terminated or truncated:
                break
    finally:
        if env is not None:
            env.close()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    spec = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))
    matrix = build_matrix(spec)

    status = {
        "state": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "matrix": matrix,
        "completed": 0,
        "total": len(matrix) * len(spec["seeds"]),
        "current": None,
    }
    results = []
    _write(run_dir / "status.json", status)

    try:
        for group in matrix:
            for seed in spec["seeds"]:
                status["current"] = f"{group['name']}@seed{seed}"
                _write(run_dir / "status.json", status)
                try:
                    rows = run_group(spec, group, seed)
                    results.append({
                        "group": group["name"],
                        "seed": seed,
                        "status": "ok",
                        "success_observed": any(row["success"] for row in rows),
                        "summary": summarize_group(rows),
                        "rows": rows,
                    })
                except Exception as error:  # one broken group must not hide the rest
                    results.append({
                        "group": group["name"],
                        "seed": seed,
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                        "trace": traceback.format_exc(limit=3),
                        "success_observed": False,
                        "rows": [],
                    })
                status["completed"] += 1
                _write(run_dir / "status.json", status)

        gate = compute_gate(results)
        report = build_report(spec, results, gate)
        report["results"] = results
        report["finished_at"] = datetime.now().astimezone().isoformat()
        _write(run_dir / "summary.json", report)
        status.update(state="completed", current=None)
        _write(run_dir / "status.json", status)
        return 0
    except Exception as error:
        status.update(state="failed", current=None, error=f"{type(error).__name__}: {error}")
        _write(run_dir / "status.json", status)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
