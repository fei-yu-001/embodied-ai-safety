"""Run the RoboCasa observation-patch prototype and save auditable artifacts.

From the workspace root (PowerShell):
    & 'D:/ruangong/conda-envs/robocasa/python.exe' experiments/scripts/run_patch_attack.py

The seeded random policy exercises the observation pipeline. It is not a victim
model, so the output does not measure adversarial attack success or task loss.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import sys
from time import perf_counter

import cv2
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np

# Support running the script directly from any working directory.
EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from attacks.digital_patch import make_noise_patch
from attacks.observation import DigitalPatchWrapper


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="robocasa/PickPlaceCounterToCabinet")
    parser.add_argument("--split", choices=("pretrain", "target", "all"), default="pretrain")
    parser.add_argument("--steps", type=positive_int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patch-seed", type=int, default=1)
    parser.add_argument("--patch-size", type=positive_int, default=50)
    parser.add_argument("--camera-key", default="video.robot0_agentview_left")
    parser.add_argument(
        "--location",
        choices=("center", "top_left", "top_right", "bottom_left", "bottom_right"),
        default="center",
    )
    parser.add_argument("--offset", type=int, nargs=2, metavar=("DY", "DX"), default=(0, 0))
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--save-every", type=positive_int, default=25)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.seed < 0 or args.patch_seed < 0:
        parser.error("seeds must be nonnegative")
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be in [0, 1]")
    return args


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, default=json_default, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def describe_observation(observation: dict) -> dict:
    schema = {}
    for key, value in observation.items():
        if isinstance(value, np.ndarray):
            schema[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        else:
            schema[key] = {"type": type(value).__name__, "value": value}
    return schema


def comparison_frame(clean: np.ndarray, attacked: np.ndarray, step: int) -> np.ndarray:
    """Add labels only to the side-by-side video, never to the policy input."""
    header_height = 32
    height, width = clean.shape[:2]
    frame = np.full((height + header_height, width * 2, 3), 24, dtype=np.uint8)
    frame[header_height:, :width] = clean
    frame[header_height:, width:] = attacked
    for text, x in ((f"CLEAN | t={step}", 8), ("PATCHED | random noise", width + 8)):
        cv2.putText(frame, text, (x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
    return frame


class SeededRandomPolicy:
    """Plumbing placeholder: observes patched inputs, samples independent actions."""

    def __init__(self, action_space: gym.spaces.Dict, seed: int) -> None:
        self.action_space = action_space
        self.action_space.seed(seed)
        self.observation_calls = 0

    def __call__(self, observation: dict) -> dict:
        # Replace this function with victim-model inference for attack evaluation.
        if not isinstance(observation, dict):
            raise TypeError("policy expects a Dict observation")
        self.observation_calls += 1
        action = self.action_space.sample()
        # Match the existing smoke demo: prevent the mobile base from jittering.
        action["action.base_motion"][:] = 0.0
        return action


def run(args: argparse.Namespace) -> Path:
    # Import only when actually running; --help does not initialize MuJoCo.
    import robocasa  # noqa: F401 -- registers Gym environments

    random.seed(args.seed)
    np.random.seed(args.seed)
    started = perf_counter()
    output_dir = args.output_dir or (
        EXPERIMENTS_DIR / "scripts" / "output" / f"patch_attack_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    output_dir = output_dir.resolve()
    # Each run owns a fresh directory; never overwrite a prior experiment.
    output_dir.mkdir(parents=True, exist_ok=False)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir()
    patch = make_noise_patch(args.patch_size, seed=args.patch_seed)
    imageio.imwrite(output_dir / "patch.png", patch)
    write_json(output_dir / "config.json", vars(args) | {"output_dir": output_dir})
    print(f"[1/3] Creating {args.env_id}; output: {output_dir}", flush=True)

    raw_env = gym.make(args.env_id, split=args.split, seed=args.seed)
    try:
        env = DigitalPatchWrapper(
            raw_env,
            patch,
            camera_key=args.camera_key,
            location=args.location,
            offset=tuple(args.offset),
            alpha=args.alpha,
        )
        policy = SeededRandomPolicy(env.action_space, args.seed)
        observation, info = env.reset(seed=args.seed)
        schema = describe_observation(observation)
        write_json(output_dir / "observation_schema.json", schema)
        placement = env.placement
        mask = np.zeros(env.last_clean_image.shape[:2], dtype=np.uint8)
        mask[placement.top:placement.bottom, placement.left:placement.right] = 255
        imageio.imwrite(output_dir / "patch_mask.png", mask)
        kitchen = raw_env.unwrapped.env
        fps = int(kitchen.control_freq)
        episode_metadata = kitchen.get_ep_meta()
        # Scene and initial physics state make later replay possible without
        # relying solely on package-specific RNG behavior.
        (output_dir / "initial_scene.xml").write_text(kitchen.sim.model.get_xml(), encoding="utf-8")
        np.savez_compressed(output_dir / "initial_state.npz", state=kitchen.sim.get_state().flatten())

        print(f"      Camera: {args.camera_key}, shape={env.last_clean_image.shape}, RGB uint8", flush=True)
        print(f"      Patch bbox [left, top, right, bottom]: {[placement.left, placement.top, placement.right, placement.bottom]}", flush=True)
        print(f"[2/3] Running {args.steps} steps with a random policy ...", flush=True)
        rows = []
        success_seen = bool(info.get("success", False))
        stop_reason = "step_limit"
        with ExitStack() as stack:
            writers = {}
            for name in ("clean", "attacked", "comparison"):
                writer = imageio.get_writer(
                    output_dir / f"{name}.mp4", fps=fps, codec="libx264",
                    quality=8, macro_block_size=1,
                )
                stack.callback(writer.close)
                writers[name] = writer
            metrics_file = stack.enter_context((output_dir / "metrics.jsonl").open("w", encoding="utf-8"))
            actions_file = stack.enter_context((output_dir / "actions.jsonl").open("w", encoding="utf-8"))

            def record(step, current, reward, terminated, truncated, current_info, final=False):
                clean = env.last_clean_image
                attacked = current[args.camera_key]
                combined = comparison_frame(clean, attacked, step)
                writers["clean"].append_data(clean)
                writers["attacked"].append_data(attacked)
                writers["comparison"].append_data(combined)
                row = {
                    "step": step, "simulation_time_seconds": step / fps,
                    "reward": float(reward), "terminated": bool(terminated),
                    "truncated": bool(truncated), "success": bool(current_info.get("success", False)),
                    **env.last_metrics,
                }
                rows.append(row)
                metrics_file.write(json.dumps(row, allow_nan=False) + "\n")
                if step % args.save_every == 0 or final:
                    for label, image in (("clean", clean), ("attacked", attacked), ("comparison", combined)):
                        imageio.imwrite(frames_dir / f"{step:04d}_{label}.png", image)

            record(0, observation, 0.0, False, False, info)
            for step in range(1, args.steps + 1):
                action = policy(observation)  # This is the patched observation.
                actions_file.write(json.dumps({"step": step, "action": action}, default=json_default) + "\n")
                observation, reward, terminated, truncated, info = env.step(action)
                success_seen = success_seen or bool(info.get("success", False))
                final = bool(terminated or truncated or step == args.steps)
                record(step, observation, reward, terminated, truncated, info, final=final)
                if step % 25 == 0 or final:
                    print(f"      step {step}/{args.steps}; changed pixels={env.last_metrics['changed_pixels']}", flush=True)
                if terminated or truncated:
                    stop_reason = "terminated" if terminated else "truncated"
                    break

        summary = {
            "experiment": "digital_noise_patch_pipeline_smoke",
            "created_at": datetime.now().astimezone().isoformat(),
            "env_id": args.env_id, "split": args.split,
            "seed": args.seed, "patch_seed": args.patch_seed,
            "policy": "seeded_random_with_fixed_base",
            "interpretation": "Validates observation injection only; no victim model or attack-success metric.",
            "camera_key": args.camera_key, "image_shape": list(env.last_clean_image.shape),
            "image_orientation": "RoboCasa Gym RGB observations are already upright; no extra flip applied.",
            "patch": {
                "kind": "fixed_uniform_random_noise", "optimized": False,
                "alpha": args.alpha, "placement": placement.to_dict(),
                "sha256_rgb": hashlib.sha256(patch.tobytes()).hexdigest(),
                "area_fraction": placement.area / (placement.image_height * placement.image_width),
            },
            "steps_requested": args.steps, "steps_completed": len(rows) - 1,
            "frames_written": len(rows), "fps": fps, "stop_reason": stop_reason,
            "success_observed": success_seen,
            "elapsed_seconds": round(perf_counter() - started, 3),
            "mean_abs_delta": float(np.mean([row["mean_abs_delta"] for row in rows])),
            "mean_changed_fraction": float(np.mean([row["changed_fraction"] for row in rows])),
            "min_changed_pixels": min(row["changed_pixels"] for row in rows),
            "max_changed_pixels": max(row["changed_pixels"] for row in rows),
            "episode_metadata": episode_metadata,
            "versions": {name: version(name) for name in (
                "robocasa", "robosuite", "mujoco", "gymnasium", "numpy", "imageio", "imageio-ffmpeg",
            )},
            "source_sha256": {
                str(path.relative_to(EXPERIMENTS_DIR)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (Path(__file__), EXPERIMENTS_DIR / "attacks" / "digital_patch.py", EXPERIMENTS_DIR / "attacks" / "observation.py")
            },
        }
        write_json(output_dir / "summary.json", summary)
        print(f"[3/3] Saved {len(rows)} frames in each video; comparison: {output_dir / 'comparison.mp4'}", flush=True)
        print(f"      Mean changed fraction: {summary['mean_changed_fraction']:.6f}; no victim-model claim.", flush=True)
    finally:
        raw_env.close()
    return output_dir


if __name__ == "__main__":
    run(parse_args())
