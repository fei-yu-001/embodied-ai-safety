"""Resident RoboCasa session and finite batch experiment worker."""

from contextlib import ExitStack
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
from time import perf_counter, sleep, time
import traceback

import cv2
import imageio.v2 as imageio
import numpy as np

from .jobs import read_json
from .storage import replace_snapshot
from .scene_runtime import PhysicsParameters, OrbitCamera, describe_scene

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def write_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, default=json_default, ensure_ascii=False,
                                    allow_nan=False, indent=2) + "\n", encoding="utf-8")
    replace_snapshot(temporary, path)


def write_jpeg(path, image):
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                              [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise RuntimeError(f"could not encode frame: {path.name}")
    temporary = path.with_name(path.stem + ".tmp.jpg")
    temporary.write_bytes(encoded.tobytes())
    replace_snapshot(temporary, path)


def comparison_frame(clean, attacked, defended, step):
    height, width = clean.shape[:2]
    frame = np.full((height + 32, width * 3, 3), (18, 23, 29), np.uint8)
    for index, (image, label) in enumerate(((clean, "CLEAN"), (attacked, "ATTACKED"), (defended, "DEFENDED"))):
        frame[32:, index * width:(index + 1) * width] = image
        cv2.putText(frame, f"{label} | t={step}", (index * width + 10, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (228, 236, 240), 1, cv2.LINE_AA)
    return frame


def run_experiment(config, output_dir):
    """Hold a loaded environment until a user closes the session.

    Session control only changes state between physics steps. Loading does not
    advance physics; pause and single-step acknowledgements are published after
    the requested state/frame is available. Videos are optional archival output.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "summary.json").exists():
        raise FileExistsError("Refusing to overwrite a completed run")
    started = perf_counter()
    created_at = datetime.now().astimezone().isoformat()
    raw_env = None
    step = episode_step = control_seq = 0
    physics_seconds = episode_physics_seconds = 0.0
    episode = 1
    fps = config.control_freq
    scene = camera = kitchen = None
    physics = PhysicsParameters()
    episode_done = False
    state = "loading"
    last_frame_at = None
    rows, events = [], []

    def progress(status, phase, **extra):
        write_json(output_dir / "progress.json", {
            "status": status, "phase": phase, "step": step, "total": config.steps,
            "run_mode": config.run_mode, "episode": episode, "episode_step": episode_step,
            "episode_done": episode_done, "simulation_seconds": round(physics_seconds, 9),
            "episode_simulation_seconds": round(episode_physics_seconds, 9),
            "control_freq": fps, "control_seq": control_seq, "frame_index": len(rows),
            "last_frame_at": last_frame_at, "event_count": len(events),
            "scene": scene, "camera": camera.describe() if camera else None,
            "contact_count": int(kitchen.sim.data.ncon) if kitchen else 0,
            "elapsed_seconds": round(perf_counter() - started, 2),
            "updated_at": datetime.now().astimezone().isoformat(),
            "latest": rows[-1] if rows else None, **extra,
        })

    def cancelled():
        return (output_dir / "cancel.request").exists()

    try:
        progress("loading", "正在载入仿真依赖")
        import gymnasium as gym
        import robocasa  # noqa: F401 -- registers environments
        from ..attacks.digital_patch import make_noise_patch, image_delta_metrics
        from ..attacks.observation import DigitalPatchWrapper
        from ..defenses.filters import DefenseWrapper, ImageDefense
        from ..policies.random_policy import SeededRandomPolicy, HoldPolicy
        from ..safety_monitor.runtime import ActionGuard, TemporalImageMonitor

        random.seed(config.seed)
        np.random.seed(config.seed)
        write_json(output_dir / "config.json", config.to_dict())
        if cancelled():
            progress("cancelled", "已停止（尚未创建场景）")
            return
        progress("loading", "正在创建厨房场景，首次载入约需 30–60 秒")
        print("Creating resident RoboCasa kitchen ...", flush=True)
        raw_env = gym.make(config.env_id, **config.environment_kwargs())
        if cancelled():
            progress("cancelled", "已停止（场景载入结束）")
            return
        patch = make_noise_patch(config.patch_size, seed=config.patch_seed)
        attack = DigitalPatchWrapper(
            raw_env, patch, camera_key=config.camera_key, location=config.location,
            offset=config.offset, alpha=config.alpha if config.attack_mode == "noise_patch" else 0.0,
        )
        defense = ImageDefense(config.defense, config.median_kernel)
        env = DefenseWrapper(attack, defense, config.camera_key)
        monitor = TemporalImageMonitor(config.temporal_threshold)
        guard = ActionGuard(env.action_space)
        policy = (SeededRandomPolicy(env.action_space, config.seed, config.action_scale)
                  if config.policy == "random" else HoldPolicy(env.action_space))
        progress("loading", "正在重置场景与渲染相机")
        observation, info = env.reset(seed=config.seed)
        kitchen = raw_env.unwrapped.env
        physics.apply(kitchen, config)
        camera = OrbitCamera(kitchen)
        scene = describe_scene(kitchen, config, observation)
        fps = int(kitchen.control_freq)
        placement = attack.placement
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        imageio.imwrite(output_dir / "patch.png", patch)
        mask = np.zeros(attack.last_clean_image.shape[:2], np.uint8)
        if config.attack_mode != "none" and config.alpha > 0:
            mask[placement.top:placement.bottom, placement.left:placement.right] = 255
        imageio.imwrite(output_dir / "patch_mask.png", mask)
        schema = {
            key: ({"shape": list(value.shape), "dtype": str(value.dtype)} if isinstance(value, np.ndarray)
                  else {"type": type(value).__name__, "value": value})
            for key, value in observation.items()
        }
        write_json(output_dir / "observation_schema.json", schema)

        def save_episode():
            write_json(output_dir / "scene.json", scene)
            prefix = "initial" if episode == 1 else f"episode_{episode:03d}"
            (output_dir / f"{prefix}_scene.xml").write_text(kitchen.sim.model.get_xml(), encoding="utf-8")
            np.savez_compressed(output_dir / f"{prefix}_state.npz", state=kitchen.sim.get_state().flatten())

        save_episode()
        stop_reason = "user_closed"
        last_images = {}
        last_preview = 0.0
        with ExitStack() as stack:
            writers = {}
            if config.record_video:
                for name in ("clean", "attacked", "defended", "comparison"):
                    writer = imageio.get_writer(output_dir / f"{name}.mp4", fps=fps,
                                                codec="libx264", quality=8, macro_block_size=1)
                    stack.callback(writer.close)
                    writers[name] = writer
            metrics_file = stack.enter_context((output_dir / "metrics.jsonl").open("w", encoding="utf-8"))
            actions_file = stack.enter_context((output_dir / "actions.jsonl").open("w", encoding="utf-8"))
            event_file = stack.enter_context((output_dir / "events.jsonl").open("w", encoding="utf-8"))

            def emit(kind, message):
                event = {"step": step, "episode": episode, "kind": kind, "message": message,
                         "wall_time": datetime.now().astimezone().isoformat()}
                events.append(event)
                event_file.write(json.dumps(event, ensure_ascii=False) + "\n")
                event_file.flush()

            def render_live(current):
                nonlocal last_frame_at, last_preview
                # This is the actual MuJoCo center camera, not the old decorative
                # Three.js model. This extra render never changes policy inputs.
                scene = kitchen.sim.render(height=480, width=768,
                                           camera_name="robot0_agentview_center")[::-1].copy()
                previews = {**last_images, "scene": scene, "orbit": camera.render(kitchen),
                            "left": current["video.robot0_agentview_left"],
                            "right": current["video.robot0_agentview_right"],
                            "wrist": current["video.robot0_eye_in_hand"]}
                # Camera selector shows clean sensor views even if that camera
                # was selected as the attack surface for this session.
                clean_camera = config.camera_key.rsplit("_", 1)[-1]
                alias = {"left": "left", "right": "right", "hand": "wrist"}[clean_camera]
                previews[alias] = attack.last_clean_image
                for name, image in previews.items():
                    filename = "preview.jpg" if name == "comparison" else f"{name}.jpg"
                    write_jpeg(output_dir / filename, image)
                last_frame_at = time()
                last_preview = perf_counter()

            def record(current, reward, terminated, truncated, current_info, force_preview=False):
                clean = attack.last_clean_image
                attacked = env.last_input_image
                defended = current[config.camera_key]
                # Only this evaluator gets a clean reference, never the defense,
                # runtime monitor or policy. Also measure clean-filter distortion.
                clean_defended = defense(clean)
                attack_metrics = image_delta_metrics(clean, attacked)
                defense_metrics = image_delta_metrics(clean, defended)
                clean_metrics = image_delta_metrics(clean, clean_defended)
                temporal = monitor.update(defended)
                row = {
                    "frame_index": len(rows), "step": step, "episode": episode,
                    "episode_step": episode_step, "simulation_time_seconds": round(physics_seconds, 9),
                    "reward": float(reward), "terminated": bool(terminated), "truncated": bool(truncated),
                    "success": bool(current_info.get("success", False)), **attack_metrics,
                    "defended_mean_abs_delta": defense_metrics["mean_abs_delta"],
                    "clean_filter_mean_abs_delta": clean_metrics["mean_abs_delta"],
                    "defense_latency_ms": env.last_latency_ms, **temporal,
                }
                rows.append(row)
                metrics_file.write(json.dumps(row, allow_nan=False) + "\n")
                metrics_file.flush()
                if temporal["temporal_alert"]:
                    emit("observation_change", "相邻帧变化超过阈值；运动和光照也可能触发，未判定攻击或物理伤害。")
                last_images.clear()
                last_images.update(clean=clean, attacked=attacked, defended=defended,
                                   comparison=comparison_frame(clean, attacked, defended, episode_step))
                for name, image in last_images.items():
                    if name in writers:
                        writers[name].append_data(image)
                    if episode_step % config.save_every == 0 or force_preview:
                        imageio.imwrite(frames_dir / f"e{episode:03d}_{episode_step:06d}_{name}.png", image)
                if force_preview or perf_counter() - last_preview >= 0.1:
                    render_live(current)
                progress(state, "仿真运行中" if state == "running" else "环境已就绪，等待操作")

            state = "ready" if config.run_mode == "session" else "running"
            record(observation, 0.0, False, False, info, force_preview=True)
            emit("ready", "真实厨房环境已加载并保持在内存中；实时相机已就绪。")
            progress(state, "环境已就绪，点击运行或单步" if state == "ready" else "定步实验运行中")
            print(f"Environment ready: mode={config.run_mode}, fps={fps}, recording={config.record_video}", flush=True)
            heartbeat = perf_counter()
            while True:
                camera_request = read_json(output_dir / 'camera.request.json', {})
                if camera_request.get('seq', 0) > camera.seq:
                    camera.apply(camera_request['parameters'], camera_request['seq'])
                    render_live(observation)
                    progress(state, '观察视角已更新，仿真步数不变')
                if cancelled():
                    stop_reason = "cancelled"
                    emit("cancelled", "用户结束会话；已保存快照、指标和已录制的帧。")
                    break
                if config.run_mode == "batch" and step >= config.steps:
                    stop_reason = "step_limit"
                    break
                single_step = False
                pending_seq = None
                if config.run_mode == "session":
                    request = read_json(output_dir / "control.request.json", {})
                    if request.get("seq", 0) > control_seq:
                        pending_seq = request["seq"]
                        command = request.get("command")
                        if command == "resume":
                            state = "running"
                            control_seq = pending_seq
                            emit("resumed", "持续运行已开始；只有暂停或结束会话才停止推进。")
                        elif command == "pause":
                            state = "paused"
                            control_seq = pending_seq
                            render_live(observation)
                            emit("paused", "已暂停物理步进，环境仍保持加载。")
                        elif command == "step":
                            single_step = True
                        elif command == "reset":
                            progress("loading", "正在重置当前厨房场景")
                            observation, info = env.reset(seed=config.seed)
                            kitchen = raw_env.unwrapped.env
                            physics.apply(kitchen, config)
                            camera_seq = camera.seq
                            camera = OrbitCamera(kitchen)
                            camera.seq = camera_seq
                            scene = describe_scene(kitchen, config, observation)
                            monitor = TemporalImageMonitor(config.temporal_threshold)
                            episode += 1
                            episode_step = 0
                            episode_physics_seconds = 0.0
                            episode_done = False
                            save_episode()
                            state = "paused"
                            record(observation, 0.0, False, False, info, force_preview=True)
                            control_seq = pending_seq
                            emit("reset", "环境已重置并暂停，可继续运行或单步。")
                        else:
                            raise ValueError(f"unknown worker command: {command}")
                        if not single_step:
                            progress(state, "仿真运行中" if state == "running" else "环境已暂停，等待操作")
                    if state != "running" and not single_step:
                        if perf_counter() - heartbeat > 1:
                            progress(state, "环境已就绪，点击运行或单步" if state == "ready" else "环境已暂停，等待操作")
                            heartbeat = perf_counter()
                        sleep(0.05)
                        continue
                action = policy(observation)
                valid, reason = guard.check(action)
                if not valid:
                    stop_reason = "action_guard"
                    emit("action_rejected", f"动作未发送至仿真：{reason}")
                    break
                actions_file.write(json.dumps({"step": step + 1, "episode": episode, "action": action}, default=json_default) + "\n")
                actions_file.flush()
                time_before_step = float(kitchen.sim.data.time)
                observation, reward, terminated, truncated, info = env.step(action)
                actual_delta = float(kitchen.sim.data.time) - time_before_step
                physics_seconds += actual_delta
                episode_physics_seconds += actual_delta
                step += 1
                episode_step += 1
                episode_done = bool(terminated or truncated)
                if single_step or (episode_done and config.run_mode == "session"):
                    state = "paused"
                record(observation, reward, terminated, truncated, info, force_preview=single_step or episode_done)
                if single_step:
                    control_seq = pending_seq
                    emit("single_step", "已推进一个控制步，仿真继续保持暂停。")
                progress(state, "本轮已结束，请重置环境" if episode_done else "仿真运行中" if state == "running" else "环境已暂停，等待操作")
                if step % 25 == 0:
                    print(f"step={step}, episode={episode}, episode_step={episode_step}", flush=True)
                if episode_done and config.run_mode == "batch":
                    stop_reason = "terminated" if terminated else "truncated"
                    break
            render_live(observation)
            for name, image in last_images.items():
                imageio.imwrite(frames_dir / f"e{episode:03d}_{episode_step:06d}_{name}.png", image)

        def mean(key):
            return float(np.mean([row[key] for row in rows]))

        status = "cancelled" if stop_reason == "cancelled" else "failed" if stop_reason == "action_guard" else "completed"
        source_paths = (
            Path(__file__), Path(__file__).with_name("config.py"),
            Path(__file__).with_name("scene_runtime.py"), Path(__file__).with_name("catalog.py"),
            EXPERIMENTS_DIR / "attacks/digital_patch.py", EXPERIMENTS_DIR / "attacks/observation.py",
            EXPERIMENTS_DIR / "defenses/filters.py", EXPERIMENTS_DIR / "safety_monitor/runtime.py",
            EXPERIMENTS_DIR / "policies/random_policy.py",
        )
        summary = {
            "schema_version": 4, "scene": scene, "experiment": "resident_simulation" if config.run_mode == "session" else "integrated_observation_pipeline",
            "created_at": created_at, "status": status, "config": config.to_dict(),
            "env_id": config.env_id, "split": config.split, "seed": config.seed,
            "camera_key": config.camera_key, "image_shape": list(attack.last_clean_image.shape),
            "task_description": observation.get("annotation.human.task_description", ""),
            "policy": policy.name, "policy_uses_vision": policy.uses_vision,
            "interpretation": "真实仿真环境的观测注入与滤波基线。当前随机/保持策略均不使用视觉；像素误差和变化提醒不代表攻击成功率、防御有效性或物理安全。",
            "run_mode": config.run_mode, "record_video": config.record_video,
            "defense": config.defense, "monitor": "temporal_image_change_and_action_validation",
            "hardware_connected": False, "stop_reason": stop_reason,
            "steps_requested": config.steps if config.run_mode == "batch" else None,
            "steps_completed": step, "episodes": episode, "episode_step": episode_step,
            "frames_observed": len(rows), "frames_written": len(rows) if config.record_video else 0,
            "fps": fps, "simulation_seconds": round(physics_seconds, 9),
            "video_duration_seconds": len(rows) / fps if config.record_video else None,
            "elapsed_seconds": round(perf_counter() - started, 3),
            "success_observed": any(row["success"] for row in rows),
            "mean_abs_delta": mean("mean_abs_delta"),
            "mean_changed_fraction": mean("changed_fraction"),
            "mean_defended_abs_delta": mean("defended_mean_abs_delta"),
            "mean_clean_filter_abs_delta": mean("clean_filter_mean_abs_delta"),
            "mean_defense_latency_ms": mean("defense_latency_ms"),
            "temporal_alert_count": sum(row["temporal_alert"] for row in rows),
            "action_rejection_count": sum(event["kind"] == "action_rejected" for event in events),
            "patch": {"kind": config.attack_mode, "optimized": False, "placement": placement.to_dict(),
                      "alpha": config.alpha if config.attack_mode != "none" else 0,
                      "area_fraction": float(np.count_nonzero(mask) / mask.size),
                      "sha256_rgb": hashlib.sha256(patch.tobytes()).hexdigest()},
            "episode_metadata": kitchen.get_ep_meta(),
            "versions": {name: version(name) for name in ("robocasa", "robosuite", "mujoco", "gymnasium", "numpy", "imageio", "flask")},
            "source_sha256": {str(path.relative_to(EXPERIMENTS_DIR)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        }
        write_json(output_dir / "summary.json", summary)
        report = f"""# 具身智能安全集成实验报告

## 目录

- [会话与结果](#results)
- [时间与录像](#time)
- [解释范围](#limits)

## <a id="results"></a>会话与结果

- 实验：{output_dir.name}
- 时间：{created_at}
- 模式：{config.run_mode}
- 任务：{config.env_id}
- 厨房布局 / 风格：{scene["layout_id"]} / {scene["style_id"]}
- 重力：{config.gravity_z} m/s²；物体质量倍率：{config.object_mass_scale}；滑动摩擦倍率：{config.object_friction_scale}
- 策略：{config.policy}；动作幅度：{config.action_scale}
- 相机：{config.camera_key}
- 攻击：{config.attack_mode}，补丁 {config.patch_size}×{config.patch_size}，alpha={config.alpha}
- 防御：{config.defense}，kernel={config.median_kernel}
- 累计步数：{step}；当前场景步数：{episode_step}；场景轮数：{episode}；结束原因：{stop_reason}
- 攻击后相对原图 MAE：{summary['mean_abs_delta']:.4f}
- 防御后相对原图 MAE：{summary['mean_defended_abs_delta']:.4f}
- 干净图像经过滤波的 MAE：{summary['mean_clean_filter_abs_delta']:.4f}
- 防御处理平均耗时：{summary['mean_defense_latency_ms']:.3f} ms
- 观测变化提醒：{summary['temporal_alert_count']}；非法动作拦截：{summary['action_rejection_count']}

## <a id="time"></a>时间与录像

仿真控制频率 {fps} Hz。累计实际物理模拟时间 {physics_seconds:.3f} 秒，墙钟耗时 {summary['elapsed_seconds']:.2f} 秒（包含载入、暂停与渲染）。
按控制频率折算，100 步约 {100 / fps:.2f} 秒，101 帧录像约 {101 / fps:.2f} 秒。物理时间按 MuJoCo 实际步长累加；频率不能整除物理步长时，实际值可能略小于折算值。旧版默认 20 Hz 下 100 步对应 5 秒。
本次视频录制：{'开启' if config.record_video else '关闭'}；观测帧数 {len(rows)}，录制帧数 {summary['frames_written']}。
会话模式默认不限制运行步数，加载后先待命；暂停不卸载环境。录像只保存物理步进与重置帧，不复制暂停期间的重复帧。

## <a id="limits"></a>解释范围

{summary['interpretation']}

滤波只接收受扰相机图像，没有获得干净参考或补丁掩码。干净图像只用于旁路评测。
中央实时场景来自 MuJoCo 的实际相机。当前未接入 VLA、YOLO 或 ESP32。
summary.json、metrics.jsonl、actions.jsonl、相机快照，以及启用录制时的视频保存在同目录。
"""
        (output_dir / "report.md").write_text(report, encoding="utf-8")
        progress(status, "定步实验完成" if status == "completed" else "会话已结束，仿真环境已释放" if status == "cancelled" else "动作检查阻止继续执行")
        print(f"Finished: {status}. Output: {output_dir}", flush=True)
    except Exception as error:
        (output_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        progress("failed", "运行失败，查看日志了解原因", error=str(error))
        raise
    finally:
        if raw_env is not None:
            raw_env.close()
