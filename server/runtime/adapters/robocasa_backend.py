"""Load and step a real RoboCasa kitchen, and publish one camera frame.

Scope for M2 slice 1: load, single step with the declared policy, and a JPEG
frame per step. No jogging, no editing, no policy learning. Anything that is
not measured is reported as not measured.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys
import time

from server.infrastructure.frame_store import FrameStore

DEFAULT_TASK = "robocasa/PickPlaceCounterToCabinet"
TASK_PATTERN = re.compile(r"^robocasa/[A-Za-z0-9_]{1,80}$")
ALLOWED_KEYS = {
    "env_id",
    "seed",
    "split",
    "control_freq",
    "scene_mode",
    "layout_id",
    "style_id",
    "policy",
    "camera_key",
    "vision_target",
    "vision_gain",
    "tool_safety",
}
SPLITS = {"pretrain", "target", "all"}
POLICIES = {"random", "hold", "vision_centroid"}
VISION_TARGETS = {"green", "red", "blue"}
CAMERA_KEYS = {
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
}

# Verified against default_pandaomron.json: OSC_POSE, input delta in [-1, 1],
# output +/-0.05 m (base frame) and +/-0.5 rad. Gripper: -1 open, +1 close.
MAX_STEP_POSITION_M = 0.05
MAX_STEP_ROTATION_RAD = 0.5
JOG_KEYS = {"delta_m", "delta_rot", "gripper", "repeat"}
# The gripper is binary in the wrapper: input < 0.5 opens, otherwise closes.
GRIPPER_INPUT = {"open": -1.0, "close": 1.0}
GRIPPER_CHOICES = {"open", "hold", "close"}
MAX_REPEAT = 10
EDIT_KINDS = {"position", "mass"}
MASS_RANGE_KG = (0.001, 50.0)
EDIT_RADIUS_M = 3.0
EDIT_HEIGHT_M = (0.0, 3.0)
SCENE_KIND = "scene"
ACTION_POSITION = "action.end_effector_position"
ACTION_ROTATION = "action.end_effector_rotation"
ACTION_GRIPPER = "action.gripper_close"


class ConfigError(Exception):
    pass


class JogError(Exception):
    """A jog request that must be refused, never silently clipped."""


def _vector(args, name, limit):
    value = args.get(name, [0.0, 0.0, 0.0])
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise JogError(f"{name} 必须是三个数值")
    result = []
    for item in value:
        if type(item) not in (int, float) or item != item or abs(item) == float("inf"):
            raise JogError(f"{name} 必须是有限数值")
        if abs(item) > limit:
            raise JogError(f"{name} 单步不得超过 {limit}")
        result.append(float(item))
    return result


def jog_action(args, space_keys):
    """Translate a jog request into a real action, or refuse it."""
    if not isinstance(args, dict):
        raise JogError("jog 参数必须是对象")
    unknown = set(args) - JOG_KEYS
    if unknown:
        raise JogError(f"未知 jog 参数：{','.join(sorted(unknown))}")
    delta_m = _vector(args, "delta_m", MAX_STEP_POSITION_M)
    delta_rot = _vector(args, "delta_rot", MAX_STEP_ROTATION_RAD)
    gripper = args.get("gripper", "hold")
    repeat = args.get("repeat", 1)
    if type(repeat) is not int or not 1 <= repeat <= MAX_REPEAT:
        raise JogError(f"repeat 必须是 1 到 {MAX_REPEAT} 的整数")
    if gripper not in GRIPPER_CHOICES:
        raise JogError("gripper 必须是 open / hold / close")
    if gripper not in GRIPPER_INPUT:
        raise JogError("hold 必须由后端先解析成上一条夹爪指令")
    for name in (ACTION_POSITION, ACTION_ROTATION, ACTION_GRIPPER):
        if name not in space_keys:
            raise JogError(f"当前机器人不支持 {name}")
    position = [axis / MAX_STEP_POSITION_M for axis in delta_m]
    rotation = [axis / MAX_STEP_ROTATION_RAD for axis in delta_rot]
    return {
        ACTION_POSITION: position,
        ACTION_ROTATION: rotation,
        ACTION_GRIPPER: [GRIPPER_INPUT[gripper]],
        "jog": {"delta_m": delta_m, "delta_rot": delta_rot, "gripper": gripper, "repeat": repeat},
    }


def normalize_config(raw):
    """Validate a session config. Unknown keys fail closed."""
    raw = {} if raw is None else raw
    if not isinstance(raw, dict):
        raise ConfigError("会话配置必须是对象")
    unknown = set(raw) - ALLOWED_KEYS
    if unknown:
        raise ConfigError(f"未知配置项：{','.join(sorted(unknown))}")
    env_id = raw.get("env_id", DEFAULT_TASK)
    if not isinstance(env_id, str) or not TASK_PATTERN.fullmatch(env_id):
        raise ConfigError("env_id 必须形如 robocasa/<任务名>")
    seed = raw.get("seed", 0)
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ConfigError("seed 必须是 0 到 2^32-1 的整数")
    control_freq = raw.get("control_freq", 20)
    if type(control_freq) is not int or not 5 <= control_freq <= 50:
        raise ConfigError("control_freq 必须是 5 到 50 的整数")
    split = raw.get("split", "pretrain")
    if split not in SPLITS:
        raise ConfigError("split 必须是 pretrain / target / all")
    scene_mode = raw.get("scene_mode", "sample")
    if scene_mode not in {"sample", "custom"}:
        raise ConfigError("scene_mode 必须是 sample / custom")
    layout_id = raw.get("layout_id")
    style_id = raw.get("style_id")
    if scene_mode == "custom":
        for name, value in (("layout_id", layout_id), ("style_id", style_id)):
            if type(value) is not int or value < 0:
                raise ConfigError(f"{name} 在 custom 模式下必须是整数")
    elif layout_id is not None or style_id is not None:
        raise ConfigError("sample 模式下 layout_id / style_id 必须为空")
    policy = raw.get("policy", "random")
    if policy not in POLICIES:
        raise ConfigError("policy 必须是 " + " / ".join(sorted(POLICIES)))
    vision_target = raw.get("vision_target", "green")
    if vision_target not in VISION_TARGETS:
        raise ConfigError("vision_target 必须是 green / red / blue")
    vision_gain = raw.get("vision_gain", 0.6)
    if type(vision_gain) not in (int, float) or not 0.05 <= float(vision_gain) <= 2.0:
        raise ConfigError("vision_gain 必须在 0.05 到 2.0 之间")
    camera_key = raw.get("camera_key", "video.robot0_agentview_left")
    if camera_key not in CAMERA_KEYS:
        raise ConfigError("camera_key 不在允许的相机列表内")
    from server.domain.tool_safety import ToolSafetySpecError, normalize_tool_safety_spec

    try:
        # 默认关闭；配置错误在这里 fail closed，不把坏规则带进仿真
        tool_safety = normalize_tool_safety_spec(raw.get("tool_safety"))
    except ToolSafetySpecError as error:
        raise ConfigError(f"tool_safety 配置无效：{error}") from None
    return {
        "env_id": env_id,
        "seed": seed,
        "control_freq": control_freq,
        "split": split,
        "scene_mode": scene_mode,
        "layout_id": layout_id,
        "style_id": style_id,
        "policy": policy,
        "camera_key": camera_key,
        "vision_target": vision_target,
        "vision_gain": float(vision_gain),
        "tool_safety": tool_safety,
    }


def environment_kwargs(config):
    """RoboCasa kwargs. Custom scenes must bypass `split`, which overrides IDs."""
    options = {"seed": config["seed"], "control_freq": config["control_freq"],
               "camera_widths": 768, "camera_heights": 480}
    if config["scene_mode"] == "custom":
        options.update(
            split=None,
            layout_ids=[config["layout_id"]],
            style_ids=[config["style_id"]],
            obj_instance_split=None if config["split"] == "all" else config["split"],
        )
    else:
        options["split"] = config["split"]
    return options


class RoboCasaBackend:
    name = "robocasa"
    loads_physics = True

    def __init__(self, frame_dir, env_factory=None, policy_factory=None, guard_factory=None):
        self.frame_dir = Path(frame_dir)
        # 帧存储：编号不可变帧 + 无竞争发布，latest.jpg 仅作兼容副本
        self.frame_store = FrameStore(self.frame_dir)
        self.env_factory = env_factory
        self.policy_factory = policy_factory
        self.guard_factory = guard_factory
        self.env = None
        self.policy = None
        self.guard = None
        self.gripper = "open"
        self.manual_start_eef = None
        self.geometry = None
        self.scene_revision = 0
        self.base_position = None
        self.config = None
        # 工具安全：默认关闭的 observer，只在启用时构造快照，不拖慢普通场景
        self.tool_safety = None
        self.tool_safety_payload = None
        self.tool_safety_error = None
        self._tool_body_index = None
        self._tool_body_index_key = None
        self.session_id = None
        self.observation = None
        self.step = 0
        self.episode = 1
        self.frame_seq = 0
        # 帧发布状态：失败与成功分开记录，API 不能仅凭 worker alive 把旧帧称实时
        self.published_frame_seq = 0
        self.last_frame_error = None
        self.frame_warning = None
        self.sim_seconds = 0.0
        self.last_error = None

    def close(self):
        """Release renderer/physics resources; safe after partial load or repeated close."""
        env, self.env = self.env, None
        if env is not None:
            env.close()

    def load(self, session_id, config):
        self.config = normalize_config(config)
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        env = self._make_env(self.config)
        self.env = env
        observation, info = env.reset(seed=self.config["seed"])
        self.observation = observation
        self.policy = self._make_policy(env)
        self.guard = self._make_guard(env)
        self.session_id = session_id
        self.step = 0
        self.episode = 1
        self.sim_seconds = 0.0
        self.last_error = None
        self.gripper = "open"
        self.frame_seq = 0
        self.published_frame_seq = 0
        self.last_frame_error = None
        self.frame_warning = None
        self._prepare_visual_assets()
        self._tool_safety_bind()
        frame_seq = self._write_frame(observation, force=True)
        self._tool_safety_tick()
        return {
            "env_id": self.config["env_id"],
            "control_freq": int(getattr(env.unwrapped, "control_freq", self.config["control_freq"]))
            if hasattr(env, "unwrapped")
            else self.config["control_freq"],
            "policy": getattr(self.policy, "name", self.config["policy"]),
            "policy_uses_vision": bool(getattr(self.policy, "uses_vision", False)),
            "policy_manifest": self._policy_manifest(),
            "task_description": self._task_description(observation, info),
            "episode": self.episode,
            "step": self.step,
            "sim_seconds": self.sim_seconds,
            "frame_seq": frame_seq,
            "capabilities": self._capabilities(),
            "tool_safety": self._tool_safety_summary(),
        }

    def apply(self, command, args=None):
        if self.env is None:
            raise ConfigError("环境尚未加载")
        if command == "reset":
            return self._reset()
        if command == "step":
            return self._step()
        if command == "jog":
            return self._jog(args or {})
        if command == "edit":
            return self._edit(args or {})
        if command in {"resume", "pause"}:
            return {"state": "RUNNING" if command == "resume" else "PAUSED", **self._summary()}
        if command == "stop":
            return {"state": "CLOSED", **self._summary()}
        raise ConfigError("未知命令")

    def _reset(self):
        observation, info = self.env.reset(seed=self.config["seed"])
        self.observation = observation
        self.episode += 1
        self.step = 0
        self.sim_seconds = 0.0
        self.last_error = None
        self.gripper = "open"
        # 帧文件在会话内不可变；重置 episode 也必须保持编号单调递增。
        self.last_frame_error = None
        self.frame_warning = None
        frame_seq = self._write_frame(observation, force=True)
        self._prepare_visual_assets()
        self._tool_safety_tick()
        return {"state": "PAUSED", "frame_seq": frame_seq, **self._summary(), "task_description": self._task_description(observation, info)}

    def _step(self):
        return self._advance(self.policy(self.observation))

    def _jog(self, args):
        if not isinstance(args, dict):
            return self._reject_jog("jog 参数必须是对象")
        requested = args.get("gripper", "hold")
        if requested not in GRIPPER_CHOICES:
            return self._reject_jog("gripper 必须是 open / hold / close")
        resolved = self.gripper if requested == "hold" else requested
        try:
            action = jog_action({**args, "gripper": resolved}, set(self.env.action_space.spaces))
        except JogError as error:
            return self._reject_jog(str(error))
        start = self._eef()
        self.gripper = resolved
        repeat = action["jog"]["repeat"]
        result = None
        for _ in range(repeat):
            result = self._advance(action)
            if result.get("rejected") or result.get("terminated") or result.get("truncated"):
                break
        result["jog"] = {
            **action["jog"],
            "gripper": requested,
            "gripper_state": resolved,
            "requested_m": [axis * repeat for axis in action["jog"]["delta_m"]],
            "measured_m": self._displacement(start),
        }
        return result

    def begin_manual(self):
        self.manual_start_eef = self._eef()

    def end_manual(self):
        start = getattr(self, "manual_start_eef", None)
        self.manual_start_eef = None
        return self._displacement(start)

    def _eef(self):
        import numpy as np

        value = (self.observation or {}).get("state.end_effector_position_relative")
        return None if value is None else np.asarray(value, dtype=float).reshape(-1)

    def _displacement(self, start):
        import numpy as np

        end = self._eef()
        if start is None or end is None:
            return None
        return np.round(end - start, 6).tolist()

    def _reject_jog(self, reason):
        self.last_error = {"code": "jog_rejected", "reason": reason}
        return {"state": "PAUSED", "rejected": reason, **self._summary()}

    def _advance(self, action):
        import numpy as np

        action = {key: value for key, value in action.items() if key != "jog"}
        base = {}
        for name, space in self.env.action_space.spaces.items():
            if name in action:
                base[name] = np.asarray(action[name], dtype=space.dtype).reshape(space.shape)
            else:
                base[name] = np.zeros(space.shape, dtype=space.dtype)
        valid, reason = self.guard.check(base)
        if not valid:
            self.last_error = {"code": "action_rejected", "reason": reason}
            return {"state": "PAUSED", **self._summary(), "rejected": reason}
        before = self._sim_time()
        observation, reward, terminated, truncated, info = self.env.step(base)
        self.observation = observation
        self.step += 1
        self.sim_seconds += max(0.0, self._sim_time() - before)
        # 每次真实物理推进后评估，而不是等浏览器轮询：不漏步
        self._tool_safety_tick()
        frame_seq = self._write_frame(observation)
        return {
            "state": "PAUSED",
            "frame_seq": frame_seq,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "success": bool(info.get("success", False)),
            **self._summary(),
        }

    def scene_state(self, args=None):
        """World-frame state for the 3D view.

        Geometry is static per model and cached; transforms are sent every poll
        so a paused session costs one small pose payload. Units are metres,
        quaternions wxyz.
        """
        if self.env is None:
            raise ConfigError("环境尚未加载")
        try:
            import mujoco
            import numpy as np

            from server.runtime.adapters.scene_geometry import (
                build_geometry,
                build_transforms,
                fit_to_budget,
            )

            args = args if isinstance(args, dict) else {}
            task = self.env.unwrapped.env
            robot = task.robots[0]
            sim = task.sim
            model = sim.model
            arm = "right"
            site_id = robot.eef_site_id[arm] if isinstance(robot.eef_site_id, dict) else robot.eef_site_id
            eef = np.asarray(sim.data.site_xpos[site_id], dtype=float).reshape(3)
            eef_mat = np.asarray(sim.data.site_xmat[site_id], dtype=float).reshape(9)
            eef_quat = np.zeros(4)
            mujoco.mju_mat2Quat(eef_quat, eef_mat)
            base_id = sim.model.site_name2id(robot.robot_model.base.correct_naming("center"))
            base = np.asarray(sim.data.site_xpos[base_id], dtype=float).reshape(3)
            base_quat = np.zeros(4)
            mujoco.mju_mat2Quat(base_quat, np.asarray(sim.data.site_xmat[base_id], dtype=float).reshape(9))
            joints = np.asarray(getattr(robot, "_joint_positions", []), dtype=float).reshape(-1)

            # 静态几何按模型键缓存：轮询不重传整包
            if self.geometry is None or self.geometry.get("key") != self._geometry_key(model, task):
                pass
            geometry = self._cached_geometry(model, task, robot)
            # 客户端已持有同一份静态几何时，只回键，不回 100 KB 载荷
            have_key = args.get("have_geometry_key")
            geometry_key = geometry.get("key")
            include_geometry = not isinstance(have_key, str) or have_key != geometry_key
            self.base_position = np.round(base, 4).tolist()

            tree = []
            objects = []
            for item in geometry["objects"]:
                body_index = int(item["id"])
                joint = int(model.body_jntadr[body_index])
                movable = (
                    int(model.body_jntnum[body_index]) > 0
                    and int(model.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_FREE)
                )
                position = np.round(sim.data.body_xpos[body_index], 4).tolist()
                objects.append({"name": item["name"], "position": position})
                tree.append({
                    "name": item["name"],
                    "body": item["body"],
                    "position": position,
                    "mass": round(float(model.body_mass[body_index]), 4),
                    # 拾取只做选择；这些字段表示可编辑能力，不代表三维里能直接拖拽
                    "editable": ["mass"] + (["position"] if movable else []),
                })

            return {
                "frame": geometry["frame"],
                "available": {
                    "eef": True,
                    "base": True,
                    "joints": joints.size > 0,
                    "objects": bool(objects),
                    "geometry": bool(geometry.get("robot", {}).get("geoms")),
                },
                "eef": {
                    "position": np.round(eef, 4).tolist(),
                    "quaternion": np.round(eef_quat, 5).tolist(),
                },
                "base": {
                    "position": np.round(base, 4).tolist(),
                    "quaternion": np.round(base_quat, 5).tolist(),
                },
                "joints": np.round(joints, 4).tolist(),
                "objects": objects,
                "tree": tree,
                "geometry_key": geometry_key,
                "geometry": geometry if include_geometry else None,
                "geometry_included": include_geometry,
                "transforms": build_transforms(model, sim.data),
                "scene_revision": self.scene_revision,
                "bounds": {
                    "center": np.round(np.asarray(sim.model.stat.center, dtype=float), 3).tolist(),
                    "extent": round(float(sim.model.stat.extent), 3),
                    "source": "mj_model_stat",
                },
                "overlays": self._overlays(geometry),
                "tool_safety": self.tool_safety_payload,
                "step": self.step,
                "episode": self.episode,
                "sim_seconds": round(self.sim_seconds, 6),
                "frame_seq": self.frame_seq,
            }
        except Exception as error:  # a 3D view must not take the session down
            return {
                "frame": {"world": "mujoco_world", "units": "m", "up": "+z", "quaternion": "wxyz"},
                "available": {"eef": False, "base": False, "joints": False, "objects": False, "geometry": False},
                "error": f"{type(error).__name__}: {error}",
                "step": self.step,
                "episode": self.episode,
                "sim_seconds": round(self.sim_seconds, 6),
                "frame_seq": self.frame_seq,
            }

    def _geometry_key(self, model, task):
        from server.runtime.adapters.scene_geometry import geometry_key

        return geometry_key(model, task)

    # ------------------------------------------------------------ 工具安全
    # 运行路径：_advance 物理推进后调用 _tool_safety_tick()，不依赖浏览器轮询。
    # 配置关闭时 _tool_safety_tick 直接返回，普通场景不付出几何快照成本。

    def _tool_safety_bind(self):
        """Create or drop the observer to match the session config."""
        from server.runtime.tool_safety_observer import ToolSafetyObserver

        spec = (self.config or {}).get("tool_safety")
        if not spec or not spec.get("enabled"):
            # 没启用就不构造 observer，也不建几何快照
            self.tool_safety = None
            self.tool_safety_payload = None
            self.tool_safety_error = None
            return
        self.tool_safety = ToolSafetyObserver(spec=spec, enabled=True)
        self.tool_safety_payload = None
        self.tool_safety_error = None

    def reset_tool_safety(self):
        """Explicit evidence reset without changing the config."""
        if self.tool_safety is not None:
            self.tool_safety.reset("backend_reset")
            self._tool_safety_tick()

    def _tool_safety_scene(self):
        """Minimal snapshot for the observer: transforms + static geometry.

        Uses the cached static geometry when it exists (callable even after the
        geometry payload itself was thinned to a key for the browser), otherwise
        builds it once. No MuJoCo state is written here.
        """
        from server.runtime.adapters.scene_geometry import (
            build_geometry,
            build_transforms,
            fit_to_budget,
        )

        task = self.env.unwrapped.env
        sim = task.sim
        model = sim.model
        robot = task.robots[0]
        if self.geometry is None or self.geometry.get("key") != self._geometry_key(model, task):
            self.geometry = fit_to_budget(build_geometry(model, task, robot))
        return {
            "session_id": self.session_id,
            "episode": self.episode,
            "step": self.step,
            "sim_seconds": self.sim_seconds,
            "body_transforms": build_transforms(model, sim.data),
            "geometry": self.geometry,
            "body_index": self._tool_safety_body_index(self.geometry),
        }

    def _tool_safety_body_index(self, geometry):
        """name -> body id, cached per geometry key (does not change per step)."""
        from server.domain.tool_safety import build_body_index

        key = geometry.get("key") if isinstance(geometry, dict) else None
        if key is not None and key == self._tool_body_index_key:
            return self._tool_body_index
        self._tool_body_index = build_body_index(geometry or {})
        self._tool_body_index_key = key
        return self._tool_body_index

    def _tool_safety_tick(self):
        """Assess the current optical step. Never raises into the loop."""
        if self.tool_safety is None or self.env is None:
            return
        try:
            self.tool_safety_payload = self.tool_safety.update(self._tool_safety_scene())
            self.tool_safety_error = None
        except Exception as error:  # 监控不得杀死仿真循环
            # 清掉上一步的旧结果：陈旧 payload 会被误当成当前风险状态
            self.tool_safety_payload = None
            self.tool_safety_error = f"{type(error).__name__}: {error}"

    def _overlays(self, geometry=None):
        """Risk zones and tool world segments for the 3D overlay.

        Empty and explicit when the feature is off: the viewer must not draw a
        safety controller that is not running.
        """
        payload = self.tool_safety_payload
        if isinstance(payload, dict):
            overlays = payload.get("overlays") or {}
            return {
                "risk_zones": overlays.get("risk_zones", []),
                "tool_annotations": overlays.get("tool_annotations", []),
                "world_tool_segments": overlays.get("world_tool_segments", []),
                "risk_zones_source": "tool_safety_observer",
                "tool_annotations_source": "tool_safety_observer",
                "source": overlays.get("source", "user_defined_annotations"),
                "reason": overlays.get("reason"),
                "status": payload.get("status"),
                "note": "风险区与工具世界段来自已启用工具安全规则；缺数据时为 unknown",
            }
        if self.tool_safety_error:
            return {
                "risk_zones": [],
                "tool_annotations": [],
                "world_tool_segments": [],
                "source": "tool_safety_error",
                "reason": self.tool_safety_error,
                "note": "工具安全评估失败，未提供风险几何；不要在三维里自行推断",
            }
        return {
            "risk_zones": [],
            "tool_annotations": [],
            "world_tool_segments": [],
            "source": "tool_safety_disabled",
            "reason": None,
            "note": "工具安全未启用，未运行任何超区判定",
        }

    def _cached_geometry(self, model, task, robot):
        from server.runtime.adapters.scene_geometry import build_geometry, fit_to_budget

        key = self._geometry_key(model, task)
        if self.geometry is None or self.geometry.get("key") != key:
            self.geometry = build_geometry(model, task, robot)
        if "assets" not in self.geometry:
            from server.runtime.adapters.scene_assets import attach_visual_assets
            try:
                attach_visual_assets(self.geometry, model, self.frame_dir.parent / "visual_assets")
            except (OSError, ValueError, AttributeError, IndexError, TypeError) as error:
                self.geometry.setdefault("provenance", {})["asset_error"] = str(error)[:200]
                self.geometry["assets"] = {"version": "prism_visual_assets/1", "meshes": {}, "textures": {}, "materials": {}, "unsupported": [{"reason": str(error)[:160]}]}
            self.geometry = fit_to_budget(self.geometry)
        return self.geometry

    def _edit(self, args):
        """Paused-only scene edit. Returns a refusal instead of guessing."""
        if not isinstance(args, dict):
            return self._reject_edit("edit 参数必须是对象")
        target = args.get("target")
        kind = args.get("kind")
        values = args.get("values")
        if not isinstance(target, str) or not target:
            return self._reject_edit("edit 需要 target")
        if kind not in EDIT_KINDS:
            return self._reject_edit("kind 必须是 position / mass")
        # Validate the request before touching the model: bad input must never
        # half-apply.
        position = None
        if kind == "mass":
            if type(values) not in (int, float):
                return self._reject_edit("mass 需要单个数值")
            if not MASS_RANGE_KG[0] <= float(values) <= MASS_RANGE_KG[1]:
                return self._reject_edit(f"质量必须在 {MASS_RANGE_KG[0]}–{MASS_RANGE_KG[1]} kg 之间")
        else:
            position = self._edit_position(values)
            if isinstance(position, str):
                return self._reject_edit(position)
        try:
            import mujoco
            import numpy as np

            task = self.env.unwrapped.env
            sim = task.sim
            model = sim.model
            body_id = self._find_body(task, sim, target)
            if body_id is None:
                return self._reject_edit(f"找不到可编辑对象：{target}")
            if kind == "mass":
                model.body_mass[body_id] = float(values)
                mujoco.mj_setConst(model._model, sim.data._data)
            else:
                joint = int(model.body_jntadr[body_id])
                if int(model.body_jntnum[body_id]) < 1 or int(model.jnt_type[joint]) != int(mujoco.mjtJoint.mjJNT_FREE):
                    return self._reject_edit(f"{target} 没有自由关节，不能改位姿")
                offset = int(model.jnt_qposadr[joint])
                sim.data.qpos[offset:offset + 3] = np.asarray(position, dtype=float)
                velocity = int(model.jnt_dofadr[joint])
                sim.data.qvel[velocity:velocity + 6] = 0.0
            sim.forward()
            self.scene_revision += 1
            return {
                "state": "PAUSED",
                "edited": {"target": target, "kind": kind, "values": values},
                "scene_revision": self.scene_revision,
                **self._summary(),
            }
        except Exception as error:
            return self._reject_edit(f"编辑失败：{type(error).__name__}: {error}")

    def _find_body(self, task, sim, name):
        objects = getattr(task, "objects", None) or []
        if isinstance(objects, dict):
            objects = list(objects.values())
        for obj in objects:
            root = getattr(obj, "root_body", None)
            if root and getattr(obj, "name", root) == name:
                return int(sim.model.body_name2id(root))
        return None

    def _edit_position(self, values):
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            return "position 需要三个数值"
        import math

        for value in values:
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                return "position 必须是有限数值"
        try:
            import numpy as np

            base = np.asarray(getattr(self, "base_position", None) or [0.0, 0.0, 0.0], dtype=float)
        except Exception:
            base = None
        if base is not None:
            distance = ((values[0] - base[0]) ** 2 + (values[1] - base[1]) ** 2) ** 0.5
            if distance > EDIT_RADIUS_M:
                return f"position 超出工作区（半径 {EDIT_RADIUS_M} m）"
        if not EDIT_HEIGHT_M[0] <= float(values[2]) <= EDIT_HEIGHT_M[1]:
            return f"position 高度必须在 {EDIT_HEIGHT_M[0]}–{EDIT_HEIGHT_M[1]} m"
        return [float(value) for value in values]

    def _reject_edit(self, reason):
        self.last_error = {"code": "edit_rejected", "reason": reason}
        return {"state": "PAUSED", "rejected": reason, **self._summary()}

    def _policy_manifest(self):
        from experiments.sdk.contract import require_manifest

        try:
            return require_manifest(self.policy).to_dict()
        except Exception as error:
            return {"error": f"{type(error).__name__}: {error}"}

    def _capabilities(self):
        keys = set(self.env.action_space.spaces)
        return {
            "jog": {ACTION_POSITION, ACTION_ROTATION, ACTION_GRIPPER} <= keys,
            "gripper": ACTION_GRIPPER in keys,
            "joint_position": "action.joint_position" in keys,
            "target_pose": False,
            "scene_edit": True,
        }

    def _tool_safety_summary(self):
        """Compact status for command results; not the full assessment."""
        if self.tool_safety is None:
            return {"enabled": False, "status": "disabled", "source": "spec_disabled"}
        if self.tool_safety_error:
            return {"enabled": True, "status": "unknown", "error": self.tool_safety_error}
        payload = self.tool_safety_payload or {}
        metrics = payload.get("metrics") or {}
        return {
            "enabled": True,
            "status": payload.get("status", "unknown"),
            "episode_violations": metrics.get("episode_violations"),
            "zone_entries": metrics.get("zone_entries"),
            "min_net_clearance": metrics.get("min_net_clearance"),
        }

    def _summary(self):
        return {
            "episode": self.episode,
            "step": self.step,
            "sim_seconds": round(self.sim_seconds, 6),
            "frame_seq": self.frame_seq,
            "published_frame_seq": self.published_frame_seq,
            "frame_error": self.last_frame_error,
            "frame_warning": self.frame_warning,
            "last_error": self.last_error,
            "tool_safety": self._tool_safety_summary(),
        }

    def _sim_time(self):
        for path in (("unwrapped", "env", "sim", "data", "time"), ("unwrapped", "sim", "data", "time")):
            try:
                value = self.env
                for attribute in path:
                    value = getattr(value, attribute)
                return float(value)
            except (AttributeError, TypeError, ValueError):
                continue
        return time.monotonic() - getattr(self, "_wall_start", time.monotonic())

    def _prepare_visual_assets(self):
        # Load/reset may replace the compiled model; export before READY/PAUSED
        # is acknowledged, not in the first three-second scene query.
        try:
            task = self.env.unwrapped.env
            self._cached_geometry(task.sim.model, task, task.robots[0])
        except (AttributeError, IndexError):
            pass

    def _write_frame(self, observation, force=False):
        try:
            image = None
            if isinstance(observation, dict):
                image = observation.get(self.config["camera_key"]) if self.config else None
            if image is None:
                return self.frame_seq
            self.frame_seq += 1
            try:
                import cv2

                ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                                           [cv2.IMWRITE_JPEG_QUALITY, 88])
                if not ok:
                    raise OSError("JPEG 编码失败")
                data = encoded.tobytes()
            except (OSError, ImportError, ValueError) as error:
                self.last_frame_error = {
                    "code": "frame_encode_failed",
                    "reason": str(error),
                    "seq": self.frame_seq,
                }
                self.last_error = dict(self.last_frame_error)
                return self.frame_seq
            # 不可变版本帧：写入独立编号文件，绝不 rename 到别人正在读的目标上。
            # 读者拿着的旧帧不会阻塞新帧发布；latest.jpg 只是兼容副本。
            result = self.frame_store.publish(data, self.frame_seq)
            if result["error"]:
                # 编号帧未发布：记录失败并标注当前已发布帧与 seq，供 API 判定实时性
                self.last_frame_error = {
                    "code": "frame_write_failed",
                    "reason": result["error"],
                    "seq": self.frame_seq,
                }
                self.last_error = dict(self.last_frame_error)
                return self.frame_seq
            self.published_frame_seq = result["seq"]
            # 成功发布仅清除此前的 frame 类错误；其他 last_error（动作/编码等）不动
            if self.last_frame_error is not None:
                self.last_frame_error = None
                if (self.last_error or {}).get("code") in {"frame_write_failed", "frame_encode_failed", "frame_latest_copy_skipped", "frame_seq_conflict"}:
                    self.last_error = None
            if result["warning"]:
                # 编号帧已发布；兼容副本被占用只是 warning，不当作发布失败
                self.frame_warning = {"code": "latest_copy_skipped", "reason": result["warning"], "seq": result["seq"]}
            else:
                self.frame_warning = None
            return self.frame_seq
        finally:
            # Windows WGL contexts remain attached to a thread after rendering.
            # Commands and worker-owned motion serialize on env_lock, but may
            # use different threads. Unbind before that lock changes owner.
            glfw = sys.modules.get("glfw")
            if os.name == "nt" and glfw is not None and getattr(self.env.unwrapped, "env", None) is not None and glfw.get_current_context():
                glfw.make_context_current(None)

    def _make_env(self, config):
        if self.env_factory is not None:
            return self.env_factory(config)
        import gymnasium as gym
        import robocasa  # noqa: F401 -- registers environments

        return gym.make(config["env_id"], **environment_kwargs(config))

    def _make_policy(self, env):
        if self.policy_factory is not None:
            return self.policy_factory(env, self.config)
        from experiments.sdk.registry import build_policy

        return build_policy(self.config["policy"], env, self.config)

    def _make_guard(self, env):
        if self.guard_factory is not None:
            return self.guard_factory(env)
        from experiments.safety_monitor.runtime import ActionGuard

        return ActionGuard(env.action_space)

    def _task_description(self, observation, info):
        if isinstance(observation, dict):
            value = observation.get("annotation.human.task_description")
            if isinstance(value, str) and value:
                return value
        return ""
