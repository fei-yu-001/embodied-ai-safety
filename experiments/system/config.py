"""Validated configuration shared by the CLI, worker and local web API."""

from dataclasses import asdict, dataclass, fields
import math

from .catalog import task_ids, layout_ids, style_ids, task_info


CAMERAS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)
LOCATIONS = ("center", "top_left", "top_right", "bottom_left", "bottom_right")
ENV_ID = "robocasa/PickPlaceCounterToCabinet"


@dataclass(frozen=True)
class ExperimentConfig:
    env_id: str = ENV_ID
    split: str = "pretrain"
    scene_mode: str = "sample"
    layout_id: int | None = None
    style_id: int | None = None
    control_freq: int = 20
    gravity_z: float = -9.81
    object_mass_scale: float = 1.0
    object_friction_scale: float = 1.0
    action_scale: float = 1.0
    steps: int = 1200
    run_mode: str = "session"
    record_video: bool = False
    seed: int = 0
    patch_seed: int = 1
    camera_key: str = CAMERAS[0]
    attack_mode: str = "noise_patch"
    patch_size: int = 50
    location: str = "center"
    offset: tuple[int, int] = (0, 0)
    alpha: float = 1.0
    defense: str = "median"
    median_kernel: int = 5
    temporal_threshold: float = 12.0
    save_every: int = 25
    policy: str = "random"

    @classmethod
    def from_dict(cls, payload: dict):
        if not isinstance(payload, dict):
            raise ValueError("配置必须是 JSON 对象")
        unknown = set(payload) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError("未知配置项：" + ", ".join(sorted(map(str, unknown))))
        data = asdict(cls()) | payload
        for key, low, high in (
            ("steps", 1, 2000), ("control_freq", 5, 50), ("seed", 0, 2**32 - 1),
            ("patch_seed", 0, 2**32 - 1), ("patch_size", 1, 256),
            ("save_every", 1, 2000),
        ):
            value = data[key]
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{key} 必须是 {low}–{high} 之间的整数")
        choices = {
            "env_id": tuple(task_ids()), "split": ("pretrain", "target", "all"),
            "camera_key": CAMERAS, "attack_mode": ("none", "noise_patch"),
            "location": LOCATIONS, "defense": ("none", "median"),
            "median_kernel": (3, 5, 7), "policy": ("random", "hold"),
            "scene_mode": ("sample", "custom"),
            "run_mode": ("session", "batch"),
        }
        for key, allowed in choices.items():
            if type(data[key]) is not type(allowed[0]) or data[key] not in allowed:
                raise ValueError(f"{key} 可选值：{', '.join(map(str, allowed))}")
        for key, low, high in (("alpha", 0, 1), ("temporal_threshold", 0, 255),
                               ("gravity_z", -20, 0), ("object_mass_scale", .25, 4),
                               ("object_friction_scale", .1, 3), ("action_scale", 0, 1)):
            value = data[key]
            if type(value) not in (int, float) or not low <= value <= high or not math.isfinite(value):
                raise ValueError(f"{key} 必须是 {low}–{high} 之间的有限数值")
        for key, available in (("layout_id", layout_ids()), ("style_id", style_ids())):
            value = data[key]
            if data["scene_mode"] == "sample":
                if value is not None:
                    raise ValueError("随机场景模式下 layout_id/style_id 应为 null")
            elif type(value) is not int or value not in available:
                raise ValueError(f"{key} 必须选择本地目录中已安装的编号")
        selected_task = task_info(data['env_id'])
        if data['scene_mode'] == 'custom' and (data['layout_id'] in selected_task['excluded_layouts'] or data['style_id'] in selected_task['excluded_styles']):
            raise ValueError('当前任务不兼容选定布局或风格，请选择其他编号或随机场景')
        if type(data["record_video"]) is not bool:
            raise ValueError("record_video 必须是布尔值")
        offset = data["offset"]
        if not isinstance(offset, (list, tuple)) or len(offset) != 2 or any(type(v) is not int for v in offset):
            raise ValueError("offset 必须是两个整数 [dy, dx]")
        data["offset"] = tuple(offset)
        # The supported RoboCasa camera observations are fixed at 256x256.
        edge = 256 - data["patch_size"]
        top, left = {
            "center": (edge // 2, edge // 2), "top_left": (0, 0),
            "top_right": (0, edge), "bottom_left": (edge, 0),
            "bottom_right": (edge, edge),
        }[data["location"]]
        if not (0 <= top + offset[0] <= edge and 0 <= left + offset[1] <= edge):
            raise ValueError("补丁位置超出 256×256 相机画面，请调整尺寸或偏移")
        return cls(**data)

    def environment_kwargs(self):
        options = {"seed": self.seed, "control_freq": self.control_freq}
        if self.scene_mode == "custom":
            # RoboCasa split overrides explicit IDs; custom scenes must bypass it.
            options.update(split=None, layout_ids=[self.layout_id], style_ids=[self.style_id],
                           obj_instance_split=None if self.split == "all" else self.split)
        else:
            options["split"] = self.split
        return options

    def to_dict(self):
        return asdict(self)
