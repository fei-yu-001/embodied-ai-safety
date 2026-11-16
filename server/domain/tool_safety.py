"""Kitchen tool safety slice: versioned spec and analytic clearance.

Pure logic and geometry. This module never imports MuJoCo: the caller passes a
scene_state snapshot holding world-frame body transforms and the static
geometry object dictionary the 3D view already uses.

Contract:
- the spec is versioned and disabled by default;
- protective zones are finite geometry in `mujoco_world` metres: a sphere or a
  capsule (segment plus radius). Tools are annotated as local segments or
  spheres, plus the world pose of the tool body, composed the same way MuJoCo
  composes a geom pose with its body pose;
- the distance between a tool primitive and a zone primitive comes from a
  closed-form segment-to-segment computation, not from sampling, so a segment
  that passes through the middle of a zone cannot slip between samples;
- a missing body pose, a missing tool, a tool without segments, a truncated
  check list or a non-finite number gives status "unknown" and is never
  reported as safe.

What this deliberately does not do: decide grasp strategy, simulate cutting or
material failure, estimate human harm, or guarantee real-robot behaviour.
Holding a tool is not by itself a violation; only the configured geometry is.
"""

from __future__ import annotations

import math
import re

TOOL_SAFETY_SPEC_VERSION = "tool_safety_spec/1"
STATUS_DISABLED = "disabled"
STATUS_OK = "ok"
STATUS_VIOLATION = "violation"
STATUS_UNKNOWN = "unknown"
STATUSES = (STATUS_DISABLED, STATUS_OK, STATUS_VIOLATION, STATUS_UNKNOWN)

FRAME = {"world": "mujoco_world", "units": "m", "up": "+z", "quaternion": "wxyz"}

PRIMITIVE_KINDS = ("sphere", "segment")
DEFAULT_TOOL_APPROXIMATION = "tool_local_segment"
DEFAULT_ZONE_SOURCE = "user_defined"

NOTE = (
    "工具安全为有限几何近似评估：线段/球近似、world 净距与保护区进入事件。"
    "不代表刀刃伤害程度、材料切割结果或真机安全保证。"
)

MAX_ZONES = 16
MAX_TOOLS = 16
MAX_SEGMENTS_PER_TOOL = 12
# 单步检查行上限：超出即标记 unknown，不允许用截断结果冒充全覆盖
MAX_CHECKS_PER_STEP = 512
# 净距比较容差：数学上恰好等于 0 的边界不能因为浮点误差被判违规
NET_CLEARANCE_TOLERANCE = 1e-12

_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.-]{1,40}$")
_SPEC_KEYS = {"version", "enabled", "risk_zones", "tool_annotations", "note"}
_ZONE_KEYS = {
    "id", "kind", "center", "start", "end", "radius", "clearance", "label", "source",
}
_TOOL_KEYS = {"tool_id", "body", "label", "approximation", "source", "segments"}
_SEGMENT_KEYS = {
    "id", "kind", "center", "start", "end", "radius", "local_pose",
    "label", "approximation", "source",
}
_LOCAL_POSE_KEYS = {"position", "quaternion"}


class ToolSafetySpecError(Exception):
    """Raised for a malformed specification. Never raised for missing scene data."""


class ToolSafetyDataError(Exception):
    """Raised when the scene snapshot cannot supply required data.

    `reason` is a stable code such as `missing_body_pose:knife`. The assessment
    turns it into an unknown reason; it is never turned into a "safe" verdict.
    """

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


# --------------------------------------------------------------------------
# numeric helpers
# --------------------------------------------------------------------------


def _finite(value, where):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolSafetySpecError(f"{where} 必须是数值")
    number = float(value)
    if not math.isfinite(number):
        raise ToolSafetySpecError(f"{where} 必须是有限数值")
    return number


def _vec(raw, size, where):
    if not isinstance(raw, (list, tuple)) or len(raw) != size:
        raise ToolSafetySpecError(f"{where} 必须是 {size} 个数值的数组")
    return tuple(_finite(raw[index], f"{where}[{index}]") for index in range(size))


def _identifier(raw, where, fallback=None):
    value = fallback if raw is None else raw
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ToolSafetySpecError(f"{where} 必须是 1-40 位字母数字或 _.-")
    return value


def _radius(raw, where, allow_zero):
    value = _finite(raw, where)
    if value < 0 or (value == 0 and not allow_zero):
        comparator = "大于等于 0" if allow_zero else "大于 0"
        raise ToolSafetySpecError(f"{where} 必须{comparator}")
    return value


def _dot(left, right):
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _sub(left, right):
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _add(left, right):
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _scale(vector, factor):
    return (vector[0] * factor, vector[1] * factor, vector[2] * factor)


def _norm(vector):
    return math.sqrt(_dot(vector, vector))


def _cross(left, right):
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _clamp(value, low, high):
    return low if value < low else (high if value > high else value)


# --------------------------------------------------------------------------
# quaternion / pose composition (wxyz, +Z)
# --------------------------------------------------------------------------


def quaternion_normalize(quaternion):
    norm = math.sqrt(sum(component * component for component in quaternion))
    if norm <= 0.0:
        raise ToolSafetySpecError("四元数模不可为零")
    return tuple(component / norm for component in quaternion)


def quaternion_rotate(quaternion, vector):
    """Rotate `vector` by a unit quaternion in wxyz order."""
    w, x, y, z = quaternion
    axis = (x, y, z)
    twice_cross = _scale(_cross(axis, vector), 2.0)
    return _add(_add(vector, _scale(twice_cross, w)), _cross(axis, twice_cross))


def quaternion_multiply(left, right):
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def compose_pose(parent_position, parent_quaternion, local_position, local_quaternion):
    """Compose a local pose inside a parent pose. Returns (position, quaternion)."""
    rotated = quaternion_rotate(parent_quaternion, local_position)
    position = _add(parent_position, rotated)
    quaternion = quaternion_normalize(
        quaternion_multiply(parent_quaternion, local_quaternion)
    )
    return position, quaternion


# --------------------------------------------------------------------------
# analytic distance between two finite primitives
# --------------------------------------------------------------------------


def segment_segment_distance(p1, q1, p2, q2):
    """Shortest distance between two segments (Ericson, Real-Time Collision Detection).

    Returns `(distance, closest_on_first, closest_on_second)`. A sphere is the
    degenerate case where the segment endpoints coincide. This is the analytic
    kernel: a tool segment that crosses a zone returns distance 0 regardless of
    how the endpoints are sampled.
    """
    d1 = _sub(q1, p1)
    d2 = _sub(q2, p2)
    offset = _sub(p1, p2)
    a = _dot(d1, d1)
    e = _dot(d2, d2)
    f = _dot(d2, offset)
    epsilon = 1e-12

    if a <= epsilon and e <= epsilon:
        return _norm(offset), p1, p2

    if a <= epsilon:
        s = 0.0
        t = _clamp(f / e, 0.0, 1.0)
    else:
        c = _dot(d1, offset)
        if e <= epsilon:
            t = 0.0
            s = _clamp(-c / a, 0.0, 1.0)
        else:
            b = _dot(d1, d2)
            denominator = a * e - b * b
            s = _clamp((b * f - c * e) / denominator, 0.0, 1.0) if denominator > epsilon else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = _clamp(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = _clamp((b - c) / a, 0.0, 1.0)

    closest_first = _add(p1, _scale(d1, s))
    closest_second = _add(p2, _scale(d2, t))
    return _norm(_sub(closest_first, closest_second)), closest_first, closest_second


# --------------------------------------------------------------------------
# specification normalisation
# --------------------------------------------------------------------------


def default_tool_safety_spec():
    """Disabled, empty spec. Enabling must be explicit."""
    return {
        "version": TOOL_SAFETY_SPEC_VERSION,
        "enabled": False,
        "risk_zones": [],
        "tool_annotations": [],
        "note": NOTE,
    }


def _normalize_local_pose(raw, where):
    if not isinstance(raw, dict):
        raise ToolSafetySpecError(f"{where} 必须是对象")
    unknown = set(raw) - _LOCAL_POSE_KEYS
    if unknown:
        raise ToolSafetySpecError(f"{where} 含未知字段：{','.join(sorted(unknown))}")
    position = _vec(raw.get("position", [0.0, 0.0, 0.0]), 3, f"{where}.position")
    quaternion = _vec(raw.get("quaternion", [1.0, 0.0, 0.0, 0.0]), 4, f"{where}.quaternion")
    return {"position": position, "quaternion": quaternion_normalize(quaternion)}


def _normalize_primitive(raw, where, allow_zero_radius):
    kind = raw.get("kind", "segment")
    if kind not in PRIMITIVE_KINDS:
        raise ToolSafetySpecError(f"{where}.kind 必须是 " + " / ".join(PRIMITIVE_KINDS))
    radius = _radius(raw.get("radius", 0.0), f"{where}.radius", allow_zero_radius)
    if kind == "sphere":
        center = _vec(raw.get("center"), 3, f"{where}.center")
        start = end = center
    else:
        start = _vec(raw.get("start"), 3, f"{where}.start")
        end = _vec(raw.get("end"), 3, f"{where}.end")
        center = None
    return {"kind": kind, "start": start, "end": end, "center": center, "radius": radius}


def _normalize_zone(raw, index):
    where = f"risk_zones[{index}]"
    if not isinstance(raw, dict):
        raise ToolSafetySpecError(f"{where} 必须是对象")
    unknown = set(raw) - _ZONE_KEYS
    if unknown:
        raise ToolSafetySpecError(f"{where} 含未知字段：{','.join(sorted(unknown))}")
    zone_id = _identifier(raw.get("id"), f"{where}.id")
    primitive = _normalize_primitive(raw, where, allow_zero_radius=False)
    clearance = _finite(raw.get("clearance", 0.0), f"{where}.clearance")
    if clearance < 0:
        raise ToolSafetySpecError(f"{where}.clearance 必须大于等于 0")
    return {
        "id": zone_id,
        "kind": primitive["kind"],
        "start": primitive["start"],
        "end": primitive["end"],
        "center": primitive["center"],
        "radius": primitive["radius"],
        "clearance": clearance,
        "label": str(raw.get("label") or zone_id),
        "source": str(raw.get("source") or DEFAULT_ZONE_SOURCE),
    }


def _normalize_segment(raw, index, tool_id):
    where = f"tool_annotations[{tool_id}].segments[{index}]"
    if not isinstance(raw, dict):
        raise ToolSafetySpecError(f"{where} 必须是对象")
    unknown = set(raw) - _SEGMENT_KEYS
    if unknown:
        raise ToolSafetySpecError(f"{where} 含未知字段：{','.join(sorted(unknown))}")
    segment_id = _identifier(raw.get("id"), f"{where}.id", fallback=f"seg{index}")
    kind = raw.get("kind", "segment")
    primitive = _normalize_primitive(raw, where, allow_zero_radius=kind == "segment")
    local_pose = raw.get("local_pose")
    return {
        "id": segment_id,
        "kind": primitive["kind"],
        "start": primitive["start"],
        "end": primitive["end"],
        "center": primitive["center"],
        "radius": primitive["radius"],
        "local_pose": _normalize_local_pose(local_pose, f"{where}.local_pose") if local_pose else None,
        "label": str(raw.get("label") or segment_id),
        "approximation": str(raw.get("approximation") or DEFAULT_TOOL_APPROXIMATION),
        "source": str(raw.get("source") or DEFAULT_ZONE_SOURCE),
    }


def _normalize_tool(raw, index):
    where = f"tool_annotations[{index}]"
    if not isinstance(raw, dict):
        raise ToolSafetySpecError(f"{where} 必须是对象")
    unknown = set(raw) - _TOOL_KEYS
    if unknown:
        raise ToolSafetySpecError(f"{where} 含未知字段：{','.join(sorted(unknown))}")
    tool_id = _identifier(raw.get("tool_id"), f"{where}.tool_id")
    body = raw.get("body")
    if not isinstance(body, str) or not body.strip():
        raise ToolSafetySpecError(f"{where}.body 必填：工具 body 名称或 id")
    segments_raw = raw.get("segments") or []
    if not isinstance(segments_raw, list):
        raise ToolSafetySpecError(f"{where}.segments 必须是数组")
    if len(segments_raw) > MAX_SEGMENTS_PER_TOOL:
        raise ToolSafetySpecError(f"{where}.segments 最多 {MAX_SEGMENTS_PER_TOOL} 条")
    segments = [_normalize_segment(item, position, tool_id) for position, item in enumerate(segments_raw)]
    ids = [segment["id"] for segment in segments]
    if len(set(ids)) != len(ids):
        raise ToolSafetySpecError(f"{where}.segments id 重复")
    return {
        "tool_id": tool_id,
        "body": body,
        "label": str(raw.get("label") or tool_id),
        "approximation": str(raw.get("approximation") or DEFAULT_TOOL_APPROXIMATION),
        "source": str(raw.get("source") or DEFAULT_ZONE_SOURCE),
        "segments": segments,
    }


def normalize_tool_safety_spec(raw):
    """Validate and canonicalise a user spec. Unknown keys are rejected, not ignored."""
    if raw is None:
        return default_tool_safety_spec()
    if not isinstance(raw, dict):
        raise ToolSafetySpecError("工具安全配置必须是对象")
    unknown = set(raw) - _SPEC_KEYS
    if unknown:
        raise ToolSafetySpecError(f"未知配置项：{','.join(sorted(unknown))}")

    version = raw.get("version", TOOL_SAFETY_SPEC_VERSION)
    if version != TOOL_SAFETY_SPEC_VERSION:
        raise ToolSafetySpecError(
            f"version 必须是 {TOOL_SAFETY_SPEC_VERSION}，收到 {version!r}"
        )
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ToolSafetySpecError("enabled 必须是布尔值")

    zones_raw = raw.get("risk_zones") or []
    tools_raw = raw.get("tool_annotations") or []
    if not isinstance(zones_raw, list) or not isinstance(tools_raw, list):
        raise ToolSafetySpecError("risk_zones 与 tool_annotations 必须是数组")
    if len(zones_raw) > MAX_ZONES:
        raise ToolSafetySpecError(f"risk_zones 最多 {MAX_ZONES} 个")
    if len(tools_raw) > MAX_TOOLS:
        raise ToolSafetySpecError(f"tool_annotations 最多 {MAX_TOOLS} 个")

    zones = [_normalize_zone(item, index) for index, item in enumerate(zones_raw)]
    tools = [_normalize_tool(item, index) for index, item in enumerate(tools_raw)]
    zone_ids = [zone["id"] for zone in zones]
    tool_ids = [tool["tool_id"] for tool in tools]
    if len(set(zone_ids)) != len(zone_ids):
        raise ToolSafetySpecError("risk_zones id 重复")
    if len(set(tool_ids)) != len(tool_ids):
        raise ToolSafetySpecError("tool_annotations tool_id 重复")

    return {
        "version": TOOL_SAFETY_SPEC_VERSION,
        "enabled": enabled,
        "risk_zones": zones,
        "tool_annotations": tools,
        "note": str(raw.get("note") or NOTE),
    }


# --------------------------------------------------------------------------
# scene snapshot resolution
# --------------------------------------------------------------------------


def build_body_index(geometry):
    """Map body name -> body id string from the static geometry object dictionary."""
    index = {}
    if not isinstance(geometry, dict):
        return index
    robot = geometry.get("robot")
    if isinstance(robot, dict):
        for row in robot.get("bodies") or []:
            if not isinstance(row, dict):
                continue
            name = row.get("name")
            if isinstance(name, str) and name:
                index[name] = str(row.get("id"))
    for item in geometry.get("objects") or []:
        if not isinstance(item, dict):
            continue
        identifier = str(item.get("id")) if item.get("id") is not None else None
        for key in ("name", "body"):
            name = item.get(key)
            if isinstance(name, str) and name and identifier is not None:
                index.setdefault(name, identifier)
    return index


def _read_pose(raw, where):
    if not isinstance(raw, dict):
        raise ToolSafetyDataError(f"invalid_body_pose:{where}")
    position_raw = raw.get("p", raw.get("position"))
    quaternion_raw = raw.get("q", raw.get("quaternion"))
    try:
        position = _vec(position_raw, 3, f"{where}.position")
    except ToolSafetySpecError:
        raise ToolSafetyDataError(f"invalid_body_pose:{where}") from None
    # 缺朝向不猜：没有四元数就无法把工具局部段放到世界系，必须按 unknown 处理
    if quaternion_raw is None:
        raise ToolSafetyDataError(f"missing_body_orientation:{where}")
    try:
        quaternion = quaternion_normalize(_vec(quaternion_raw, 4, f"{where}.quaternion"))
    except ToolSafetySpecError:
        raise ToolSafetyDataError(f"invalid_body_orientation:{where}") from None
    return position, quaternion


def resolve_body_pose(transforms, body_ref, body_index=None):
    """World pose of a tool body, from polled transforms.

    `transforms` may be keyed by body id string (as scene_geometry emits) or by
    body name. Raises ToolSafetyDataError with a stable reason when the pose is
    absent or unusable, so the caller can report unknown instead of safe.
    """
    if not isinstance(body_ref, str) or not body_ref:
        raise ToolSafetyDataError("missing_tool_body")
    if not isinstance(transforms, dict):
        raise ToolSafetyDataError(f"missing_body_pose:{body_ref}")
    candidates = [body_ref]
    if body_index:
        mapped = body_index.get(body_ref)
        if mapped is not None and str(mapped) not in candidates:
            candidates.append(str(mapped))
    for candidate in candidates:
        if candidate in transforms:
            return _read_pose(transforms[candidate], body_ref)
    raise ToolSafetyDataError(f"missing_body_pose:{body_ref}")


def _world_primitive(segment, pose):
    position, quaternion = pose
    local_pose = segment.get("local_pose")
    if local_pose:
        position, quaternion = compose_pose(
            position, quaternion, local_pose["position"], local_pose["quaternion"]
        )
    start = _add(position, quaternion_rotate(quaternion, segment["start"]))
    end = _add(position, quaternion_rotate(quaternion, segment["end"]))
    return start, end


def _finite_point(point):
    return all(math.isfinite(component) for component in point)


# --------------------------------------------------------------------------
# assessment
# --------------------------------------------------------------------------


def _zone_overlay(zone):
    return {
        "id": zone["id"],
        "kind": zone["kind"],
        "center": list(zone["center"]) if zone["center"] else None,
        "start": list(zone["start"]),
        "end": list(zone["end"]),
        "radius": zone["radius"],
        "clearance": zone["clearance"],
        "label": zone["label"],
        "source": zone["source"],
    }


def _tool_overlay(tool):
    return {
        "tool_id": tool["tool_id"],
        "body": tool["body"],
        "label": tool["label"],
        "approximation": tool["approximation"],
        "source": tool["source"],
        "segments": [
            {
                "id": segment["id"],
                "kind": segment["kind"],
                "start": list(segment["start"]),
                "end": list(segment["end"]),
                "radius": segment["radius"],
                "approximation": segment["approximation"],
                "source": segment["source"],
            }
            for segment in tool["segments"]
        ],
    }


def _empty_result(status, scene_state):
    return {
        "spec_version": TOOL_SAFETY_SPEC_VERSION,
        "enabled": False,
        "status": status,
        "session_id": scene_state.get("session_id"),
        "episode": scene_state.get("episode"),
        "step": scene_state.get("step"),
        "sim_seconds": scene_state.get("sim_seconds"),
        "checks": [],
        "violations": [],
        "zone_clearance": [],
        "unknown_reasons": [],
        "sources": {"zones": [], "tools": []},
        "approximations": [],
        "overlays": {
            "risk_zones": [],
            "tool_annotations": [],
            "world_tool_segments": [],
            "source": "tool_safety_disabled",
            "reason": None,
        },
        "frame": dict(FRAME),
        "note": NOTE,
    }


def assess_tool_safety(spec, scene_state):
    """Assess one scene snapshot. Pure: no session state, no MuJoCo.

    `status` precedence is violation > unknown > ok, so a partially unknown
    assessment never reads as safe.

    This evaluates the sampled step, it does not sweep the interval between two
    steps: see `sampling` / `collision_guarantee` in the result. A fast tool can
    pass through a zone between two sampled steps without either step showing a
    violation.

    The spec is always revalidated, even when it already looks canonical: a
    field-set shortcut would let a raw dict with a NaN slip through unvalidated.
    `normalize_tool_safety_spec` is idempotent, so this is safe to call per poll.
    """
    if not isinstance(scene_state, dict):
        scene_state = {}
    spec = normalize_tool_safety_spec(spec)

    if not spec.get("enabled"):
        result = _empty_result(STATUS_DISABLED, scene_state)
        result["overlays"]["reason"] = "spec_disabled"
        return result

    zones = spec["risk_zones"]
    tools = spec["tool_annotations"]
    unknown_reasons = []
    if not zones:
        unknown_reasons.append("no_risk_zones")
    if not tools:
        unknown_reasons.append("no_tool_annotations")

    transforms = scene_state.get("body_transforms")
    if not isinstance(transforms, dict):
        transforms = scene_state.get("transforms")
    transforms = transforms if isinstance(transforms, dict) else {}
    body_index = scene_state.get("body_index")
    if not isinstance(body_index, dict):
        body_index = build_body_index(scene_state.get("geometry") or {})

    world_segments = []
    approximations = set()
    for tool in tools:
        approximations.add(tool["approximation"])
        try:
            pose = resolve_body_pose(transforms, tool["body"], body_index)
        except ToolSafetyDataError as error:
            unknown_reasons.append(f"{error.reason}:{tool['tool_id']}")
            continue
        if not tool["segments"]:
            unknown_reasons.append(f"no_segments:{tool['tool_id']}")
            continue
        for segment in tool["segments"]:
            approximations.add(segment["approximation"])
            start, end = _world_primitive(segment, pose)
            if not _finite_point(start) or not _finite_point(end):
                unknown_reasons.append(f"invalid_world_geometry:{tool['tool_id']}:{segment['id']}")
                continue
            world_segments.append({
                "tool_id": tool["tool_id"],
                "segment_id": segment["id"],
                "kind": segment["kind"],
                "body": tool["body"],
                "start": start,
                "end": end,
                "radius": segment["radius"],
                "approximation": segment["approximation"],
                "source": segment["source"],
            })

    checks = []
    violations = []
    truncated = False
    for zone in zones:
        for index, segment in enumerate(world_segments):
            if len(checks) >= MAX_CHECKS_PER_STEP:
                truncated = True
                break
            distance, _, _ = segment_segment_distance(
                segment["start"], segment["end"], zone["start"], zone["end"]
            )
            surface_distance = distance - segment["radius"] - zone["radius"]
            net_clearance = surface_distance - zone["clearance"]
            row = {
                "zone_id": zone["id"],
                "tool_id": segment["tool_id"],
                "segment_id": segment["segment_id"],
                "distance": distance,
                "surface_distance": surface_distance,
                "margin": zone["clearance"],
                "net_clearance": net_clearance,
                "violation": net_clearance < -NET_CLEARANCE_TOLERANCE,
                "zone_source": zone["source"],
                "tool_source": segment["source"],
                "approximation": segment["approximation"],
                "world_segment_index": index,
            }
            checks.append(row)
            if row["violation"]:
                violations.append({**row, "rule_id": f"{zone['id']}::{segment['tool_id']}::{segment['segment_id']}"})
        if truncated:
            break
    if truncated:
        unknown_reasons.append("checks_truncated")

    zone_clearance = []
    for zone in zones:
        rows = [row for row in checks if row["zone_id"] == zone["id"]]
        if not rows:
            zone_clearance.append({
                "zone_id": zone["id"],
                "status": STATUS_UNKNOWN,
                "min_net_clearance": None,
                "min_distance": None,
                "reason": "no_evaluated_segment",
            })
            continue
        best = min(rows, key=lambda row: row["net_clearance"])
        zone_clearance.append({
            "zone_id": zone["id"],
            "status": STATUS_VIOLATION if best["violation"] else STATUS_OK,
            "min_net_clearance": best["net_clearance"],
            "min_distance": best["distance"],
            "nearest_tool_id": best["tool_id"],
            "nearest_segment_id": best["segment_id"],
            "approximation": best["approximation"],
            "source": best["zone_source"],
        })

    if violations:
        status = STATUS_VIOLATION
    elif unknown_reasons:
        status = STATUS_UNKNOWN
    else:
        status = STATUS_OK

    return {
        "spec_version": TOOL_SAFETY_SPEC_VERSION,
        "enabled": True,
        "status": status,
        "sampling": "discrete_step",
        "collision_guarantee": False,
        "session_id": scene_state.get("session_id"),
        "episode": scene_state.get("episode"),
        "step": scene_state.get("step"),
        "sim_seconds": scene_state.get("sim_seconds"),
        "checks": checks,
        "violations": violations,
        "zone_clearance": zone_clearance,
        "unknown_reasons": unknown_reasons,
        "sources": {
            "zones": sorted({zone["source"] for zone in zones}),
            "tools": sorted({tool["source"] for tool in tools}),
        },
        "approximations": sorted(approximations),
        "overlays": {
            "risk_zones": [_zone_overlay(zone) for zone in zones],
            "tool_annotations": [_tool_overlay(tool) for tool in tools],
            "world_tool_segments": [
                {
                    "tool_id": segment["tool_id"],
                    "segment_id": segment["segment_id"],
                    "kind": segment["kind"],
                    "start": list(segment["start"]),
                    "end": list(segment["end"]),
                    "radius": segment["radius"],
                    "approximation": segment["approximation"],
                    "source": segment["source"],
                }
                for segment in world_segments
            ],
            "source": "user_defined_annotations",
            "reason": unknown_reasons[0] if unknown_reasons else None,
        },
        "frame": dict(FRAME),
        "note": NOTE,
    }
