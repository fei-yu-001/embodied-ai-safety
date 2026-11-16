"""Real scene geometry for the 3D inspection view.

Static geometry (shape, size, colour, local pose, stable geom id) and dynamic
transforms (body poses) are built separately so polling only ships poses.
Nothing here invents a robot: every shape comes from the live MuJoCo model, and
what is approximated or omitted is named in `provenance`.

Geoms carry their own local frame. A geom placed at `geom_pos`/`geom_quat`
inside a body must be composed with the body pose, otherwise every link's
shapes collapse onto the body origin.
"""

from __future__ import annotations

import hashlib

import numpy as np

ROBOT_BODY_PREFIX = "robot0_"
GRIPPER_PREFIX = "gripper0_"
MAX_OBJECT_BODIES = 24
MAX_ROBOT_GEOMS = 200
MAX_WORLD_GEOMS = 400
MAX_MISSING_LISTED = 12
KITCHEN_PRIMITIVES = {"box", "cylinder"}

# worker_protocol 的单条消息上限是 1,000,000 字节；这里留出信封与转义余量。
QUERY_BYTE_BUDGET = 900_000

# 实测本任务：group 0 = 碰撞代理，group 1 = 视觉几何，group 2 只有
# left/right_eef_target_box|sphere —— MuJoCo 调试目标标记，不是场景外观。
VISUAL_GROUPS = {1}
# 原图 data/sessions/sess_5619b7e491e9/frames/latest.jpg：木柜、灰墙、黑石台、白臂。
NEUTRAL_GEOM_RGBA = [0.62, 0.62, 0.64, 1.0]

GEOM_TYPE_NAMES = {
    2: "sphere",
    3: "capsule",
    4: "ellipsoid",
    5: "cylinder",
    6: "box",
    7: "mesh",
}


def _encoded_size(payload):
    import json

    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def geometry_key(model, task=None):
    """Stable key for this model's static geometry.

    Covers everything that changes what is drawn: geom type/size/local pose,
    colour, body composition and naming. Two layouts with identical sizes but
    different placement or naming must not share a key.
    """
    digest = hashlib.sha256()
    for array in (model.geom_type, model.body_geomnum, model.body_parentid, model.body_geomadr):
        digest.update(np.asarray(array, dtype=np.int32).tobytes())
    for array in (model.geom_size, model.geom_pos, model.geom_rgba):
        digest.update(np.asarray(array, dtype=np.float64).tobytes())
    digest.update(np.asarray(model.geom_quat, dtype=np.float64).tobytes())
    # mesh 数据也参与：不同布局可能复用同样的 geom_size
    # 材质绑定与分组决定外观；不纳入会导致换材质后命中旧缓存
    for attribute in ("geom_group", "geom_matid", "mat_texid", "geom_dataid",
                      "mesh_vertadr", "mesh_vertnum", "tex_dataadr",
                      "tex_adr", "tex_width", "tex_height", "tex_nchannel"):
        array = getattr(model, attribute, None)
        if array is not None:
            try:
                digest.update(np.asarray(array, dtype=np.int64).tobytes())
            except (TypeError, ValueError):
                pass
    for attribute in ("mat_rgba", "tex_rgba"):
        array = getattr(model, attribute, None)
        if array is not None:
            try:
                digest.update(np.asarray(array, dtype=np.float64).tobytes())
            except (TypeError, ValueError):
                pass
    for attribute in ("geom_dataid", "mesh_vertadr", "mesh_vertnum"):
        array = getattr(model, attribute, None)
        if array is not None:
            digest.update(np.asarray(array, dtype=np.int64).tobytes())
    vertices = getattr(model, "mesh_vert", None)
    if vertices is not None:
        digest.update(np.asarray(vertices, dtype=np.float64).tobytes())
    # 命名参与哈希：不同布局可能复用同样的几何尺寸
    for index in range(model.nbody):
        digest.update((model.body_id2name(index) or "").encode("utf-8"))
    for name in _object_names(model, task):
        digest.update(name.encode("utf-8"))
    return f"g{int(model.ngeom)}b{int(model.nbody)}_{digest.hexdigest()[:16]}"


def _object_names(model, task):
    names = []
    for index, _name in _object_bodies(model, task):
        names.append(_name)
    return names


def _size(size, max_len=3):
    return [round(float(v), 5) for v in np.asarray(size, dtype=float).reshape(-1)[:max_len]]


def _mesh_bbox(model, geom_id):
    """Half-extents and centre offset of a mesh geom, from compiled vertices.

    Measured on MuJoCo 3.3.1: `mesh_vert` is already compiled, so it carries
    the asset scale, and MuJoCo recentres vertices onto the geom frame (the
    offset is folded into `geom_pos`). `geom_size` is therefore NOT a mesh
    scale and must not be applied again -- doing so inflated a 2x1x4-scaled
    test mesh from half [1,1,6] to [1,1,36].
    """
    try:
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0:
            return None
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        if count <= 0:
            return None
        verts = np.asarray(model.mesh_vert[start:start + count], dtype=float).reshape(-1, 3)
        low, high = verts.min(axis=0), verts.max(axis=0)
        half = (high - low) / 2.0
        centre = (high + low) / 2.0
        return {
            "size": [round(float(v), 5) for v in half],
            "centre": [round(float(v), 5) for v in centre],
        }
    except (AttributeError, IndexError, TypeError, ValueError):
        return None


def _vec(values, length):
    return [round(float(v), 5) for v in np.asarray(values, dtype=float).reshape(-1)[:length]]


def _texture_mean_rgb(model, texture_id):
    """Mean RGB of an embedded MuJoCo texture, as a 0..1 triple.

    RoboCasa stores the real appearance in textures (`mat_rgba` is a white tint),
    so the material tint alone renders cabinets white. Averaging the embedded
    pixels is a measurement of the texture, not a guess at a nicer colour.
    Returns None when the texture cannot be read.
    """
    try:
        adr = int(model.tex_adr[texture_id])
        width = int(model.tex_width[texture_id])
        height = int(model.tex_height[texture_id])
        channel = int(model.tex_nchannel[texture_id])
        if channel < 3 or width <= 0 or height <= 0:
            return None
        pixels = np.asarray(model.tex_data[adr:adr + width * height * channel], dtype=np.float64)
        pixels = pixels.reshape(-1, channel)[:, :3]
        if pixels.size == 0:
            return None
        mean = pixels.mean(axis=0) / 255.0
        return [round(float(min(1.0, max(0.0, c))), 3) for c in mean]
    except (AttributeError, IndexError, TypeError, ValueError):
        return None


def _colour(model, geom_id):
    """Effective visible colour of a geom.

    Priority: the material's texture (measured mean), then the material tint,
    then the geom's own rgba. Textures are not exported as images yet, so the
    mean carries the real hue while the HUD reports textures are approximate.
    """
    material = -1
    try:
        material = int(model.geom_matid[geom_id])
    except (AttributeError, IndexError, TypeError, ValueError):
        material = -1
    if material >= 0:
        try:
            slots = np.asarray(model.mat_texid[material]).reshape(-1)
        except (AttributeError, IndexError, TypeError, ValueError):
            slots = np.asarray([], dtype=int)
        for slot in slots:
            slot = int(slot)
            if slot < 0:
                continue
            mean = _texture_mean_rgb(model, slot)
            if mean is not None:
                alpha = 1.0
                try:
                    mat = np.asarray(model.mat_rgba[material], dtype=float).reshape(-1)
                    if mat.size > 3:
                        alpha = round(float(min(1.0, max(0.0, mat[3]))), 3)
                except (AttributeError, IndexError, TypeError, ValueError):
                    pass
                return mean, alpha, "material_texture"
        try:
            mat = np.asarray(model.mat_rgba[material], dtype=float).reshape(-1)
            rgb = [round(float(min(1.0, max(0.0, c))), 3) for c in mat[:3]]
            alpha = round(float(min(1.0, max(0.0, mat[3] if mat.size > 3 else 1.0))), 3)
            return rgb, alpha, "material"
        except (AttributeError, IndexError, TypeError, ValueError):
            pass
    rgba = np.asarray(model.geom_rgba[geom_id], dtype=float).reshape(-1)
    rgb = [round(float(min(1.0, max(0.0, c))), 3) for c in rgba[:3]]
    alpha = round(float(min(1.0, max(0.0, rgba[3] if rgba.size > 3 else 1.0))), 3)
    return rgb, alpha, "geom_rgba"


def _entries(model, body_ids, limit, mesh_policy="approximate", visual_only=True):
    """Geometry entries with local pose. Returns (entries, truncated).

    `visual_only` drops group-0 collision proxies and fully transparent geoms,
    which is what made cabinets look like flat brown blocks and green boxes.
    """
    entries = []
    skipped = {"collision": 0, "transparent": 0}
    for body_id in body_ids:
        start = int(model.body_geomadr[body_id])
        count = int(model.body_geomnum[body_id])
        for geom_id in range(start, start + count):
            kind = GEOM_TYPE_NAMES.get(int(model.geom_type[geom_id]))
            if kind is None:
                continue
            if kind == "mesh" and mesh_policy == "skip":
                continue
            group = int(model.geom_group[geom_id]) if hasattr(model, "geom_group") else 0
            if visual_only and group not in VISUAL_GROUPS:
                skipped["collision"] += 1
                continue
            rgb, alpha, colour_source = _colour(model, geom_id)
            if visual_only and alpha <= 0.01:
                skipped["transparent"] += 1
                continue
            is_mesh = kind == "mesh"
            size = _size(model.geom_size[geom_id])
            # 基元几何以 geom_pos 为中心；mesh 用编译顶点算出的包围盒
            centre = [0.0, 0.0, 0.0]
            representation = "primitive"
            if is_mesh:
                bbox = _mesh_bbox(model, geom_id)
                if bbox:
                    size = bbox["size"]
                    centre = bbox["centre"]
                    representation = "mesh_bbox"
                else:
                    # 拿不到顶点时退回小方盒，并标明这是占位
                    size = [0.02, 0.02, 0.02]
                    representation = "mesh_placeholder"
            entries.append({
                "body": int(body_id),
                "geom": int(geom_id),
                "kind": kind,
                "size": size,
                "bbox_centre": centre,
                # 相对所属 body 的局部变换：前端必须与 body pose 复合
                "local_position": _vec(model.geom_pos[geom_id], 3),
                "local_quaternion": _vec(model.geom_quat[geom_id], 4),
                "colour": rgb,
                "alpha": alpha,
                "colour_source": colour_source,
                "group": group,
                # 原始 mesh 未导出：按真实包围盒或占位体近似绘制
                "representation": representation,
                "source": "mj_model_geom",
            })
            if len(entries) >= limit:
                return entries, True, skipped
    return entries, False, skipped


def _body_names(model):
    return [model.body_id2name(index) or f"body_{index}" for index in range(model.nbody)]


def _object_bodies(model, task):
    """Return [(body_id, display_name)] for the task's movable objects."""
    rows = []
    task_objects = getattr(task, "objects", None) or []
    if isinstance(task_objects, dict):
        task_objects = list(task_objects.values())
    for obj in list(task_objects)[:MAX_OBJECT_BODIES]:
        root = getattr(obj, "root_body", None)
        if not root:
            continue
        try:
            index = int(model.body_name2id(str(root)))
        except (KeyError, ValueError):
            continue
        rows.append((index, str(getattr(obj, "name", root))))
    return rows


def build_geometry(model, task, robot):
    """Static geometry only. Safe to cache and re-serve without re-sending."""
    names = _body_names(model)
    robot_bodies = [i for i, n in enumerate(names) if n.startswith(ROBOT_BODY_PREFIX) or n.startswith(GRIPPER_PREFIX)]
    object_rows = _object_bodies(model, task)
    object_bodies = [index for index, _ in object_rows]
    object_names = dict(object_rows)

    kitchen_bodies = []
    for index in range(model.nbody):
        if index in robot_bodies or index in object_bodies:
            continue
        if int(model.body_geomnum[index]) <= 0 or int(model.body_geomadr[index]) < 0:
            continue
        kitchen_bodies.append(index)

    robot_geoms, robot_truncated, robot_skipped = _entries(model, robot_bodies, MAX_ROBOT_GEOMS)
    kitchen_geoms, kitchen_truncated, kitchen_skipped = _entries(model, kitchen_bodies, MAX_WORLD_GEOMS)
    # 厨房也保留视觉 mesh（此前只留 box/cylinder，把橱柜视觉网格整片丢掉了）
    primitives = [entry for entry in kitchen_geoms if entry["kind"] in KITCHEN_PRIMITIVES]
    visuals = [entry for entry in kitchen_geoms if entry["kind"] not in KITCHEN_PRIMITIVES]

    # 任务物体带自己的几何，形状/局部变换与机器人同源
    objects = []
    for index, name in object_rows:
        geoms, _truncated, _skipped = _entries(model, [index], MAX_WORLD_GEOMS)
        objects.append({
            "id": str(index),
            "name": name,
            "body": names[index],
            "geoms": geoms,
        })
    object_geom_count = sum(len(item["geoms"]) for item in objects)
    kitchen_visual_meshes = sum(1 for e in kitchen_geoms if e["kind"] not in KITCHEN_PRIMITIVES)
    kitchen_primitives = len(kitchen_geoms) - kitchen_visual_meshes

    exported = set(robot_bodies) | set(object_bodies) | set(kitchen_bodies)
    missing = [names[i] for i in range(model.nbody) if i not in exported]

    geometry = {
        "key": geometry_key(model, task),
        "frame": {"world": "mujoco_world", "units": "m", "up": "+z", "quaternion": "wxyz"},
        "provenance": {
            "geometry": "mj_model_geom",
            "representation": "visual_group + mesh_bbox",
            "note": (
                "连杆、夹爪、任务物体与厨房几何取自 MuJoCo 视觉组（group 1/2）；"
                "group 0 是碰撞代理（robosuite 重着色并移除材质），已排除；"
                "颜色优先取材质纹理的平均色（纹理图未导出，属近似），"
                "其次取材质 tint，再次取 geom_rgba；"
                "纹理图像与原始 mesh 未导出，mesh 按真实包围盒绘制"
            ),
            "visual_only": True,
            "colour_sources": ["material_texture", "material", "geom_rgba"],
            "textures_exported": False,
            "texture_colour": "mean_of_embedded_texture",
            "bodies_total": int(model.nbody),
            "bodies_exported": len(exported),
            # 必须等于实际发出的数量：robot + objects + world(primitives + visual meshes)
            "geoms_exported": len(robot_geoms) + object_geom_count + len(kitchen_geoms),
            "geoms_robot": len(robot_geoms),
            "geoms_objects": object_geom_count,
            "geoms_world_primitives": kitchen_primitives,
            "geoms_world_visual_meshes": kitchen_visual_meshes,
            "geoms_total": int(model.ngeom),
            # 未导出 = 碰撞代理 + 全透明 + 非视觉分组；统计必须能对上总数
            "geoms_skipped": (
                kitchen_skipped["collision"] + robot_skipped["collision"]
                + kitchen_skipped["transparent"] + robot_skipped["transparent"]
            ),
            "truncated": bool(robot_truncated or kitchen_truncated),
            "missing_bodies": missing[:MAX_MISSING_LISTED],
            "missing_count": len(missing),
        },
        "robot": {
            "bodies": [{"id": str(i), "name": names[i]} for i in robot_bodies],
            "geoms": robot_geoms,
            "hierarchy": _hierarchy(model, names, robot_bodies),
        },
        "objects": objects,
        "world": {
            # 只放一份几何；重复字段会把静态载荷成倍序列化
            "geoms": kitchen_geoms,
            "counts": {
                "total": len(kitchen_geoms),
                "primitives": len(primitives),
                "visual_meshes": len(visuals),
            },
            "visual_only": True,
            "skipped_collision": kitchen_skipped["collision"] + robot_skipped["collision"],
            "skipped_transparent": kitchen_skipped["transparent"] + robot_skipped["transparent"],
        },
    }
    return geometry


def fit_to_budget(geometry, budget=QUERY_BYTE_BUDGET):
    """Shrink static geometry until it fits the worker message budget.

    Degradation is ordered, recorded, and guaranteed to terminate: each step
    strictly reduces the payload, and metadata is trimmed before giving up.
    """
    steps = []
    size = _encoded_size(geometry)

    def remeasure():
        return _encoded_size(geometry)

    if size > budget:
        if geometry["world"]["geoms"]:
            geometry["world"]["geoms"] = []
            steps.append("world_geoms_dropped")
            size = remeasure()

    if size > budget:
        before = len(geometry["robot"]["geoms"])
        geometry["robot"]["geoms"] = [g for g in geometry["robot"]["geoms"] if g["kind"] != "mesh"]
        if len(geometry["robot"]["geoms"]) != before:
            steps.append("robot_mesh_approximations_dropped")
            size = remeasure()

    if size > budget:
        world_before = len(geometry["world"]["geoms"])
        geometry["world"]["geoms"] = []
        if world_before:
            steps.append("world_geoms_dropped")
            size = remeasure()

    # 严格递减的截断：len==1 时不能再取 max(1, len//2)
    while size > budget and geometry["robot"]["geoms"]:
        current = len(geometry["robot"]["geoms"])
        keep = current // 2 if current > 1 else 0
        geometry["robot"]["geoms"] = geometry["robot"]["geoms"][:keep] if keep else []
        size = remeasure()
        if "robot_geoms_halved" not in steps:
            steps.append("robot_geoms_halved")

    # 元数据本身也可能超限：先瘦身曝光字段，再放弃几何
    if size > budget:
        geometry["provenance"]["missing_bodies"] = geometry["provenance"]["missing_bodies"][:3]
        geometry["provenance"]["note"] = "载荷超限，已降级为最小元数据"
        for item in geometry["objects"]:
            item["geoms"] = []
        size = remeasure()
        steps.append("metadata_trimmed")

    if size > budget:
        geometry["robot"]["hierarchy"] = []
        geometry["robot"]["bodies"] = []
        geometry["objects"] = []
        size = remeasure()
        steps.append("hierarchy_dropped")

    if size > budget:
        geometry["key"] = str(geometry.get("key", ""))[:24]
        geometry["provenance"] = {
            "geometry": "mj_model_geom",
            "representation": "unavailable",
            "note": "静态几何超出单条消息预算，未发送",
            "geoms_exported": 0,
            "geoms_total": int(geometry.get("provenance", {}).get("geoms_total", 0)),
            "truncated": True,
            "missing_bodies": [],
            "missing_count": 0,
        }
        size = remeasure()
        steps.append("geometry_omitted")

    geometry["provenance"]["truncated"] = bool(geometry["provenance"].get("truncated") or steps)
    geometry["provenance"]["degradation"] = steps
    geometry["budget"] = {"bytes": size, "budget": budget, "degraded": bool(steps), "steps": steps}
    return geometry


def build_transforms(model, data):
    """Dynamic body poses for this instant. Small payload, sent every poll."""
    import mujoco

    poses = {}
    for index in range(model.nbody):
        position = np.asarray(data.body_xpos[index], dtype=float).reshape(3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.asarray(data.body_xmat[index], dtype=float).reshape(9))
        poses[str(index)] = {
            "p": [round(float(v), 5) for v in position],
            "q": [round(float(v), 5) for v in quat],
        }
    return poses


def _hierarchy(model, names, indexes):
    rows = []
    for index in indexes:
        parent = int(model.body_parentid[index])
        rows.append({
            "id": str(index),
            "name": names[index],
            "parent": str(parent) if parent else None,
        })
    return rows
