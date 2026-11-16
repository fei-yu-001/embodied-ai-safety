"""Real mesh and texture assets for the 3D inspection view.

`scene_geometry` can only afford a bounding box per mesh, because one worker
message is capped at 1,000,000 bytes. Real triangles, normals, UVs and texture
images are therefore exported once into a session cache next to the frames, and
the scene payload only needs a short relative URL per geom.

Everything here is read from the live compiled `mjModel`. Nothing is resampled,
decimated or invented, and no external path is ever opened: the model already
holds its assets in memory. Anything that cannot be exported is named in
`unsupported` with a machine-readable reason code.

Measured on MuJoCo 3.3.1 (reproduced in `server/tests/test_visual_assets.py` and
`server/scripts/smoke_visual_assets.py`):

- `mesh_vert` is already compiled. The asset `scale` is applied and MuJoCo
  re-centres the vertices onto the geom frame, so `geom_size` is NOT a mesh
  scale and must not be multiplied in again. Composing
  `body_pose * geom_pos/geom_quat * vertex` reproduces the original XML vertex
  to 1e-14 (checked against a site carrying the same XML pose).
- `mesh_face`, `mesh_facenormal` and `mesh_facetexcoord` hold LOCAL indices
  into that mesh's own `mesh_vert` / `mesh_normal` / `mesh_texcoord` slice.
  Two meshes compiled into one model never share an index space.
- `mesh_normal` is already in the compiled geom frame: it equals the geometric
  normal of the compiled triangle.
- `tex_data` is the post-`vflip` image stored row-major with row 0 at the top,
  which is the convention a glTF / three.js texture expects with the default
  `flipY = true`. The compiled UVs already agree: uv (0, 1) samples row 0.
- A cube texture is ONE image with six faces stacked as `h == 6 * w`, in an
  order derived from the XML `gridlayout`; a single square file is replicated
  on all six faces.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
from pathlib import Path
import uuid

import numpy as np

# 与 scene_geometry 保持同一套可见性口径，否则两边的 geom id 会对不上。
from server.runtime.adapters.scene_geometry import VISUAL_GROUPS

ASSET_SCHEMA = "prism.visual_assets.v1"
ASSET_DIR_NAME = "visual_assets"
MANIFEST_NAME = "manifest.json"
# 内容哈希命名：相同内容的资产自然去重，且已存在的文件永不重写
MESH_NAME = "mesh_{digest}.json"
TEXTURE_NAME = "tex_{digest}.png"
# 只允许受控文件名：编号 + 十六进制摘要 + 固定后缀
SAFE_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,79}$")

MESH_BYTE_LIMIT = 8 * 1024 * 1024
TEXTURE_BYTE_LIMIT = 12 * 1024 * 1024
ASSET_BYTE_BUDGET = 128 * 1024 * 1024
MAX_LISTED = 24

POSITION_DECIMALS = 6
NORMAL_DECIMALS = 5
UV_DECIMALS = 5
POSE_DECIMALS = 5

MESH_TYPE = 7
TEXTURE_2D, TEXTURE_CUBE, TEXTURE_SKYBOX = 0, 1, 2
GEOM_KINDS = {0: "plane", 1: "hfield", 2: "sphere", 3: "capsule", 4: "ellipsoid",
              5: "cylinder", 6: "box", 7: "mesh", 8: "sdf", 105: "flex", 106: "skin"}
ROLE_NAMES = ("user", "rgb", "occlusion", "roughness", "metallic", "normal",
              "opacity", "emissive", "rgba", "orm")
BASE_COLOUR_ROLES = ("rgba", "rgb")

# 实测：cube 纹理在 mjModel 中把 6 个面纵向叠成 h == 6*w，格子顺序由 XML
# gridlayout 决定。对 gridlayout="LRFBUD"（robocasa 生成纹理用的就是它），
# 编译后的 6 个 block 依次是 R L U D F B。这里只作对照，不用于改写像素。
CUBE_BLOCK_ORDER_LRFBUD = ("R", "L", "U", "D", "F", "B")
# 实测：box 几何 + cube 纹理时，MuJoCo 按几何局部方向选 cube 面。
BOX_CUBE_FACE_MAP = {"+x": "L", "-x": "R", "+y": "D", "-y": "U", "+z": "B", "-z": "F"}


def asset_directory(frame_dir):
    """Cache directory for a session: `<session>/visual_assets`.

    `frame_dir` is `<session>/frames`, so the assets live beside the frames and
    never inside them (the frame directory is scanned for `frame_*.jpg`).
    """
    return Path(frame_dir).parent / ASSET_DIR_NAME


def relative_url(name, base=ASSET_DIR_NAME):
    """Short relative URL for one exported file. Only the file name is used."""
    if not isinstance(name, str) or not SAFE_NAME.fullmatch(name):
        raise ValueError(f"资产文件名不受控：{name!r}")
    return f"{base}/{name}" if base else name


# --------------------------------------------------------------------- reading
def _int_at(model, attribute, index, default=-1):
    array = getattr(model, attribute, None)
    if array is None:
        return default
    try:
        return int(np.asarray(array).reshape(-1)[int(index)])
    except (IndexError, TypeError, ValueError):
        return default


def _id_name(model, kind, index):
    """Object name via mj_id2name. None when unavailable -- never invented."""
    if index is None or int(index) < 0:
        return None
    try:
        import mujoco

        return mujoco.mj_id2name(model, kind, int(index))
    except Exception:
        return None


def _geom_alpha(model, geom_id, material_id):
    """Effective alpha, same priority as scene_geometry: material, then geom."""
    if material_id >= 0:
        rgba = getattr(model, "mat_rgba", None)
        if rgba is not None:
            try:
                values = np.asarray(rgba)[material_id].reshape(-1)
                if values.size >= 4:
                    return float(values[3])
            except (IndexError, TypeError, ValueError):
                pass
    try:
        values = np.asarray(model.geom_rgba)[geom_id].reshape(-1)
        return float(values[3]) if values.size >= 4 else 1.0
    except (AttributeError, IndexError, TypeError, ValueError):
        return 1.0


def _material_textures(model, material_id):
    """{role_name: texture_id} for the roles this material actually binds."""
    table = getattr(model, "mat_texid", None)
    if table is None or material_id < 0:
        return {}
    try:
        row = np.asarray(table)[material_id].reshape(-1)
    except (IndexError, TypeError, ValueError):
        return {}
    bound = {}
    for role, texture_id in enumerate(row):
        if int(texture_id) >= 0:
            bound[ROLE_NAMES[role] if role < len(ROLE_NAMES) else f"role_{role}"] = int(texture_id)
    return bound


def _base_colour_texture(bound):
    for role in BASE_COLOUR_ROLES:
        if role in bound:
            return bound[role]
    return None


def _visual_geoms(model):
    """Visible geoms with their material/mesh/texture links.

    Filter mirrors `scene_geometry._entries(visual_only=True)`: group 0 is a
    robosuite collision proxy and fully transparent geoms are not drawn.
    """
    rows = []
    for geom_id in range(int(model.ngeom)):
        group = _int_at(model, "geom_group", geom_id, 0)
        if group not in VISUAL_GROUPS:
            continue
        type_id = _int_at(model, "geom_type", geom_id, -1)
        kind = GEOM_KINDS.get(type_id)
        if kind is None:
            continue
        material_id = _int_at(model, "geom_matid", geom_id, -1)
        if _geom_alpha(model, geom_id, material_id) <= 0.01:
            continue
        mesh_id = _int_at(model, "geom_dataid", geom_id) if type_id == MESH_TYPE else -1
        rows.append({
            "geom": geom_id,
            "kind": kind,
            "type_id": type_id,
            "group": group,
            "mesh": mesh_id,
            "material": material_id,
        })
    return rows


# ----------------------------------------------------------------- mesh payload
def _face_indices(model, attribute, face_adr, face_num, limit):
    """Validated (face_num, 3) local index block, or None when unusable."""
    array = getattr(model, attribute, None)
    if array is None:
        return None
    try:
        block = np.asarray(array, dtype=np.int64).reshape(-1, 3)[face_adr:face_adr + face_num]
    except (IndexError, TypeError, ValueError):
        return None
    if block.shape != (face_num, 3) or block.size == 0:
        return None
    if block.min() < -1 or (limit >= 0 and block.max() >= limit):
        return None
    return block


def _mesh_payload(model, mesh_id):
    """Compact JSON triangle payload for one compiled mesh.

    Returns `(payload_bytes, stats)` or `(None, reason_code)`. Corners are
    deduplicated on `(vertex, normal, texcoord)` so a shared vertex is stored
    once; no triangle or vertex is dropped.
    """
    vert_adr = _int_at(model, "mesh_vertadr", mesh_id, 0)
    vert_num = _int_at(model, "mesh_vertnum", mesh_id, 0)
    face_adr = _int_at(model, "mesh_faceadr", mesh_id, 0)
    face_num = _int_at(model, "mesh_facenum", mesh_id, 0)
    if vert_num <= 0 or face_num <= 0:
        return None, "mesh_empty"
    try:
        vertices = np.asarray(model.mesh_vert, dtype=np.float64).reshape(-1, 3)
        vertices = vertices[vert_adr:vert_adr + vert_num]
        faces = np.asarray(model.mesh_face, dtype=np.int64).reshape(-1, 3)[face_adr:face_adr + face_num]
    except (AttributeError, IndexError, TypeError, ValueError):
        return None, "mesh_vertex_data_missing"
    if vertices.shape != (vert_num, 3) or faces.shape != (face_num, 3):
        return None, "mesh_vertex_data_missing"
    if faces.min() < 0 or faces.max() >= vert_num:
        # 局部顶点索引越界：宁可报告，也不按猜测修正
        return None, "mesh_face_index_out_of_range"

    normal_adr = _int_at(model, "mesh_normaladr", mesh_id, -1)
    normal_num = _int_at(model, "mesh_normalnum", mesh_id, 0)
    normals = None
    if normal_adr >= 0 and normal_num > 0:
        try:
            normals = np.asarray(model.mesh_normal, dtype=np.float64).reshape(-1, 3)[
                normal_adr:normal_adr + normal_num]
            if normals.shape != (normal_num, 3):
                normals = None
        except (AttributeError, IndexError, TypeError, ValueError):
            normals = None
    face_normal = _face_indices(model, "mesh_facenormal", face_adr, face_num, normal_num) if normals is not None else None

    uv_adr = _int_at(model, "mesh_texcoordadr", mesh_id, -1)
    uv_num = _int_at(model, "mesh_texcoordnum", mesh_id, 0)
    uvs = None
    if uv_adr >= 0 and uv_num > 0:
        try:
            uvs = np.asarray(model.mesh_texcoord, dtype=np.float64).reshape(-1, 2)[uv_adr:uv_adr + uv_num]
            if uvs.shape != (uv_num, 2):
                uvs = None
        except (AttributeError, IndexError, TypeError, ValueError):
            uvs = None
    face_uv = _face_indices(model, "mesh_facetexcoord", face_adr, face_num, uv_num) if uvs is not None else None
    if uvs is not None and face_uv is None and uv_num == vert_num:
        # 内联 mesh（XML 里直接给 texcoord）没有 f/vt 索引，按顶点一一对应
        face_uv = faces

    corner_vertex = faces.reshape(-1)
    # 法线槽：优先用编译法线索引；没有编译法线时退化为导出三角形自身的
    # 几何法线（由顶点算出，不是凭空发明的方向），用 -1 - face 区分。
    if face_normal is not None and np.all(face_normal >= 0):
        corner_normal = face_normal.reshape(-1)
        normal_values = normals[corner_normal]
        normal_source = "mj_mesh_normal"
    else:
        corner_normal = -(np.repeat(np.arange(face_num, dtype=np.int64), 3) + 1)
        normal_values = _face_normals(vertices, faces)[np.repeat(np.arange(face_num), 3)]
        normal_source = "geometric_face_normal"
    corner_uv = face_uv.reshape(-1) if face_uv is not None else np.full(corner_vertex.shape, -1, dtype=np.int64)
    keys = np.stack([corner_vertex, corner_normal, corner_uv], axis=1)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)

    # 同一槽位的角点共享法线/UV，散点写入即可（重复值相同）
    normal_by_slot = np.empty((unique.shape[0], 3), dtype=np.float64)
    normal_by_slot[inverse] = normal_values
    payload = {
        "positions": _flat(vertices[unique[:, 0]], POSITION_DECIMALS),
        "indices": inverse.astype(np.int64).tolist(),
        "normals": _flat(normal_by_slot, NORMAL_DECIMALS),
    }
    stats = {
        "vertices": int(unique.shape[0]),
        "triangles": int(face_num),
        "corners": int(corner_vertex.shape[0]),
        "has_normals": True,
        "has_uvs": False,
        "uv_coverage": 0.0,
        "normal_source": normal_source,
    }
    if uvs is not None:
        selected_uv = unique[:, 2]
        if selected_uv.min() >= 0:
            uv_by_slot = np.empty((unique.shape[0], 2), dtype=np.float64)
            uv_by_slot[inverse] = uvs[corner_uv]
            payload["uvs"] = _flat(uv_by_slot, UV_DECIMALS)
            stats["has_uvs"] = True
            stats["uv_coverage"] = 1.0
        else:
            # 部分面缺 UV：缺失角点回落到 (0, 0)，覆盖率如实报告
            available = corner_uv >= 0
            uv_values = np.zeros((corner_uv.shape[0], 2), dtype=np.float64)
            uv_values[available] = uvs[corner_uv[available]]
            uv_by_slot = np.empty((unique.shape[0], 2), dtype=np.float64)
            uv_by_slot[inverse] = uv_values
            payload["uvs"] = _flat(uv_by_slot, UV_DECIMALS)
            stats["has_uvs"] = True
            stats["uv_coverage"] = float(np.count_nonzero(corner_uv >= 0) / max(1, corner_uv.size))
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    stats["uv_complete"] = bool(stats["has_uvs"] and abs(stats["uv_coverage"] - 1.0) < 1e-9)
    return encoded, stats


def _face_normals(vertices, faces):
    """Geometric normal per triangle, in the compiled geom frame."""
    tri = vertices[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(normal, axis=1)
    length[length == 0] = 1.0
    return normal / length[:, None]


def _flat(array, decimals):
    values = np.asarray(array, dtype=np.float64).reshape(-1)
    return [round(float(value), decimals) for value in values]


# -------------------------------------------------------------- texture payload
def _texture_raw(model, texture_id):
    """`(raw_bytes, width, height, channels)` or `(None, reason_code)`."""
    width = _int_at(model, "tex_width", texture_id, 0)
    height = _int_at(model, "tex_height", texture_id, 0)
    channels = _int_at(model, "tex_nchannel", texture_id, 0)
    address = _int_at(model, "tex_adr", texture_id, -1)
    if width <= 0 or height <= 0 or channels <= 0 or address < 0:
        return None, "texture_data_missing"
    count = width * height * channels
    try:
        data = np.asarray(model.tex_data).reshape(-1)[address:address + count]
    except (AttributeError, IndexError, TypeError, ValueError):
        return None, "texture_data_missing"
    if data.size != count:
        return None, "texture_data_truncated"
    return np.ascontiguousarray(data, dtype=np.uint8), width, height, channels


def _texture_png(raw, width, height, channels):
    """PNG bytes. Row 0 of the file is row 0 of `tex_data` (post-vflip)."""
    from PIL import Image

    pixels = raw.reshape(height, width, channels)
    if channels == 1:
        image = Image.fromarray(pixels[:, :, 0], "L")
    elif channels == 3:
        image = Image.fromarray(pixels, "RGB")
    elif channels == 4:
        image = Image.fromarray(pixels, "RGBA")
    else:
        return None, "texture_channels_unsupported"
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=6)
    return buffer.getvalue(), None


def _cube_layout(raw, width, height, channels):
    """How the six cube faces sit in the compiled image."""
    if height == width * 6:
        blocks = raw.reshape(6, width, width, channels)
        digests = [hashlib.sha256(np.ascontiguousarray(block).tobytes()).hexdigest()[:12] for block in blocks]
        uniform = len(set(digests)) == 1
        layout = {
            "faces": 6,
            "face_size": [width, width],
            "uniform": bool(uniform),
            "block_digests": digests,
        }
        if not uniform:
            layout["block_order_source"] = "muJoCo_3.3.1_gridlayout_LRFBUD"
            layout["block_letters"] = list(CUBE_BLOCK_ORDER_LRFBUD)
        return layout
    if height == width:
        return {"faces": 6, "face_size": [width, width], "uniform": True,
                "block_digests": [hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()[:12]]}
    return {"faces": None, "face_size": None, "uniform": None,
            "reason": "cube_texture_layout_unrecognized"}


def _texture_byte_priority(bound_roles):
    return 0 if any(role in bound_roles for role in BASE_COLOUR_ROLES) else 1


# ---------------------------------------------------------------------- writing
def _remove_quietly(path):
    try:
        path.unlink()
    except OSError:
        pass


def _write_atomic(path, payload):
    """Write once, then publish with os.replace. Never a half-written asset."""
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(temp, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        _remove_quietly(temp)


def _cache_file(directory, name, payload):
    """Content-hash naming makes the cache immutable: an existing file is reused."""
    if not SAFE_NAME.fullmatch(name):
        raise ValueError(f"资产文件名不受控：{name!r}")
    target = directory / name
    if target.is_file():
        return target, int(target.stat().st_size), False
    _write_atomic(target, payload)
    return target, int(target.stat().st_size), True


def _ensure_directory(directory):
    path = Path(directory)
    if path.exists() and not path.is_dir():
        raise ValueError(f"资产目录被同名文件占用：{path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


# ------------------------------------------------------------------- manifest
def export_visual_assets(model, directory, include_textures=True,
                         mesh_byte_limit=MESH_BYTE_LIMIT,
                         texture_byte_limit=TEXTURE_BYTE_LIMIT,
                         byte_budget=ASSET_BYTE_BUDGET):
    """Export meshes, textures and material bindings; return the manifest.

    `directory` is the cache directory (`asset_directory(frame_dir)`). Only
    generated `<kind>_<id>_<digest>.<ext>` names are written, so a caller can
    never make this read or overwrite an arbitrary path.

    The budget is enforced by excluding WHOLE assets and reporting the reason.
    A mesh is never sampled, decimated or trimmed to fit.
    """
    directory = _ensure_directory(directory)
    geoms = _visual_geoms(model)

    mesh_ids = sorted({row["mesh"] for row in geoms if row["mesh"] >= 0})
    material_ids = sorted({row["material"] for row in geoms if row["material"] >= 0})
    materials = {material_id: _material_record(model, material_id) for material_id in material_ids}

    texture_roles = {}
    for material_id, record in materials.items():
        for role, texture_id in record["textures"].items():
            texture_roles.setdefault(texture_id, set()).add(role)
    texture_ids = list(texture_roles)

    unsupported = []
    total_mesh_bytes = 0
    total_texture_bytes = 0
    excluded_for_budget = 0
    written_files = set()
    meshes = {}
    for mesh_id in mesh_ids:
        encoded, stats = _mesh_payload(model, mesh_id)
        if encoded is None:
            unsupported.append(_unsupported("mesh", mesh_id, stats))
            continue
        if len(encoded) > mesh_byte_limit:
            unsupported.append(_unsupported("mesh", mesh_id, "mesh_over_byte_limit",
                                            {"bytes": len(encoded), "limit": int(mesh_byte_limit)}))
            continue
        if total_mesh_bytes + len(encoded) > byte_budget:
            excluded_for_budget += 1
            unsupported.append(_unsupported("mesh", mesh_id, "asset_budget_exhausted",
                                            {"bytes": len(encoded), "used": total_mesh_bytes}))
            continue
        digest = hashlib.sha256(encoded).hexdigest()[:12]
        name = MESH_NAME.format(digest=digest)
        _path, size, fetched = _cache_file(directory, name, encoded)
        total_mesh_bytes += size
        written_files.add(name)
        meshes[mesh_id] = {
            "file": name, "format": "json", "bytes": size, "sha256": digest,
            "written": fetched,
            "frame": "mujoco_geom_local",
            "layout": {"positions": "flat xyz", "indices": "flat triangles",
                       "normals": "flat xyz", "uvs": "flat uv"},
            **stats,
        }

    textures = {}
    if include_textures:
        # 底色纹理先落地：超预算时先牺牲辅助贴图，也不动几何
        ordered = sorted(texture_ids, key=lambda tid: (_texture_byte_priority(texture_roles[tid]), tid))
        for texture_id in ordered:
            record, reason = _export_texture(model, directory, texture_id,
                                             texture_byte_limit,
                                             byte_budget - total_mesh_bytes - total_texture_bytes,
                                             texture_roles[texture_id])
            if record is None:
                if reason == "asset_budget_exhausted":
                    excluded_for_budget += 1
                unsupported.append(_unsupported("texture", texture_id, reason))
                continue
            total_texture_bytes += record["bytes"]
            written_files.add(record["file"])
            textures[texture_id] = record

    for row in geoms:
        row["texture"] = _base_colour_texture(materials.get(row["material"], {}).get("textures", {}))
        row["texture_support"] = _texture_support(model, row, materials.get(row["material"]))

    records = [_geom_record(model, row, materials.get(row["material"]), meshes, textures) for row in geoms]
    listed = [row for row in records if row["mesh_file"] or row["texture_file"]]
    # 同内容文件只落盘一份，所以磁盘占用按唯一文件名统计；引用字节数会重复计入
    distinct_bytes = sum((directory / name).stat().st_size for name in sorted(written_files))
    revisions = sorted([item["sha256"] for item in list(meshes.values()) + list(textures.values())])
    manifest = {
        "schema": ASSET_SCHEMA,
        "source": "live_mujoco_model",
        "mujoco_version": _mujoco_version(),
        "directory": directory.name,
        "url_base": ASSET_DIR_NAME,
        "asset_key": _asset_key(revisions),
        "counts": {
            "meshes": len(meshes), "textures": len(textures),
            "materials": len(materials), "geoms": len(records),
            "geoms_with_assets": len(listed),
            "mesh_candidates": len(mesh_ids), "texture_candidates": len(texture_ids),
            "files": len(written_files),
        },
        "budget": {
            "mesh_bytes": total_mesh_bytes, "texture_bytes": total_texture_bytes,
            "referenced_bytes": total_mesh_bytes + total_texture_bytes,
            "distinct_bytes": distinct_bytes, "distinct_files": len(written_files),
            "budget": int(byte_budget),
            "mesh_byte_limit": int(mesh_byte_limit), "texture_byte_limit": int(texture_byte_limit),
            # 导出结果在预算内，且没有因为预算丢掉资产
            "within_budget": (total_mesh_bytes + total_texture_bytes <= byte_budget
                              and excluded_for_budget == 0),
            "excluded_for_budget": excluded_for_budget,
            "policy": "exclude whole assets and report; never sample or decimate a mesh",
        },
        "precision": {"positions": POSITION_DECIMALS, "normals": NORMAL_DECIMALS,
                      "uvs": UV_DECIMALS, "geometry_pose": POSE_DECIMALS},
        "meshes": {str(key): value for key, value in meshes.items()},
        "textures": {str(key): value for key, value in textures.items()},
        "materials": {str(key): value for key, value in materials.items()},
        "geoms": listed,
        "unsupported": unsupported,
        "unsupported_count": len(unsupported),
        "provenance": {
            "vertices": "mj_model mesh_vert (compiled: asset scale applied, recentred to the geom frame)",
            "mesh_scale_applied": False,
            "indices": "mj_model mesh_face, mesh_facenormal, mesh_facetexcoord (local per mesh)",
            "normals": "mj_model mesh_normal (compiled geom frame)",
            "uvs": "mj_model mesh_texcoord, orientation matches tex_data rows",
            "textures": "mj_model tex_data, post-vflip, row 0 = top; standard flipY=true texture",
            "dedup": "identical content is written once and shared; references still list it per asset",
            "paths": "generated filenames only; no external path is read",
            "placement": "world = body_pose * geom_pos/geom_quat * exported_position; "
                         "ignore scene_geometry bbox_centre when a real mesh asset exists",
            "cube_texture_block_order_LRFBUD": list(CUBE_BLOCK_ORDER_LRFBUD),
            "box_cube_face_map": dict(BOX_CUBE_FACE_MAP),
        },
    }
    _write_manifest(directory, manifest)
    return manifest


def _export_texture(model, directory, texture_id, byte_limit, remaining, roles=()):
    raw_result = _texture_raw(model, texture_id)
    if raw_result[0] is None:
        return None, raw_result[1]
    raw, width, height, channels = raw_result
    digest = hashlib.sha256(raw.tobytes()).hexdigest()[:12]
    name = TEXTURE_NAME.format(tex=texture_id, digest=digest)
    target = directory / name
    cached = target.is_file()
    cached_size = int(target.stat().st_size) if cached else None
    if cached and cached_size is not None and cached_size > byte_limit:
        return None, "texture_over_byte_limit"
    if not cached:
        if width * height * channels > byte_limit * 4:
            # 原始数据就已经远超上限：不做有损压缩，直接报告
            return None, "texture_over_byte_limit"
    encoded, reason = (None, None) if cached else _texture_png(raw, width, height, channels)
    if not cached and encoded is None:
        return None, reason
    size = cached_size if cached else len(encoded)
    if size > byte_limit:
        return None, "texture_over_byte_limit"
    if size > remaining:
        return None, "asset_budget_exhausted"
    if not cached:
        target, size, _fetched = _cache_file(directory, name, encoded)
    record = {
        "file": name, "format": "png", "bytes": int(size), "sha256": digest,
        "written": bool(not cached),
        "width": int(width), "height": int(height), "channels": int(channels),
        "type": {TEXTURE_2D: "2d", TEXTURE_CUBE: "cube", TEXTURE_SKYBOX: "skybox"}.get(
            _int_at(model, "tex_type", texture_id, TEXTURE_2D), "unknown"),
        "name": _id_name(model, _obj_kind("mjOBJ_TEXTURE"), texture_id),
        "roles": sorted(roles),
    }
    if record["type"] in ("cube", "skybox"):
        record["cube"] = _cube_layout(raw, width, height, channels)
    return record, None


def _obj_kind(name):
    try:
        import mujoco

        return getattr(mujoco.mjtObj, name)
    except Exception:  # pragma: no cover - only when mujoco is absent
        return None


def _material_record(model, material_id):
    def scalar(attribute):
        try:
            value = float(np.asarray(getattr(model, attribute))[material_id])
            return round(value, 4)
        except (AttributeError, IndexError, TypeError, ValueError):
            return None

    def vector(attribute, length):
        try:
            values = np.asarray(getattr(model, attribute))[material_id].reshape(-1)[:length]
            return [round(float(value), 4) for value in values]
        except (AttributeError, IndexError, TypeError, ValueError):
            return None

    uniform = _int_at(model, "mat_texuniform", material_id, 0)
    return {
        "name": _id_name(model, _obj_kind("mjOBJ_MATERIAL"), material_id),
        "rgba": vector("mat_rgba", 4),
        "emission": scalar("mat_emission"),
        "specular": scalar("mat_specular"),
        "shininess": scalar("mat_shininess"),
        "reflectance": scalar("mat_reflectance"),
        "metallic": scalar("mat_metallic"),
        "roughness": scalar("mat_roughness"),
        "texrepeat": vector("mat_texrepeat", 2),
        "texuniform": bool(uniform),
        "textures": _material_textures(model, material_id),
    }


def _texture_support(model, row, material):
    """How the consumer can actually apply this geom's material."""
    if material is None or row["texture"] is None:
        return "solid_colour" if row["material"] >= 0 else "geom_rgba"
    type_id = _int_at(model, "tex_type", row["texture"], TEXTURE_2D)
    if type_id != TEXTURE_2D:
        return "box_cube_face_lookup" if row["kind"] == "box" else "cube_texture_requires_direction"
    if material["texuniform"]:
        return "planar_projection"
    if row["kind"] == "mesh":
        mesh_id = row["mesh"]
        uv_num = _int_at(model, "mesh_texcoordnum", mesh_id, 0)
        return "mesh_uv" if uv_num > 0 else "mesh_uv_absent"
    return "primitive_texture_unsupported"


def _geom_record(model, row, material, meshes, textures):
    mesh_id = row["mesh"]
    texture_id = row["texture"]
    mesh_entry = meshes.get(mesh_id)
    texture_entry = textures.get(texture_id) if texture_id is not None else None
    return {
        "geom": row["geom"],
        "body": _int_at(model, "geom_bodyid", row["geom"], -1),
        "body_name": _id_name(model, _obj_kind("mjOBJ_BODY"), _int_at(model, "geom_bodyid", row["geom"], -1)),
        "kind": row["kind"],
        "group": row["group"],
        "mesh": mesh_id if mesh_id >= 0 else None,
        "mesh_file": mesh_entry["file"] if mesh_entry else None,
        "material": row["material"] if row["material"] >= 0 else None,
        "texture": texture_id,
        "texture_file": texture_entry["file"] if texture_entry else None,
        "texture_support": row["texture_support"],
        "mesh_exported": bool(mesh_entry),
        "local_position": _pose(model, "geom_pos", row["geom"], 3),
        "local_quaternion": _pose(model, "geom_quat", row["geom"], 4),
    }


def _pose(model, attribute, index, length):
    try:
        values = np.asarray(getattr(model, attribute))[index].reshape(-1)[:length]
        return [round(float(value), POSE_DECIMALS) for value in values]
    except (AttributeError, IndexError, TypeError, ValueError):
        return None


def _unsupported(kind, asset_id, reason, detail=None):
    entry = {"kind": kind, "id": int(asset_id), "reason": reason}
    if detail:
        entry["detail"] = detail
    return entry


def _asset_key(revisions):
    if not revisions:
        return "empty"
    digest = hashlib.sha256("|".join(revisions).encode("utf-8")).hexdigest()[:16]
    return f"va{len(revisions)}_{digest}"


def _mujoco_version():
    try:
        import mujoco

        return getattr(mujoco, "__version__", None)
    except Exception:  # pragma: no cover
        return None


def _write_manifest(directory, manifest):
    payload = json.dumps(manifest, ensure_ascii=False, separators=(",", ":"),
                         sort_keys=False).encode("utf-8")
    _write_atomic(directory / MANIFEST_NAME, payload)
    manifest["manifest_bytes"] = len(payload)


def load_manifest(directory):
    """Read a previously written manifest. Raise on a non-asset directory."""
    path = Path(directory) / MANIFEST_NAME
    with open(path, "rb") as handle:
        manifest = json.loads(handle.read().decode("utf-8"))
    if manifest.get("schema") != ASSET_SCHEMA:
        raise ValueError(f"不是 {ASSET_SCHEMA} 清单：{path}")
    return manifest


def geom_references(manifest, base=None):
    """Compact `geom id -> short relative URLs` table for the scene payload.

    This is what a scene message carries instead of triangles: a few bytes per
    geom, so the 1,000,000 byte worker limit is never at risk.
    """
    base = manifest.get("url_base", ASSET_DIR_NAME) if base is None else base
    rows = {}
    for row in manifest.get("geoms", []):
        entry = {}
        if row.get("mesh_file"):
            entry["mesh"] = relative_url(row["mesh_file"], base)
        if row.get("texture_file"):
            entry["texture"] = relative_url(row["texture_file"], base)
        if not entry:
            continue
        entry["texture_support"] = row.get("texture_support")
        rows[str(row["geom"])] = entry
    return {"base": base, "asset_key": manifest.get("asset_key"), "geoms": rows}


def summarize(manifest, limit=MAX_LISTED):
    """Small payload-safe summary of an export, with reasons for every gap."""
    unsupported = manifest.get("unsupported", [])
    reasons = {}
    for item in unsupported:
        reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
    return {
        "schema": manifest.get("schema"),
        "asset_key": manifest.get("asset_key"),
        "url_base": manifest.get("url_base"),
        "counts": manifest.get("counts", {}),
        "budget": manifest.get("budget", {}),
        "within_budget": manifest.get("budget", {}).get("within_budget"),
        "unsupported_count": len(unsupported),
        "unsupported_reasons": reasons,
        "unsupported_sample": unsupported[:limit],
        "note": "真实 mesh/纹理已导出到会话缓存；场景只引用短相对 URL",
    }
