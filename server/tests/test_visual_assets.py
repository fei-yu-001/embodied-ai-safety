"""Real mesh and texture export from a live MuJoCo model.

Every contract here is measured against MuJoCo 3.3.1 behaviour, not assumed:

- compiled `mesh_vert` already carries the asset scale and is recentred onto the
  geom frame, so `geom_size` must not be applied again;
- `mesh_face` / `mesh_facenormal` / `mesh_facetexcoord` carry LOCAL indices into
  that mesh's own slice, so two meshes in one model never share an index space;
- `mesh_normal` is already in the compiled geom frame;
- `tex_data` rows are the image rows and the compiled UVs agree with them;
- a cube texture is one image with six stacked faces, and its compiled block
  order for `gridlayout="LRFBUD"` is R L U D F B.

The rest guards the operational contract: the byte budget excludes whole assets
rather than sampling them, assets are immutable and atomically written, only
controlled filenames are produced, and the scene payload stays tiny.

Assets are handed to MuJoCo as in-memory bytes (`assets=`), so these tests do
not depend on the working directory and the module never reads a file path.
"""

import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from server.runtime.adapters.scene_geometry import QUERY_BYTE_BUDGET
from server.runtime.adapters.visual_assets import (
    ASSET_SCHEMA,
    MANIFEST_NAME,
    SAFE_NAME,
    asset_directory,
    export_visual_assets,
    geom_references,
    load_manifest,
    relative_url,
    summarize,
)

CUBE_OBJ = "\n".join([
    "v 0 0 0", "v 1 0 0", "v 1 1 0", "v 0 1 0",
    "v 0 0 1", "v 1 0 1", "v 1 1 1", "v 0 1 1",
    "vt 0 0", "vt 1 0", "vt 1 1", "vt 0 1",
    "vn 0 0 1",
    "f 1/1/1 2/2/1 3/3/1", "f 1/1/1 3/3/1 4/4/1",
    "f 5/1/1 6/2/1 7/3/1", "f 5/1/1 7/3/1 8/4/1",
    "f 1/1/1 5/2/1 6/3/1", "f 1/1/1 6/3/1 2/4/1",
    "f 3/1/1 7/2/1 8/3/1", "f 3/1/1 8/3/1 4/4/1",
    "f 1/1/1 4/2/1 8/3/1", "f 1/1/1 8/3/1 5/4/1",
    "f 2/1/1 6/2/1 7/3/1", "f 2/1/1 7/3/1 3/4/1",
]) + "\n"

ASYMMETRIC_OBJ = "\n".join([
    "v 0 0 0", "v 3 0 0", "v 3 2 0", "v 0 2 0",
    "v 0 0 5", "v 3 0 5", "v 3 2 5", "v 0 2 5",
    "f 1 2 3", "f 1 3 4", "f 5 6 7", "f 5 7 8", "f 1 5 6", "f 1 6 2",
    "f 3 7 8", "f 3 8 4", "f 1 4 8", "f 1 8 5", "f 2 6 7", "f 2 7 3",
]) + "\n"
ASYMMETRIC_VERTICES = [(0, 0, 0), (3, 0, 0), (3, 2, 0), (0, 2, 0),
                       (0, 0, 5), (3, 0, 5), (3, 2, 5), (0, 2, 5)]

# MuJoCo compiles a cube texture into ONE image with six faces stacked as h == 6*w.
# A 6x1 grid of square cells is the simplest legal source image.
CUBE_CELL = 16
CUBE_TEXTURE_CELLS = ("L", "R", "F", "B", "U", "D")
# measured on MuJoCo 3.3.1 for gridlayout="LRFBUD"
CUBE_COMPILED_BLOCKS = ("R", "L", "U", "D", "F", "B")


class AssetWorkspace:
    """Temp directory for the exported cache plus the in-memory assets."""

    def __init__(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="prism-visual-assets-")
        self.root = Path(self._tmp.name)
        self.frames = self.root / "sess_test" / "frames"
        self.frames.mkdir(parents=True)
        self.assets = {"cube.obj": CUBE_OBJ.encode("utf-8")}

    @property
    def directory(self):
        return asset_directory(self.frames)

    def add_mesh(self, name, obj_text):
        self.assets[name] = obj_text.encode("utf-8")
        return name

    def add_png(self, name, pixels, mode="RGB"):
        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(np.asarray(pixels, dtype=np.uint8), mode).save(buffer, format="PNG")
        self.assets[name] = buffer.getvalue()
        return name

    def close(self):
        self._tmp.cleanup()


def build(workspace, xml):
    import mujoco

    return mujoco.MjModel.from_xml_string(xml, assets=dict(workspace.assets))


class _NoNormalModel:
    """Minimal mjModel stand-in with no compiled normal data.

    MuJoCo 3.3.1 always generates `mesh_normal`, so the geometric-normal fallback
    in the exporter cannot be reached from a real model. This stub exposes the
    same array contract with `mesh_normalnum = 0` so that branch is still tested.
    """

    def __init__(self):
        self.ngeom = 1
        self.nmesh = 1
        self.nmat = 0
        self.ntex = 0
        self.geom_type = np.array([7], dtype=np.int32)
        self.geom_group = np.array([1], dtype=np.int32)
        self.geom_matid = np.array([-1], dtype=np.int32)
        self.geom_dataid = np.array([0], dtype=np.int32)
        self.geom_rgba = np.array([[0.8, 0.2, 0.2, 1.0]], dtype=np.float32)
        self.geom_pos = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        self.geom_quat = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        self.geom_bodyid = np.array([1], dtype=np.int32)
        self.mesh_vertadr = np.array([0], dtype=np.int32)
        self.mesh_vertnum = np.array([8], dtype=np.int32)
        self.mesh_faceadr = np.array([0], dtype=np.int32)
        self.mesh_facenum = np.array([12], dtype=np.int32)
        self.mesh_normaladr = np.array([-1], dtype=np.int32)
        self.mesh_normalnum = np.array([0], dtype=np.int32)
        self.mesh_texcoordadr = np.array([-1], dtype=np.int32)
        self.mesh_texcoordnum = np.array([0], dtype=np.int32)
        self.mesh_normal = np.zeros((0, 3), dtype=np.float32)
        self.mesh_texcoord = np.zeros((0, 2), dtype=np.float32)
        self.tex_data = np.zeros(0, dtype=np.uint8)
        corners = [(-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5),
                   (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)]
        self.mesh_vert = np.array(corners, dtype=np.float32)
        faces = [(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7), (0, 4, 5), (0, 5, 1),
                 (2, 6, 7), (2, 7, 3), (0, 3, 7), (0, 7, 4), (1, 5, 6), (1, 6, 2)]
        self.mesh_face = np.array(faces, dtype=np.int32)


def cube_grid_pixels():
    """A 6x1 grid of cells, one per cube face, in the `LRFBUD` layout order.

    MuJoCo requires the PNG size to be an integer multiple of gridsize, so the
    cells stack vertically: rows L R F B U D, one column.
    """
    pixels = np.zeros((len(CUBE_TEXTURE_CELLS) * CUBE_CELL, CUBE_CELL, 3), np.uint8)
    for index in range(len(CUBE_TEXTURE_CELLS)):
        value = (index + 1) * 30
        pixels[index * CUBE_CELL:(index + 1) * CUBE_CELL] = (value, 0, 0)
    return pixels


def textured_model(workspace, texture_file=None, texture_type="2d", mesh="cube.obj",
                   material=True, **texture_attrs):
    """Cube geo with a texture. Geoms are visual (group 1), like the real kitchen."""
    if texture_file is None:
        n = 32
        pixels = np.zeros((n, n, 3), np.uint8)
        pixels[: n // 2] = (255, 0, 0)
        pixels[n // 2:] = (0, 0, 255)
        texture_file = workspace.add_png("t.png", pixels)
    attrs = " ".join(f'{key}="{value}"' for key, value in texture_attrs.items())
    material_xml = (f'<material name="mat" texture="tex" rgba="0.9 0.8 0.7 1" reflectance="0.2"/>'
                    if material else "")
    reference = ' material="mat"' if material else ""
    xml = f"""<mujoco model="m">
      <asset>
        <texture name="tex" type="{texture_type}" file="{texture_file}" {attrs}/>
        {material_xml}
        <mesh name="shape" file="{mesh}"/>
      </asset>
      <worldbody>
        <body name="bd" pos="1 2 3">
          <geom name="g" type="mesh" mesh="shape"{reference} group="1" pos="0.5 0.25 0.1"/>
          <geom name="collider" type="box" size=".1 .1 .1" group="0" rgba="0.5 0.1 0.1 1"/>
        </body>
      </worldbody>
    </mujoco>"""
    return build(workspace, xml)


class ExportBasicsTests(unittest.TestCase):
    def test_manifest_is_written_and_reloadable(self):
        workspace = AssetWorkspace()
        try:
            model = textured_model(workspace)
            manifest = export_visual_assets(model, workspace.directory)
            self.assertEqual(manifest["schema"], ASSET_SCHEMA)
            self.assertTrue((workspace.directory / MANIFEST_NAME).is_file())
            reloaded = load_manifest(workspace.directory)
            self.assertEqual(reloaded["asset_key"], manifest["asset_key"])
            self.assertEqual(reloaded["counts"], manifest["counts"])
            # 资产目录在 frames 旁边，不混进被扫描的 frames 目录
            self.assertEqual(workspace.directory.parent, workspace.frames.parent)
            self.assertEqual(manifest["mujoco_version"], "3.3.1")
        finally:
            workspace.close()

    def test_exported_files_are_controlled_names_only(self):
        workspace = AssetWorkspace()
        try:
            manifest = export_visual_assets(textured_model(workspace), workspace.directory)
            files = sorted(path.name for path in workspace.directory.iterdir())
            self.assertIn(MANIFEST_NAME, files)
            for name in files:
                self.assertTrue(SAFE_NAME.fullmatch(name), name)
                self.assertNotIn("..", name)
            for entry in manifest["meshes"].values():
                self.assertTrue(SAFE_NAME.fullmatch(entry["file"]))
            for entry in manifest["textures"].values():
                self.assertTrue(SAFE_NAME.fullmatch(entry["file"]))
        finally:
            workspace.close()

    def test_dotdot_url_is_rejected(self):
        with self.assertRaises(ValueError):
            relative_url("../../etc/passwd")
        with self.assertRaises(ValueError):
            relative_url("sub/dir.png")

    def test_assets_are_immutable_and_reused(self):
        workspace = AssetWorkspace()
        try:
            model = textured_model(workspace)
            first = export_visual_assets(model, workspace.directory)
            mesh_name = next(iter(first["meshes"].values()))["file"]
            before = (workspace.directory / mesh_name).read_bytes()
            mtime = (workspace.directory / mesh_name).stat().st_mtime_ns
            second = export_visual_assets(model, workspace.directory)
            self.assertEqual(first["asset_key"], second["asset_key"])
            entry = next(iter(second["meshes"].values()))
            self.assertFalse(entry["written"], "内容一致时必须复用而不是重写")
            self.assertEqual((workspace.directory / mesh_name).read_bytes(), before)
            self.assertEqual((workspace.directory / mesh_name).stat().st_mtime_ns, mtime)
            leftovers = [path.name for path in workspace.directory.iterdir()
                         if path.name.endswith(".tmp")]
            self.assertEqual(leftovers, [], "原子写不能留下临时文件")
        finally:
            workspace.close()

    def test_directory_that_is_a_file_fails_closed(self):
        workspace = AssetWorkspace()
        try:
            blocked = workspace.root / "blocked"
            blocked.write_text("not a directory", encoding="utf-8")
            with self.assertRaises(ValueError):
                export_visual_assets(textured_model(workspace), blocked)
        finally:
            workspace.close()


class MeshGeometryTests(unittest.TestCase):
    def test_triangles_normals_and_uvs_are_real(self):
        workspace = AssetWorkspace()
        try:
            model = textured_model(workspace)
            manifest = export_visual_assets(model, workspace.directory)
            mesh_id = int(next(iter(manifest["meshes"])))
            entry = manifest["meshes"][str(mesh_id)]
            payload = json.loads((workspace.directory / entry["file"]).read_text("utf-8"))
            self.assertEqual(entry["triangles"], 12)
            self.assertEqual(len(payload["indices"]), 36)
            self.assertEqual(len(payload["positions"]), entry["vertices"] * 3)
            self.assertEqual(len(payload["normals"]), entry["vertices"] * 3)
            self.assertEqual(len(payload["uvs"]), entry["vertices"] * 2)
            self.assertTrue(entry["has_normals"])
            self.assertTrue(entry["has_uvs"])
            self.assertTrue(entry["uv_complete"])
            self.assertEqual(max(payload["indices"]), entry["vertices"] - 1)
            # 顶点确实来自模型，而不是重采样
            vertices = np.asarray(model.mesh_vert).reshape(-1, 3)
            allowed = {tuple(np.round(vertices[index], 6)) for index in range(len(vertices))}
            for row in np.asarray(payload["positions"]).reshape(-1, 3):
                self.assertIn(tuple(np.round(row, 6)), allowed)
        finally:
            workspace.close()

    def test_compiled_vertices_already_carry_the_asset_scale(self):
        workspace = AssetWorkspace()
        try:
            workspace.add_mesh("scaled.obj", CUBE_OBJ)
            model = build(workspace,
                '<mujoco model="m"><asset>'
                '<mesh name="b" file="scaled.obj" scale="2 1 4"/>'
                '</asset><worldbody><body name="bd">'
                '<geom name="g" type="mesh" mesh="b" group="1" pos="0.5 0.25 0.1"/>'
                '</body></worldbody></mujoco>')
            manifest = export_visual_assets(model, workspace.directory, include_textures=False)
            entry = next(iter(manifest["meshes"].values()))
            payload = json.loads((workspace.directory / entry["file"]).read_text("utf-8"))
            points = np.asarray(payload["positions"]).reshape(-1, 3)
            half = (points.max(axis=0) - points.min(axis=0)) / 2.0
            # 编译时 MuJoCo 还会把 mesh 对齐到主轴，所以只比较半尺寸的多重集合
            np.testing.assert_allclose(sorted(half), sorted([1.0, 0.5, 2.0]), atol=1e-6)
            # 反向断言：把 geom_size 当 scale 再乘一遍会让某一边变成 8
            self.assertGreater(abs(half.sum() - 8.0), 1e-3, "半尺寸之和不能是二次乘法结果")
            self.assertAlmostEqual(float(half.sum()), 3.5, places=5)
            self.assertNotAlmostEqual(float(half.sum()), 8.0, places=3)
            # 顶点已重定心：包围盒中心在 geom 原点（导出的是去重后的角点，均值不适用）
            np.testing.assert_allclose((points.min(axis=0) + points.max(axis=0)) / 2.0,
                                       [0, 0, 0], atol=1e-5)
        finally:
            workspace.close()

    def test_geom_frame_reproduces_the_original_vertex(self):
        """body_pose * geom_pose * exported vertex == the XML vertex.

        A site carrying the same XML pos/quat gives an independent ground truth:
        the compiler folds the mesh recentring into `geom_pos` but leaves the
        site alone.
        """
        import mujoco

        workspace = AssetWorkspace()
        try:
            workspace.add_mesh("asym.obj", ASYMMETRIC_OBJ)
            model = build(workspace,
                '<mujoco model="m">'
                '<asset><mesh name="A" file="asym.obj" scale="2 3 1"/></asset>'
                '<worldbody><body name="bd" pos="0.3 -0.2 0.1" quat="0.92388 0 0 0.38268">'
                '<geom name="g1" type="mesh" mesh="A" group="1" pos="0.5 0.25 0.1" '
                'quat="0.968912 0 0.247404 0"/>'
                '<site name="s1" pos="0.5 0.25 0.1" quat="0.968912 0 0.247404 0"/>'
                '</body></worldbody></mujoco>')
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)

            manifest = export_visual_assets(model, workspace.directory, include_textures=False)
            record = manifest["geoms"][0]
            entry = manifest["meshes"][str(record["mesh"])]
            payload = json.loads((workspace.directory / entry["file"]).read_text("utf-8"))
            points = np.asarray(payload["positions"]).reshape(-1, 3)

            position = np.asarray(record["local_position"], dtype=float)
            quat = np.asarray(record["local_quaternion"], dtype=float)
            matrix = np.zeros(9)
            mujoco.mju_quat2Mat(matrix, quat)
            matrix = matrix.reshape(3, 3)
            body = int(record["body"])
            body_pos = np.asarray(data.xpos[body], dtype=float)
            body_mat = np.asarray(data.xmat[body], dtype=float).reshape(3, 3)

            site = model.site("s1").id
            site_pos = np.asarray(data.site_xpos[site], dtype=float)
            site_mat = np.asarray(data.site_xmat[site], dtype=float).reshape(3, 3)
            scale = np.asarray(model.mesh_scale[0], dtype=float)
            expected = [site_pos + site_mat @ (scale * np.asarray(vertex))
                        for vertex in ASYMMETRIC_VERTICES]

            # 导出的位置按 6 位小数取整，容差取 1e-4 m
            tolerance = 1e-4
            for point in points:
                world = body_pos + body_mat @ (position + matrix @ point)
                distances = [float(np.linalg.norm(target - world)) for target in expected]
                self.assertLess(min(distances), tolerance,
                                "geom 局部标架必须能复现原始 XML 顶点")
            self.assertEqual(record["local_position"],
                             [round(float(v), 5) for v in model.geom_pos[0]])
        finally:
            workspace.close()

    def test_per_mesh_local_indices_are_not_mixed_between_meshes(self):
        """Two meshes in one model: a global index would silently break the second."""
        workspace = AssetWorkspace()
        try:
            workspace.add_mesh("a.obj", CUBE_OBJ)
            workspace.add_mesh("b.obj", "\n".join([
                "v 0 0 0", "v 2 0 0", "v 2 2 0", "v 0 2 0",
                "v 0 0 2", "v 2 0 2", "v 2 2 2", "v 0 2 2",
                "vt 0 0", "vt 0.2 0", "vt 0.2 0.2", "vt 0 0.2",
                "f 1/1 2/2 3/3", "f 1/1 3/3 4/4", "f 5/4 6/1 7/2", "f 5/4 7/2 8/3",
                "f 1/2 5/1 6/3", "f 1/2 6/3 2/4", "f 3/1 7/3 8/2", "f 3/1 8/2 4/1",
                "f 1/3 4/4 8/1", "f 1/3 8/1 5/2", "f 2/1 6/2 7/4", "f 2/1 7/4 3/3",
            ]) + "\n")
            model = build(workspace,
                '<mujoco model="m"><asset>'
                '<mesh name="A" file="a.obj"/><mesh name="B" file="b.obj"/>'
                '</asset><worldbody><body name="bd">'
                '<geom name="gA" type="mesh" mesh="A" group="1"/>'
                '<geom name="gB" type="mesh" mesh="B" group="1" pos="4 0 0"/>'
                '</body></worldbody></mujoco>')
            manifest = export_visual_assets(model, workspace.directory, include_textures=False)
            self.assertEqual(len(manifest["meshes"]), 2)
            for entry in manifest["meshes"].values():
                payload = json.loads((workspace.directory / entry["file"]).read_text("utf-8"))
                self.assertEqual(len(payload["indices"]), entry["triangles"] * 3)
                self.assertLess(max(payload["indices"]), entry["vertices"])
            second = manifest["meshes"]["1"]
            payload = json.loads((workspace.directory / second["file"]).read_text("utf-8"))
            uvs = np.asarray(payload["uvs"]).reshape(-1, 2)
            self.assertLessEqual(float(abs(uvs - 0.2).min()), 1e-6,
                                 "第二个 mesh 的 UV 不在它自己的 0..0.2 范围内")
            self.assertGreater(float(uvs.max()), 0.0)
        finally:
            workspace.close()

    def test_mesh_normals_match_the_compiled_triangles(self):
        workspace = AssetWorkspace()
        try:
            workspace.add_mesh("asym.obj", ASYMMETRIC_OBJ)
            model = build(workspace,
                '<mujoco model="m"><asset><mesh name="A" file="asym.obj" scale="2 3 1"/></asset>'
                '<worldbody><body name="bd">'
                '<geom type="mesh" mesh="A" group="1"/></body></worldbody></mujoco>')
            manifest = export_visual_assets(model, workspace.directory, include_textures=False)
            entry = next(iter(manifest["meshes"].values()))
            payload = json.loads((workspace.directory / entry["file"]).read_text("utf-8"))
            points = np.asarray(payload["positions"]).reshape(-1, 3)
            normals = np.asarray(payload["normals"]).reshape(-1, 3)
            indices = np.asarray(payload["indices"]).reshape(-1, 3)
            self.assertEqual(entry["normal_source"], "mj_mesh_normal")
            # 编译法线在硬边处是逐顶点平均的，只能要求与所属面同侧（点积 >= 0），
            # 并且必须逐字来自模型，而不是重新计算的方向
            compiled = np.asarray(model.mesh_normal).reshape(-1, 3)
            imported = {tuple(np.round(row, 5)) for row in compiled}
            for row in normals:
                self.assertIn(tuple(np.round(row, 5)), imported,
                              "导出法线必须是模型里的编译法线")
            for triangle in indices:
                corners = points[triangle]
                geometric = np.cross(corners[1] - corners[0], corners[2] - corners[0])
                geometric /= np.linalg.norm(geometric)
                for corner in triangle:
                    self.assertGreaterEqual(float(normals[corner] @ geometric), -1e-6,
                                            "导出法线不得与编译三角形反向")
            np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-3)
        finally:
            workspace.close()

    def test_geometric_normals_are_used_when_the_mesh_has_none(self):
        """The fallback branch: a model whose mesh has no normal data.

        MuJoCo 3.3.1 always generates `mesh_normal` (even for an inline
        vertex/face mesh), so this branch cannot be reached with a real model and
        is driven by a stub that exposes the same array contract.
        """
        import numpy as _np  # noqa: F401  (kept for readability of the stub)

        workspace = AssetWorkspace()
        try:
            model = _NoNormalModel()
            manifest = export_visual_assets(model, workspace.directory, include_textures=False)
            entry = next(iter(manifest["meshes"].values()))
            self.assertTrue(entry["has_normals"])
            self.assertEqual(entry["normal_source"], "geometric_face_normal")
            payload = json.loads((workspace.directory / entry["file"]).read_text("utf-8"))
            points = np.asarray(payload["positions"]).reshape(-1, 3)
            normals = np.asarray(payload["normals"]).reshape(-1, 3)
            indices = np.asarray(payload["indices"]).reshape(-1, 3)
            np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-3)
            for triangle in indices:
                corners = points[triangle]
                geometric = np.cross(corners[1] - corners[0], corners[2] - corners[0])
                geometric /= np.linalg.norm(geometric)
                for corner in triangle:
                    self.assertGreater(float(normals[corner] @ geometric), 0.999,
                                       "无编译法线时必须用导出三角形的几何法线")
        finally:
            workspace.close()


class TextureTests(unittest.TestCase):
    def test_texture_png_keeps_rows_and_records_size(self):
        from PIL import Image

        workspace = AssetWorkspace()
        try:
            n = 16
            pixels = np.zeros((n, n, 3), np.uint8)
            pixels[:4] = (255, 0, 0)
            pixels[-4:] = (0, 0, 255)
            name = workspace.add_png("rows.png", pixels)
            model = textured_model(workspace, texture_file=name)
            manifest = export_visual_assets(model, workspace.directory)
            entry = next(iter(manifest["textures"].values()))
            self.assertEqual(entry["format"], "png")
            self.assertEqual((entry["width"], entry["height"], entry["channels"]), (n, n, 3))
            self.assertEqual(entry["type"], "2d")
            image = np.asarray(Image.open(workspace.directory / entry["file"]).convert("RGB"))
            self.assertEqual(image[0, 0].tolist(), [255, 0, 0])
            self.assertEqual(image[-1, 0].tolist(), [0, 0, 255])
            raw = np.asarray(model.tex_data).reshape(
                int(model.tex_height[0]), int(model.tex_width[0]), 3)
            np.testing.assert_array_equal(image, raw)
        finally:
            workspace.close()

    def test_uv_and_texel_orientation_agree(self):
        """Compiled UV (0, 1) is tex_data row 0, and the export keeps that row order."""
        workspace = AssetWorkspace()
        try:
            model = textured_model(workspace)
            manifest = export_visual_assets(model, workspace.directory)
            entry = next(iter(manifest["meshes"].values()))
            payload = json.loads((workspace.directory / entry["file"]).read_text("utf-8"))
            uvs = np.asarray(payload["uvs"]).reshape(-1, 2)
            self.assertAlmostEqual(float(uvs[:, 1].max()), 1.0, places=5)
            self.assertAlmostEqual(float(uvs[:, 1].min()), 0.0, places=5)
            compiled = np.asarray(model.mesh_texcoord).reshape(-1, 2)
            np.testing.assert_allclose(compiled[0], [0.0, 1.0], atol=1e-6)
            np.testing.assert_allclose(compiled[-1], [0.0, 0.0], atol=1e-6)
            # 顶点 0 的 uv 必须真的来自编译表，而不是重排过的
            np.testing.assert_allclose(uvs[0], compiled[0], atol=1e-5)
        finally:
            workspace.close()

    def test_cube_texture_is_reported_with_face_layout(self):
        from PIL import Image

        workspace = AssetWorkspace()
        try:
            name = workspace.add_png("cube.png", cube_grid_pixels())
            model = textured_model(workspace, texture_file=name, texture_type="cube",
                                   gridsize="6 1", gridlayout="LRFBUD")
            manifest = export_visual_assets(model, workspace.directory)
            entry = next(iter(manifest["textures"].values()))
            self.assertEqual(entry["type"], "cube")
            cube = entry["cube"]
            self.assertEqual(cube["faces"], 6)
            self.assertEqual(cube["face_size"], [CUBE_CELL, CUBE_CELL])
            self.assertFalse(cube["uniform"])
            self.assertEqual(len(set(cube["block_digests"])), 6)
            self.assertEqual(cube["block_letters"], list(CUBE_COMPILED_BLOCKS))
            # 整图作为一张 PNG 导出，尺寸不被改写
            image = np.asarray(Image.open(workspace.directory / entry["file"]).convert("RGB"))
            self.assertEqual(image.shape, (6 * CUBE_CELL, CUBE_CELL, 3))
            self.assertEqual((entry["width"], entry["height"]), (CUBE_CELL, 6 * CUBE_CELL))
        finally:
            workspace.close()

    def test_uniform_cube_texture_reports_uniform(self):
        workspace = AssetWorkspace()
        try:
            pixels = np.full((6 * CUBE_CELL, CUBE_CELL, 3), (7, 9, 11), np.uint8)
            name = workspace.add_png("flat.png", pixels)
            model = textured_model(workspace, texture_file=name, texture_type="cube",
                                   gridsize="6 1")
            manifest = export_visual_assets(model, workspace.directory)
            entry = next(iter(manifest["textures"].values()))
            self.assertTrue(entry["cube"]["uniform"])
            self.assertNotIn("block_letters", entry["cube"])
        finally:
            workspace.close()

    def test_skybox_texture_is_exported_and_typed(self):
        workspace = AssetWorkspace()
        try:
            model = build(workspace,
                '<mujoco model="m"><asset>'
                '<texture name="sky" type="skybox" builtin="gradient" '
                'rgb1=".2 .3 .4" rgb2=".5 .6 .7" width="16" height="16"/>'
                '<material name="mat" texture="sky"/>'
                '<mesh name="shape" file="cube.obj"/>'
                '</asset><worldbody><body name="bd">'
                '<geom name="g" type="mesh" mesh="shape" material="mat" group="1"/>'
                '</body></worldbody></mujoco>')
            manifest = export_visual_assets(model, workspace.directory)
            entry = next(iter(manifest["textures"].values()))
            self.assertEqual(entry["type"], "skybox")
            self.assertEqual(entry["cube"]["faces"], 6)
        finally:
            workspace.close()

    def test_material_binding_is_reported(self):
        workspace = AssetWorkspace()
        try:
            model = textured_model(workspace)
            manifest = export_visual_assets(model, workspace.directory)
            material = next(iter(manifest["materials"].values()))
            self.assertEqual(material["name"], "mat")
            self.assertEqual(material["rgba"], [0.9, 0.8, 0.7, 1.0])
            self.assertEqual(material["reflectance"], 0.2)
            self.assertIn("rgb", material["textures"])
            record = manifest["geoms"][0]
            self.assertEqual(record["texture"], material["textures"]["rgb"])
            self.assertEqual(record["texture_support"], "mesh_uv")
            self.assertEqual(record["kind"], "mesh")
            references = geom_references(manifest)
            self.assertIn("mesh", references["geoms"][str(record["geom"])])
        finally:
            workspace.close()

    def test_untextured_material_reports_solid_colour(self):
        workspace = AssetWorkspace()
        try:
            model = build(workspace,
                '<mujoco model="m"><asset>'
                '<material name="mat" rgba="0.2 0.4 0.6 1"/>'
                '<mesh name="shape" file="cube.obj"/>'
                '</asset><worldbody><body name="bd">'
                '<geom name="g" type="mesh" mesh="shape" material="mat" group="1"/>'
                '</body></worldbody></mujoco>')
            manifest = export_visual_assets(model, workspace.directory)
            self.assertEqual(manifest["counts"]["textures"], 0)
            self.assertEqual(manifest["geoms"][0]["texture_support"], "solid_colour")
            self.assertEqual(manifest["geoms"][0]["texture_file"], None)
        finally:
            workspace.close()


class VisibilityAndUnsupportedTests(unittest.TestCase):
    def test_collision_group_and_transparent_geoms_are_not_exported(self):
        workspace = AssetWorkspace()
        try:
            model = build(workspace,
                '<mujoco model="m"><asset>'
                '<mesh name="shape" file="cube.obj"/>'
                '</asset><worldbody><body name="bd">'
                '<geom name="visual" type="mesh" mesh="shape" group="1"/>'
                '<geom name="collider" type="mesh" mesh="shape" group="0" pos="2 0 0"/>'
                '<geom name="ghost" type="mesh" mesh="shape" group="1" pos="4 0 0" '
                'rgba="1 1 1 0"/>'
                '</body></worldbody></mujoco>')
            manifest = export_visual_assets(model, workspace.directory, include_textures=False)
            exported = {row["geom"] for row in manifest["geoms"]}
            self.assertEqual(exported, {int(model.geom("visual").id)})
            self.assertEqual(manifest["counts"]["meshes"], 1)
            self.assertEqual(manifest["geoms"][0]["group"], 1)
        finally:
            workspace.close()

    def test_repeated_geom_of_the_same_mesh_is_one_asset_two_placements(self):
        workspace = AssetWorkspace()
        try:
            model = build(workspace,
                '<mujoco model="m"><asset><mesh name="shape" file="cube.obj"/></asset>'
                '<worldbody><body name="bd">'
                '<geom name="a" type="mesh" mesh="shape" group="1"/>'
                '<geom name="b" type="mesh" mesh="shape" group="1" pos="2 0 0"/>'
                '</body></worldbody></mujoco>')
            manifest = export_visual_assets(model, workspace.directory, include_textures=False)
            self.assertEqual(manifest["counts"]["meshes"], 1)
            self.assertEqual(manifest["counts"]["geoms"], 2)
            files = sorted(manifest["meshes"][key]["file"] for key in manifest["meshes"])
            self.assertEqual(len(files), 1)
            positions = {row["geom"]: tuple(row["local_position"]) for row in manifest["geoms"]}
            self.assertEqual(len(positions), 2)
            self.assertEqual(len(set(positions.values())), 2, "两个实例必须带各自的局部位姿")
        finally:
            workspace.close()

    def test_oversized_texture_is_excluded_not_downscaled(self):
        workspace = AssetWorkspace()
        try:
            pixels = np.random.default_rng(0).integers(0, 256, (256, 256, 3), dtype=np.uint8)
            name = workspace.add_png("noise.png", pixels)
            model = textured_model(workspace, texture_file=name)
            manifest = export_visual_assets(model, workspace.directory, texture_byte_limit=1)
            self.assertEqual(manifest["counts"]["textures"], 0)
            self.assertIn("texture_over_byte_limit",
                          {item["reason"] for item in manifest["unsupported"]})
            leftovers = [path.name for path in workspace.directory.iterdir()
                         if path.suffix == ".png"]
            self.assertEqual(leftovers, [], "超限纹理不得写入任何（尤其是有损压缩过的）文件")
            self.assertIsNone(manifest["geoms"][0]["texture_file"])
        finally:
            workspace.close()

    def test_oversized_mesh_is_excluded_whole(self):
        workspace = AssetWorkspace()
        try:
            model = textured_model(workspace)
            manifest = export_visual_assets(model, workspace.directory, include_textures=False,
                                            mesh_byte_limit=1)
            self.assertEqual(manifest["counts"]["meshes"], 0)
            self.assertIn("mesh_over_byte_limit",
                          {item["reason"] for item in manifest["unsupported"]})
            leftovers = [path.name for path in workspace.directory.iterdir()
                         if path.suffix == ".json" and path.name != MANIFEST_NAME]
            self.assertEqual(leftovers, [], "被排除的 mesh 不得留下半个文件")
        finally:
            workspace.close()

    def test_budget_excludes_tail_assets_and_reports_them(self):
        workspace = AssetWorkspace()
        try:
            model = textured_model(workspace)
            full = export_visual_assets(model, workspace.directory, include_textures=False)
            self.assertEqual(full["counts"]["meshes"], 1)
            manifest = export_visual_assets(model, workspace.directory,
                                            byte_budget=1, include_textures=False)
            self.assertEqual(manifest["counts"]["meshes"], 0)
            self.assertIn("asset_budget_exhausted",
                          {item["reason"] for item in manifest["unsupported"]})
            self.assertFalse(manifest["budget"]["within_budget"])
        finally:
            workspace.close()

    def test_unsupported_reasons_are_machine_readable(self):
        workspace = AssetWorkspace()
        try:
            manifest = export_visual_assets(textured_model(workspace), workspace.directory,
                                            mesh_byte_limit=1, texture_byte_limit=1)
            self.assertTrue(manifest["unsupported"], "预算收紧后必须报告被排除的资产")
            summary = summarize(manifest)
            self.assertEqual(summary["unsupported_count"], len(manifest["unsupported"]))
            self.assertTrue(summary["unsupported_reasons"])
            for item in manifest["unsupported"]:
                self.assertIsInstance(item["reason"], str)
                self.assertTrue(item["reason"])
                self.assertIn(item["kind"], {"mesh", "texture"})
        finally:
            workspace.close()


class PayloadTests(unittest.TestCase):
    def test_geometry_references_are_tiny(self):
        workspace = AssetWorkspace()
        try:
            manifest = export_visual_assets(textured_model(workspace), workspace.directory)
            references = geom_references(manifest)
            encoded = json.dumps(references, separators=(",", ":")).encode("utf-8")
            self.assertLess(len(encoded), 2048)
            self.assertEqual(references["base"], "visual_assets")
            entry = references["geoms"][str(manifest["geoms"][0]["geom"])]
            self.assertTrue(entry["mesh"].startswith("visual_assets/"))
            self.assertNotIn("\\", entry["mesh"])
            self.assertNotIn("..", entry["mesh"])
            # 短相对 URL 必须能在会话目录里解析回来
            self.assertTrue((workspace.frames.parent / entry["mesh"]).is_file())
        finally:
            workspace.close()

    def test_asset_key_changes_when_content_changes(self):
        workspace = AssetWorkspace()
        try:
            model = textured_model(workspace)
            first = export_visual_assets(model, workspace.directory, include_textures=False)
            scaled = build(workspace,
                '<mujoco model="m"><asset><mesh name="shape" file="cube.obj" scale="2 1 1"/></asset>'
                '<worldbody><body name="bd">'
                '<geom name="g" type="mesh" mesh="shape" group="1"/></body></worldbody>'
                '</mujoco>')
            second = export_visual_assets(scaled, workspace.directory, include_textures=False)
            self.assertNotEqual(first["asset_key"], second["asset_key"])
        finally:
            workspace.close()

    def test_summary_stays_inside_the_worker_budget(self):
        workspace = AssetWorkspace()
        try:
            manifest = export_visual_assets(textured_model(workspace), workspace.directory)
            encoded = json.dumps(summarize(manifest), ensure_ascii=False,
                                 separators=(",", ":")).encode("utf-8")
            self.assertLess(len(encoded), QUERY_BYTE_BUDGET)
        finally:
            workspace.close()

    def test_manifest_does_not_leak_absolute_paths(self):
        workspace = AssetWorkspace()
        try:
            manifest = export_visual_assets(textured_model(workspace), workspace.directory)
            blob = json.dumps(manifest, ensure_ascii=False)
            self.assertNotIn(str(workspace.root), blob)
            self.assertEqual(manifest["directory"], "visual_assets")
            self.assertEqual(manifest["url_base"], "visual_assets")
            self.assertEqual(manifest["provenance"]["mesh_scale_applied"], False)
        finally:
            workspace.close()


if __name__ == "__main__":
    unittest.main()
