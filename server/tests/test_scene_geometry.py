"""Static geometry export: local geom pose, object geometry, budget safety.

Uses a fake MuJoCo model so the contracts are checked without a 30 s scene
load. The four regressions this guards against are: geoms collapsing to the
body origin, task-object geometry missing from the payload, a non-terminating
budget shrink, and unstable geometry keys.
"""

import json
import unittest

import numpy as np

from server.runtime.adapters.scene_geometry import (
    QUERY_BYTE_BUDGET,
    _mesh_bbox,
    build_geometry,
    build_transforms,
    fit_to_budget,
    geometry_key,
)

# 与 mjModel 一致的最小假模型：body 0 = world
BOX, SPHERE, MESH = 6, 2, 7


class FakeModel:
    """Minimal stand-in for mjModel: parents, geom ranges, local geom poses."""

    def __init__(self, geoms, geomadr, geomnum, parents, geom_pos=None, geom_quat=None,
                 rgba=None, names=None, groups=None, matids=None, mat_rgba=None):
        self.ngeom = len(geoms)
        self.nbody = len(parents)
        self.geom_type = np.array([g[0] for g in geoms], dtype=np.int32)
        self.geom_size = np.array([g[1] for g in geoms], dtype=np.float64)
        self.geom_pos = np.array(geom_pos if geom_pos is not None else [[0, 0, 0]] * self.ngeom, dtype=np.float64)
        self.geom_quat = np.array(geom_quat if geom_quat is not None else [[1, 0, 0, 0]] * self.ngeom, dtype=np.float64)
        self.geom_rgba = np.array(rgba if rgba is not None else [[0.5, 0.5, 0.5, 1.0]] * self.ngeom, dtype=np.float64)
        self.body_parentid = np.array(parents, dtype=np.int32)
        self.body_geomadr = np.array(geomadr, dtype=np.int32)
        self.body_geomnum = np.array(geomnum, dtype=np.int32)
        # 默认全部视觉组；碰撞组由具体用例显式指定
        self.geom_group = np.array(groups if groups is not None else [1] * self.ngeom, dtype=np.int32)
        self.geom_matid = np.array(matids if matids is not None else [-1] * self.ngeom, dtype=np.int32)
        self.nmat = len(mat_rgba) if mat_rgba else 0
        self.mat_rgba = np.array(mat_rgba if mat_rgba else [[1.0, 1.0, 1.0, 1.0]], dtype=np.float64)
        self._names = names or {}

    def body_id2name(self, index):
        return self._names.get(index)

    def body_name2id(self, name):
        for index, value in self._names.items():
            if value == name:
                return index
        raise KeyError(name)


def two_link_model():
    """world(0) + robot0_base(1) + robot0_link1(2) + cabinet(3) + obj(4)."""
    geoms = [
        (BOX, [0.05, 0.05, 0.05]),
        (SPHERE, [0.02, 0, 0]),
        (BOX, [0.3, 0.3, 0.4]),
        (BOX, [0.02, 0.02, 0.05]),
    ]
    geomadr = [0, 0, 1, 2, 3]
    geomnum = [0, 1, 1, 1, 1]
    parents = [0, 0, 1, 0, 0]
    names = {0: "world", 1: "robot0_base", 2: "robot0_link1", 3: "cabinet_main", 4: "obj_main"}
    geom_pos = [[0, 0, 0], [0.1, 0.0, 0.4], [0, 0, 0.5], [0, 0, 0]]
    geom_quat = [[1, 0, 0, 0], [0.7071, 0, 0, 0.7071], [1, 0, 0, 0], [1, 0, 0, 0]]
    return FakeModel(geoms, geomadr, geomnum, parents, geom_pos=geom_pos, geom_quat=geom_quat, names=names)


class FakeObject:
    def __init__(self, name, root_body):
        self.name = name
        self.root_body = root_body


class FakeTask:
    def __init__(self):
        self.objects = [FakeObject("obj", "obj_main")]


class GeomLocalPoseTests(unittest.TestCase):
    def test_geom_local_pose_is_exported_not_implicit(self):
        geometry = build_geometry(two_link_model(), FakeTask(), None)
        entries = {entry["geom"]: entry for entry in geometry["robot"]["geoms"]}
        self.assertIn(1, entries, "robot0_link1 的几何必须被导出")
        # 非零局部位移必须出现在 payload 里，前端才能与 body pose 复合
        self.assertEqual(entries[1]["local_position"], [0.1, 0.0, 0.4])
        self.assertEqual(entries[1]["local_quaternion"], [0.7071, 0.0, 0.0, 0.7071])
        self.assertNotEqual(entries[1]["local_quaternion"], [1.0, 0.0, 0.0, 0.0])
        # 零偏移的几何也必须带局部字段（而不是缺省由前端猜）
        self.assertEqual(entries[0]["local_position"], [0.0, 0.0, 0.0])
        for entry in entries.values():
            self.assertIn("local_position", entry)
            self.assertIn("local_quaternion", entry)

    def test_task_object_geometry_is_included_and_counted(self):
        geometry = build_geometry(two_link_model(), FakeTask(), None)
        self.assertEqual(len(geometry["objects"]), 1)
        obj = geometry["objects"][0]
        self.assertEqual(obj["name"], "obj")
        self.assertTrue(obj["geoms"], "任务物体必须带几何，不能只有 id/name")
        self.assertEqual(obj["geoms"][0]["representation"], "primitive")
        # 计数必须与实际发出的几何一致
        robot = len(geometry["robot"]["geoms"])
        objects = sum(len(item["geoms"]) for item in geometry["objects"])
        world = len(geometry["world"]["geoms"])
        self.assertEqual(geometry["provenance"]["geoms_exported"], robot + objects + world)

    def test_mesh_geoms_are_marked_as_approximation(self):
        model = two_link_model()
        model.geom_type = np.array([MESH, SPHERE, BOX, BOX], dtype=np.int32)
        geometry = build_geometry(model, FakeTask(), None)
        entry = next(e for e in geometry["robot"]["geoms"] if e["kind"] == "mesh")
        # 假模型没有 mesh 顶点，所以是占位盒；关键是必须标明不是基元、不是完整资产
        self.assertIn(entry["representation"], {"mesh_bbox", "mesh_placeholder"})
        self.assertNotEqual(entry["representation"], "primitive")
        self.assertIn("mesh", geometry["provenance"]["note"])


class MeshScaleTests(unittest.TestCase):
    """MuJoCo 3.3.1 实测：mesh_vert 已编译并重定心，不能再乘 geom_size。

    构造 XML 顶点 0..1 x / 0..2 y / 0..3 z + asset scale 2,1,4：
    期望 half = [1,1,6]；若误乘 geom_size 会得到 [1,1,36]。
    """

    def test_compiled_vertices_already_carry_the_asset_scale(self):
        import mujoco

        xml = (
            "<mujoco model='m'><asset>"
            "<mesh name='b' scale='2 1 4' vertex='0 0 0  1 0 0  1 2 0  0 2 0  0 0 3  1 0 3  1 2 3  0 2 3'/>"
            "</asset><worldbody><body name='bd' pos='1 2 3'>"
            "<geom name='g' type='mesh' mesh='b' pos='0.5 0.25 0.1' size='3 3 3'/>"
            "</body></worldbody></mujoco>"
        )
        model = mujoco.MjModel.from_xml_string(xml)
        geom_id = model.geom("g").id
        bbox = _mesh_bbox(model, geom_id)
        self.assertIsNotNone(bbox)
        self.assertEqual(bbox["size"], [1.0, 1.0, 6.0], "编译顶点已含 scale，不能再次相乘")
        self.assertEqual(bbox["centre"], [0.0, 0.0, 0.0], "MuJoCo 已把 mesh 重定心到 geom 原点")
        # 反向断言：错误实现会得到 36
        self.assertNotEqual(bbox["size"][2], 36.0)

    def test_compiled_geom_size_is_already_the_half_extent(self):
        """编译后 geom_size 就是半尺寸，把它当 scale 相乘是双重错误。

        XML 写 size='3 3 3'，编译后 geom_size 为 [1,1,6]（来自顶点与 asset
        scale），与包围盒 half 完全一致；相乘会得到 [1,1,36]。
        """
        import mujoco

        xml = (
            "<mujoco model='m'><asset>"
            "<mesh name='b' scale='2 1 4' vertex='0 0 0  1 0 0  1 2 0  0 2 0  0 0 3  1 0 3  1 2 3  0 2 3'/>"
            "</asset><worldbody><body name='bd' pos='1 2 3'>"
            "<geom name='g' type='mesh' mesh='b' pos='0.5 0.25 0.1' size='3 3 3'/>"
            "</body></worldbody></mujoco>"
        )
        model = mujoco.MjModel.from_xml_string(xml)
        geom_id = model.geom("g").id
        bbox = _mesh_bbox(model, geom_id)
        compiled = [float(v) for v in np.asarray(model.geom_size[geom_id]).reshape(-1)[:3]]
        self.assertEqual(compiled, [1.0, 1.0, 6.0], "编译后 geom_size 已是半尺寸")
        self.assertEqual(bbox["size"], compiled, "包围盒必须等于编译半尺寸，而不是它的平方")
        self.assertNotEqual(bbox["size"], [v * v for v in compiled])

    def test_primitive_geoms_report_zero_centre_offset(self):
        geometry = build_geometry(two_link_model(), FakeTask(), None)
        entry = next(e for e in geometry["robot"]["geoms"] if e["kind"] != "mesh")
        self.assertEqual(entry["bbox_centre"], [0.0, 0.0, 0.0])

    def test_key_covers_mesh_vertices(self):
        model = two_link_model()
        model.geom_dataid = np.array([0, -1, -1, -1], dtype=np.int32)
        model.mesh_vertadr = np.array([0], dtype=np.int64)
        model.mesh_vertnum = np.array([8], dtype=np.int64)
        first = geometry_key(model)
        model.mesh_vertnum = np.array([4], dtype=np.int64)
        self.assertNotEqual(first, geometry_key(model))


class VisualGroupTests(unittest.TestCase):
    """group 0 是 robosuite 的碰撞代理，不得当成外观。

    recolor_collision_geoms 会把 group 0 统一染成不透明红棕并移除材质，
    objects.py 又把原碰撞几何设成 OBJECT_COLLISION_COLOR。画出来就是
    「绿柜粉墙」这类错误颜色。
    """

    def test_collision_group_is_excluded_from_visual_export(self):
        model = two_link_model()
        # geom 0 -> group 0（碰撞，红棕）；其余保持视觉组
        model.geom_group = np.array([0, 1, 1, 1], dtype=np.int32)
        model.geom_rgba = np.array([[0.5, 0.0, 0.0, 1.0]] + [[0.5, 0.5, 0.5, 1.0]] * 3)
        geometry = build_geometry(model, FakeTask(), None)
        geoms = [e["geom"] for e in geometry["robot"]["geoms"]]
        self.assertNotIn(0, geoms, "碰撞代理不得出现在视觉几何里")
        self.assertEqual(geometry["world"]["skipped_collision"] + 1, 2)

    def test_fully_transparent_visual_geom_is_not_drawn(self):
        model = two_link_model()
        # group 1 但 alpha=0：本来就不可见，强画会变成绿色方块
        model.geom_rgba = np.array([[0.0, 1.0, 0.0, 0.0]] + [[0.5, 0.5, 0.5, 1.0]] * 3)
        geometry = build_geometry(model, FakeTask(), None)
        for entry in geometry["robot"]["geoms"]:
            self.assertGreater(entry["alpha"], 0.01, "alpha=0 的几何不能被绘制")
        self.assertEqual(geometry["world"]["skipped_transparent"] + 1, 2)

    def test_alpha_is_reported_truthfully_not_forced_opaque(self):
        model = two_link_model()
        model.geom_rgba = np.array([[0.5, 0.5, 0.5, 0.35]] + [[0.5, 0.5, 0.5, 1.0]] * 3)
        geometry = build_geometry(model, FakeTask(), None)
        entry = next(e for e in geometry["robot"]["geoms"] if e["geom"] == 0)
        self.assertAlmostEqual(entry["alpha"], 0.35, places=3, msg="不能把半透明 mesh 强行设为 alpha=1")

    def test_material_colour_wins_over_geom_rgba(self):
        model = two_link_model()
        # geom 0 绑材质 0（深木色），geom_rgba 是误导性的白色
        model.geom_matid = np.array([0, -1, -1, -1], dtype=np.int32)
        model.mat_rgba = np.array([[0.35, 0.22, 0.12, 1.0]], dtype=np.float64)
        model.geom_rgba = np.array([[1.0, 1.0, 1.0, 1.0]] * 4)
        geometry = build_geometry(model, FakeTask(), None)
        entry = next(e for e in geometry["robot"]["geoms"] if e["geom"] == 0)
        self.assertEqual(entry["colour_source"], "material")
        self.assertEqual(entry["colour"], [0.35, 0.22, 0.12])
        self.assertEqual(entry["colour_source"], "material")

    def test_geom_rgba_is_used_when_no_material(self):
        model = two_link_model()
        model.geom_rgba = np.array([[0.9, 0.9, 0.88, 1.0]] * 4)
        geometry = build_geometry(model, FakeTask(), None)
        entry = geometry["robot"]["geoms"][0]
        self.assertEqual(entry["colour_source"], "geom_rgba")
        self.assertEqual(entry["colour"], [0.9, 0.9, 0.88])

    def test_robot_visual_colour_is_not_the_collision_red(self):
        model = two_link_model()
        model.geom_group = np.array([0, 0, 1, 1], dtype=np.int32)
        model.geom_rgba = np.array([[0.5, 0.0, 0.0, 1.0]] * 2 + [[1.0, 1.0, 1.0, 1.0]] * 2)
        geometry = build_geometry(model, FakeTask(), None)
        colours = {tuple(e["colour"]) for e in geometry["robot"]["geoms"]}
        self.assertNotIn((0.5, 0.0, 0.0), colours, "碰撞红棕不得成为机器人外观")

    def test_texture_mean_carries_real_appearance(self):
        # 真实场景 mat_rgba 是白 tint，颜色在纹理里；只取 tint 会把橱柜画成白色
        model = two_link_model()
        model.geom_matid = np.array([0, -1, -1, -1], dtype=np.int32)
        model.mat_rgba = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float64)
        model.mat_texid = np.array([0], dtype=np.int32)
        model.ntex = 1
        # 2x1 的纯棕色纹理像素 (139, 69, 19)
        model.tex_adr = np.array([0], dtype=np.int32)
        model.tex_width = np.array([2], dtype=np.int32)
        model.tex_height = np.array([1], dtype=np.int32)
        model.tex_nchannel = np.array([3], dtype=np.int32)
        model.tex_data = np.array([139, 69, 19, 139, 69, 19], dtype=np.uint8)
        geometry = build_geometry(model, FakeTask(), None)
        entry = next(e for e in geometry["robot"]["geoms"] if e["geom"] == 0)
        self.assertEqual(entry["colour_source"], "material_texture")
        for got, want in zip(entry["colour"], [0.545, 0.271, 0.075]):
            self.assertAlmostEqual(got, want, places=2)
        self.assertNotEqual(entry["colour"], [1.0, 1.0, 1.0], "白色 tint 不得冒充外观")

    def test_group2_debug_markers_are_not_scene_appearance(self):
        # group 2 实际只有 eef_target_box/sphere 调试标记；当外观画会污染画面
        model = two_link_model()
        model.geom_group = np.array([2, 1, 1, 1], dtype=np.int32)
        model.geom_rgba = np.array([[0.9, 0.42, 0.44, 0.5]] + [[0.6, 0.6, 0.6, 1.0]] * 3)
        geometry = build_geometry(model, FakeTask(), None)
        self.assertNotIn(0, [e["geom"] for e in geometry["robot"]["geoms"]],
                         "group 2 调试标记不得作为外观导出")

    def test_payload_declares_visual_provenance(self):
        geometry = build_geometry(two_link_model(), FakeTask(), None)
        self.assertTrue(geometry["provenance"]["visual_only"])
        self.assertFalse(geometry["provenance"]["textures_exported"])
        self.assertTrue(geometry["world"]["visual_only"])
        counts = geometry["world"]["counts"]
        self.assertEqual(counts["total"], len(geometry["world"]["geoms"]))
        self.assertEqual(counts["primitives"] + counts["visual_meshes"], counts["total"])
        provenance = geometry["provenance"]
        self.assertEqual(
            provenance["geoms_exported"],
            provenance["geoms_robot"] + provenance["geoms_objects"]
            + provenance["geoms_world_primitives"] + provenance["geoms_world_visual_meshes"],
            "geoms_exported 必须等于各部分真实之和",
        )
        self.assertEqual(
            provenance["geoms_exported"] + provenance["geoms_skipped"],
            provenance["geoms_total"],
        )
        self.assertNotIn("primitives", geometry["world"], "world 只能有一份几何，避免重复序列化")
        self.assertIn("group 0", geometry["provenance"]["note"])


class GeometryKeyTests(unittest.TestCase):
    def test_key_changes_with_local_pose(self):
        first = geometry_key(two_link_model())
        model = two_link_model()
        model.geom_pos = np.array([[0, 0, 0], [0.2, 0.0, 0.4], [0, 0, 0.5], [0, 0, 0]])
        self.assertNotEqual(first, geometry_key(model))

    def test_key_changes_with_colour_and_names(self):
        first = geometry_key(two_link_model())
        coloured = two_link_model()
        coloured.geom_rgba = np.array([[0.1, 0.9, 0.2, 1.0]] * 4)
        self.assertNotEqual(first, geometry_key(coloured))

        renamed = two_link_model()
        renamed._names = {0: "world", 1: "robot0_base", 2: "robot0_link1", 3: "cabinet_other", 4: "obj_main"}
        self.assertNotEqual(first, geometry_key(renamed, FakeTask()))

    def test_key_is_stable_for_identical_models(self):
        self.assertEqual(geometry_key(two_link_model()), geometry_key(two_link_model()))


class BudgetTests(unittest.TestCase):
    def test_tiny_budget_terminates_and_strictly_reduces(self):
        geometry = build_geometry(two_link_model(), FakeTask(), None)
        result = fit_to_budget(geometry, budget=1)
        self.assertTrue(result["budget"]["degraded"])
        self.assertLessEqual(result["budget"]["bytes"], QUERY_BYTE_BUDGET)
        self.assertIn("geometry_omitted", result["budget"]["steps"])

    def test_single_robot_geom_with_tiny_budget_does_not_loop(self):
        model = two_link_model()
        # 只给 robot0_base 一个几何：截断时必须能严格递减到 0
        model.body_geomadr = np.array([0, 0, 0, 0, 0], dtype=np.int32)
        model.body_geomnum = np.array([0, 1, 0, 0, 0], dtype=np.int32)
        geometry = build_geometry(model, FakeTask(), None)
        self.assertEqual(len(geometry["robot"]["geoms"]), 1)
        result = fit_to_budget(geometry, budget=1)
        self.assertEqual(result["robot"]["geoms"], [])
        self.assertLessEqual(result["budget"]["bytes"], QUERY_BYTE_BUDGET)

    def test_normal_budget_keeps_geometry_intact(self):
        geometry = build_geometry(two_link_model(), FakeTask(), None)
        result = fit_to_budget(geometry)
        self.assertFalse(result["budget"]["degraded"])
        self.assertEqual(result["budget"]["steps"], [])
        self.assertTrue(result["robot"]["geoms"])

    def test_real_payload_stays_under_the_transport_limit(self):
        geometry = build_geometry(two_link_model(), FakeTask(), None)
        size = len(json.dumps(geometry, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        self.assertLess(size, 1_000_000, "单条 worker 消息上限是 1,000,000 字节")


class TransformTests(unittest.TestCase):
    def test_transforms_are_sent_separately_from_geometry(self):
        model = two_link_model()

        class Data:
            body_xpos = np.array([[0, 0, 0], [0.3, 0.0, 0.7], [0.4, 0.0, 1.0], [1.0, 1.0, 0.0], [0.2, 0.0, 1.0]],
                                 dtype=np.float64)
            body_xmat = np.tile(np.eye(3), (5, 1)).reshape(5, 9)

        transforms = build_transforms(model, Data())
        self.assertEqual(len(transforms), model.nbody)
        self.assertIn("p", transforms["1"])
        self.assertIn("q", transforms["1"])
        # 静态几何只带局部变换，不含世界位姿（世界位姿由 transforms 单独发送）
        geometry = build_geometry(model, FakeTask(), None)
        blob = json.dumps(geometry)
        self.assertNotIn('"world_position"', blob)
        self.assertNotIn('"world_quaternion"', blob)
        self.assertIn("local_position", blob)


if __name__ == "__main__":
    unittest.main()
