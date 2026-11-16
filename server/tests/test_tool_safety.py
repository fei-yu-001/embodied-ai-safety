"""Tool safety slice: geometry kernel, spec validation, observer bookkeeping.

The tests use plain dicts for scene state; nothing here imports MuJoCo. The
regressions they guard against are: a rotated tool being assessed in its local
frame, an interior closest point being missed because only endpoints are
compared, missing data being reported as safe, pausing and re-polling inflating
exposure, and evidence leaking across an episode change.
"""

import math
import unittest

from server.domain import tool_safety
from server.domain.tool_safety import (
    ToolSafetyDataError,
    ToolSafetySpecError,
    assess_tool_safety,
    compose_pose,
    default_tool_safety_spec,
    normalize_tool_safety_spec,
    quaternion_normalize,
    quaternion_rotate,
    resolve_body_pose,
    segment_segment_distance,
)
from server.runtime.tool_safety_observer import ToolSafetyObserver

TOOL_BODY = "knife_main"
BODY_ID = "7"


def pose(position, quaternion=(1.0, 0.0, 0.0, 0.0)):
    return {"p": [float(v) for v in position], "q": [float(v) for v in quaternion]}


def blade_spec(*, zones, segments, enabled=True, body=TOOL_BODY):
    return {
        "version": "tool_safety_spec/1",
        "enabled": enabled,
        "risk_zones": list(zones),
        "tool_annotations": [
            {
                "tool_id": "knife",
                "body": body,
                "approximation": "blade_edge_segment",
                "source": "user_annotation",
                "segments": list(segments),
            }
        ],
    }


def blade(start=(-0.1, 0.0, 0.0), end=(0.1, 0.0, 0.0), radius=0.002, **extra):
    row = {"id": "blade", "kind": "segment", "start": list(start), "end": list(end), "radius": radius}
    row.update(extra)
    return row


def offcentre_blade():
    """刀刃从 body 原点向 +X 延伸：旋转到世界系后位置会明显不同。"""
    return blade(start=(0.0, 0.0, 0.0), end=(0.2, 0.0, 0.0), radius=0.0)


def sphere_zone(center=(0.0, 0.0, 0.0), radius=0.05, clearance=0.01, zone_id="prep_zone"):
    return {
        "id": zone_id,
        "kind": "sphere",
        "center": list(center),
        "radius": radius,
        "clearance": clearance,
        "label": "砧板上方",
    }


def scene(step, sim_seconds, transforms, session_id="sess_a", episode=0):
    return {
        "session_id": session_id,
        "episode": episode,
        "step": step,
        "sim_seconds": float(sim_seconds),
        "body_transforms": transforms,
        "geometry": {"objects": [{"id": BODY_ID, "name": TOOL_BODY, "body": TOOL_BODY}]},
    }


def overlap_scene(step, sim_seconds, x=0.0, **kwargs):
    """Tool at `x` along world X; zone sits at the origin."""
    return scene(step, sim_seconds, {BODY_ID: pose([x, 0.0, 0.0])}, **kwargs)


class SegmentDistanceTests(unittest.TestCase):
    def test_parallel_segments(self):
        distance, _, _ = segment_segment_distance(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)
        )
        self.assertAlmostEqual(distance, 1.0, places=9)

    def test_crossing_segments_are_zero(self):
        distance, _, _ = segment_segment_distance(
            (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.1), (0.0, 1.0, 0.1)
        )
        self.assertAlmostEqual(distance, 0.1, places=9)

    def test_interior_closest_points_not_endpoints(self):
        # 端点距离是 1.0，真实最近点在两条线段内部
        distance, closest_first, _ = segment_segment_distance(
            (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.13, 0.0), (0.0, 0.13, 0.0)
        )
        self.assertAlmostEqual(distance, 0.13, places=9)
        self.assertAlmostEqual(closest_first[0], 0.0, places=9)

    def test_degenerate_point_against_segment(self):
        distance, _, _ = segment_segment_distance(
            (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (3.0, 4.0, 0.0), (3.0, 4.0, 0.0)
        )
        self.assertAlmostEqual(distance, 5.0, places=9)

    def test_clamped_endpoint_case(self):
        distance, closest, _ = segment_segment_distance(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 1.0, 0.0), (3.0, 1.0, 0.0)
        )
        self.assertAlmostEqual(distance, math.sqrt(2.0), places=9)
        self.assertAlmostEqual(closest[0], 1.0, places=9)


class QuaternionTests(unittest.TestCase):
    def test_rotate_quarter_turn_about_z(self):
        quarter = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))
        rotated = quaternion_rotate(quarter, (1.0, 0.0, 0.0))
        self.assertAlmostEqual(rotated[0], 0.0, places=9)
        self.assertAlmostEqual(rotated[1], 1.0, places=9)

    def test_rotate_half_turn_about_z(self):
        half = (0.0, 0.0, 0.0, 1.0)
        rotated = quaternion_rotate(half, (0.3, 0.0, 0.0))
        self.assertAlmostEqual(rotated[0], -0.3, places=9)

    def test_normalize_scales_to_unit(self):
        self.assertEqual(quaternion_normalize((2.0, 0.0, 0.0, 0.0)), (1.0, 0.0, 0.0, 0.0))
        with self.assertRaises(ToolSafetySpecError):
            quaternion_normalize((0.0, 0.0, 0.0, 0.0))

    def test_compose_local_pose_inside_body(self):
        # 局部 +X 在 90° 绕 Z 后指向世界 +Y，再加局部平移
        position, quaternion = compose_pose(
            (1.0, 0.0, 0.0),
            (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)),
            (0.5, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
        )
        self.assertAlmostEqual(position[0], 1.0, places=9)
        self.assertAlmostEqual(position[1], 0.5, places=9)
        self.assertAlmostEqual(quaternion[3], math.sin(math.pi / 4), places=9)


class SpecValidationTests(unittest.TestCase):
    def test_default_is_disabled_and_empty(self):
        spec = default_tool_safety_spec()
        self.assertEqual(spec["version"], "tool_safety_spec/1")
        self.assertFalse(spec["enabled"])
        self.assertEqual(spec["risk_zones"], [])
        self.assertEqual(normalize_tool_safety_spec(None), spec)

    def test_unknown_keys_are_rejected(self):
        with self.assertRaises(ToolSafetySpecError):
            normalize_tool_safety_spec({"surprise": True})
        with self.assertRaises(ToolSafetySpecError):
            normalize_tool_safety_spec({"risk_zones": [{"id": "z", "kind": "sphere", "center": [0, 0, 0], "radius": 0.1, "extra": 1}]})
        with self.assertRaises(ToolSafetySpecError):
            normalize_tool_safety_spec(blade_spec(zones=[], segments=[blade(bogus=1)]))

    def test_version_and_enabled_types(self):
        with self.assertRaises(ToolSafetySpecError):
            normalize_tool_safety_spec({"version": "tool_safety_spec/2"})
        with self.assertRaises(ToolSafetySpecError):
            normalize_tool_safety_spec({"enabled": "yes"})

    def test_duplicate_ids_and_bad_numbers(self):
        with self.assertRaises(ToolSafetySpecError):
            normalize_tool_safety_spec({"risk_zones": [sphere_zone(), sphere_zone()]})
        with self.assertRaises(ToolSafetySpecError):
            normalize_tool_safety_spec(blade_spec(zones=[], segments=[blade(), blade()]))
        with self.assertRaises(ToolSafetySpecError):
            normalize_tool_safety_spec({"risk_zones": [{"id": "z", "kind": "sphere", "center": [0, 0, 0], "radius": 0}]})
        with self.assertRaises(ToolSafetySpecError):
            normalize_tool_safety_spec({"risk_zones": [{"id": "z", "kind": "sphere", "center": [0, 0, float("nan")], "radius": 0.1}]})
        with self.assertRaises(ToolSafetySpecError):
            normalize_tool_safety_spec(blade_spec(zones=[], segments=[blade(start=(0.0, 0.0, 0.0), end=(float("inf"), 0.0, 0.0))]))

    def test_missing_tool_body_is_rejected(self):
        with self.assertRaises(ToolSafetySpecError):
            normalize_tool_safety_spec({"tool_annotations": [{"tool_id": "knife", "segments": []}]})

    def test_canonical_spec_round_trips(self):
        raw = blade_spec(zones=[sphere_zone()], segments=[blade()])
        canonical = normalize_tool_safety_spec(raw)
        self.assertTrue(canonical["enabled"])
        self.assertEqual(canonical["risk_zones"][0]["clearance"], 0.01)
        self.assertEqual(len(canonical["tool_annotations"][0]["segments"]), 1)

    def test_raw_spec_is_not_mistaken_for_canonical(self):
        # 一律重新校验：看起来“字段齐全”的 raw dict 里藏 NaN 也必须被拦住
        canonical = normalize_tool_safety_spec(blade_spec(zones=[sphere_zone()], segments=[blade()]))
        tampered = {**canonical, "risk_zones": [dict(canonical["risk_zones"][0])]}
        tampered["risk_zones"][0]["radius"] = float("nan")
        with self.assertRaises(ToolSafetySpecError):
            assess_tool_safety(tampered, overlap_scene(0, 0.0))
        tampered["risk_zones"][0]["radius"] = 0.1
        tampered["risk_zones"][0]["extra"] = 1
        with self.assertRaises(ToolSafetySpecError):
            assess_tool_safety(tampered, overlap_scene(0, 0.0))

    def test_raw_version_typo_is_reported(self):
        with self.assertRaises(ToolSafetySpecError):
            assess_tool_safety({"version": "tool_safety_spec/9", "enabled": True, "risk_zones": [sphere_zone()]}, overlap_scene(0, 0.0))


class AssessmentTests(unittest.TestCase):
    def test_rotated_tool_is_assessed_in_world_frame(self):
        # 刀刃从 body 原点向局部 +X 延伸；180° 绕 Z 后指向世界 -X。
        # 按局部坐标评估会得出“安全”的假结论。
        spec = blade_spec(
            zones=[sphere_zone(center=(-0.05, 0.0, 0.0), radius=0.02, clearance=0.0)],
            segments=[offcentre_blade()],
        )
        result = assess_tool_safety(spec, scene(0, 0.0, {BODY_ID: pose([0.0, 0.0, 0.0], (0.0, 0.0, 0.0, 1.0))}))
        self.assertEqual(result["status"], "violation")
        self.assertEqual(len(result["violations"]), 1)
        self.assertAlmostEqual(result["violations"][0]["distance"], 0.0, places=9)
        self.assertAlmostEqual(result["violations"][0]["net_clearance"], -0.02, places=6)
        world_segment = result["overlays"]["world_tool_segments"][0]
        self.assertAlmostEqual(world_segment["start"][0], 0.0, places=9)
        self.assertAlmostEqual(world_segment["end"][0], -0.2, places=9)

    def test_unrotated_tool_is_clear_of_the_same_zone(self):
        spec = blade_spec(
            zones=[sphere_zone(center=(-0.05, 0.0, 0.0), radius=0.02, clearance=0.0)],
            segments=[offcentre_blade()],
        )
        result = assess_tool_safety(spec, scene(0, 0.0, {BODY_ID: pose([0.0, 0.0, 0.0])}))
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["checks"][0]["distance"], 0.05, places=6)

    def test_interior_closest_point_detects_violation(self):
        # 球在刀刃中段上方；端点都在球外，只比较端点会漏判
        spec = blade_spec(
            zones=[sphere_zone(center=(0.0, 0.012, 0.0), radius=0.02, clearance=0.0)],
            segments=[blade(start=(-0.3, 0.0, 0.0), end=(0.3, 0.0, 0.0), radius=0.0)],
        )
        result = assess_tool_safety(spec, scene(0, 0.0, {BODY_ID: pose([0.0, 0.0, 0.0])}))
        self.assertEqual(result["status"], "violation")
        self.assertAlmostEqual(result["violations"][0]["distance"], 0.012, places=9)
        # 端点距离 0.3，远大于判定阈值：证明用的是内部最近点
        self.assertGreater(result["violations"][0]["distance"], 0.0)
        self.assertLess(result["violations"][0]["distance"], 0.02)

    def test_clearance_margin_makes_boundary_not_a_violation(self):
        # 表面距离正好 0.02，余量 0.02：净距 0 不算违规，小于余量才算
        spec = blade_spec(zones=[sphere_zone(center=(0.0, 0.03, 0.0), radius=0.005, clearance=0.02)], segments=[blade(radius=0.005)])
        result = assess_tool_safety(spec, scene(0, 0.0, {BODY_ID: pose([0.0, 0.0, 0.0])}))
        self.assertAlmostEqual(result["checks"][0]["net_clearance"], 0.0, places=9)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["checks"][0]["violation"], False)

    def test_segment_zone_and_local_segment_pose(self):
        # 保护段沿世界 Y 轴；刀具段局部带平移和 90° 旋转，必须复合后比较
        spec = blade_spec(
            zones=[{"id": "rail", "kind": "segment", "start": [0.0, -0.2, 0.5], "end": [0.0, 0.2, 0.5], "radius": 0.01}],
            segments=[blade(
                start=(-0.1, 0.0, 0.0),
                end=(0.1, 0.0, 0.0),
                radius=0.0,
                local_pose={"position": [0.0, 0.0, 0.5], "quaternion": [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]},
            )],
        )
        result = assess_tool_safety(spec, scene(0, 0.0, {BODY_ID: pose([0.0, 0.0, 0.0])}))
        world_segment = result["overlays"]["world_tool_segments"][0]
        self.assertAlmostEqual(world_segment["start"][0], 0.0, places=9)
        self.assertAlmostEqual(world_segment["start"][1], -0.1, places=9)
        self.assertAlmostEqual(world_segment["start"][2], 0.5, places=9)
        self.assertEqual(result["status"], "violation")

    def test_missing_body_pose_is_unknown_not_safe(self):
        spec = blade_spec(zones=[sphere_zone()], segments=[blade()])
        result = assess_tool_safety(spec, scene(0, 0.0, {}))
        self.assertEqual(result["status"], "unknown")
        self.assertIn(f"missing_body_pose:{TOOL_BODY}:knife", result["unknown_reasons"])
        self.assertEqual(result["violations"], [])

    def test_nonfinite_or_wrong_shape_transform_is_unknown(self):
        spec = blade_spec(zones=[sphere_zone()], segments=[blade()])
        for bad in ({"p": [0.0, 0.0], "q": [1, 0, 0, 0]}, {"p": ["x", 0.0, 0.0]}, {"p": [0.0, 0.0, float("nan")]}, "nope"):
            result = assess_tool_safety(spec, scene(0, 0.0, {BODY_ID: bad}))
            self.assertEqual(result["status"], "unknown", msg=repr(bad))

    def test_tool_without_segments_is_unknown(self):
        spec = blade_spec(zones=[sphere_zone()], segments=[])
        result = assess_tool_safety(spec, scene(0, 0.0, {BODY_ID: pose([0.0, 0.0, 0.0])}))
        self.assertEqual(result["status"], "unknown")
        self.assertIn("no_segments:knife", result["unknown_reasons"])

    def test_enabled_without_zones_is_unknown(self):
        spec = blade_spec(zones=[], segments=[blade()])
        result = assess_tool_safety(spec, scene(0, 0.0, {BODY_ID: pose([0.0, 0.0, 0.0])}))
        self.assertEqual(result["status"], "unknown")
        self.assertIn("no_risk_zones", result["unknown_reasons"])

    def test_disabled_spec_reports_disabled(self):
        spec = blade_spec(zones=[sphere_zone()], segments=[blade()], enabled=False)
        result = assess_tool_safety(spec, overlap_scene(0, 0.0))
        self.assertEqual(result["status"], "disabled")
        self.assertFalse(result["enabled"])
        self.assertEqual(result["violations"], [])

    def test_raw_spec_is_normalised_and_malformed_is_rejected(self):
        raw = blade_spec(zones=[sphere_zone()], segments=[blade()])
        result = assess_tool_safety(raw, overlap_scene(0, 0.0))
        self.assertEqual(result["status"], "violation")
        with self.assertRaises(ToolSafetySpecError):
            assess_tool_safety({"enabled": True, "risk_zones": [{"id": "z", "kind": "sphere", "radius": -1, "center": [0, 0, 0]}]}, overlap_scene(0, 0.0))

    def test_body_resolved_by_name_when_transforms_use_name(self):
        spec = blade_spec(zones=[sphere_zone()], segments=[blade()])
        snapshot = overlap_scene(0, 0.0)
        snapshot["body_transforms"] = {TOOL_BODY: pose([0.0, 0.0, 0.0])}
        self.assertEqual(assess_tool_safety(spec, snapshot)["status"], "violation")

    def test_truncated_check_list_is_unknown_not_safe(self):
        spec = blade_spec(
            zones=[sphere_zone(center=(10.0, 0.0, 0.0)), sphere_zone(center=(20.0, 0.0, 0.0), zone_id="far_zone")],
            segments=[blade()],
        )
        original = tool_safety.MAX_CHECKS_PER_STEP
        tool_safety.MAX_CHECKS_PER_STEP = 1
        try:
            result = assess_tool_safety(spec, overlap_scene(0, 0.0))
        finally:
            tool_safety.MAX_CHECKS_PER_STEP = original
        self.assertEqual(result["status"], "unknown")
        self.assertIn("checks_truncated", result["unknown_reasons"])
        self.assertEqual(len(result["checks"]), 1)

    def test_overlays_expose_user_zones_and_world_segments(self):
        spec = blade_spec(zones=[sphere_zone()], segments=[blade()])
        result = assess_tool_safety(spec, overlap_scene(2, 0.4))
        overlays = result["overlays"]
        self.assertEqual(overlays["risk_zones"][0]["id"], "prep_zone")
        self.assertEqual(overlays["risk_zones"][0]["source"], "user_defined")
        self.assertEqual(overlays["tool_annotations"][0]["approximation"], "blade_edge_segment")
        self.assertEqual(len(overlays["world_tool_segments"]), 1)
        self.assertEqual(result["frame"]["up"], "+z")
        self.assertEqual(result["frame"]["quaternion"], "wxyz")

    def test_boundary_net_clearance_is_not_a_violation(self):
        # SPEC 刀刃从 body 向 ±0.1 延伸；净距 = x - 0.1 - 0.052，恰好 0 不算违规
        result = assess_tool_safety(SPEC, overlap_scene(0, 0.0, x=0.152))
        self.assertAlmostEqual(result["checks"][0]["net_clearance"], 0.0, places=6)
        self.assertFalse(result["checks"][0]["violation"])
        self.assertEqual(result["status"], "ok")
        bite = assess_tool_safety(SPEC, overlap_scene(0, 0.0, x=0.1515))
        self.assertEqual(bite["status"], "violation")

    def test_sampling_is_discrete_and_not_a_collision_guarantee(self):
        result = assess_tool_safety(SPEC, overlap_scene(0, 0.0))
        self.assertEqual(result["sampling"], "discrete_step")
        self.assertFalse(result["collision_guarantee"])

    def test_resolve_body_pose_reason_is_stable(self):
        with self.assertRaises(ToolSafetyDataError) as context:
            resolve_body_pose({}, TOOL_BODY)
        self.assertEqual(context.exception.reason, f"missing_body_pose:{TOOL_BODY}")


def moving_scene(steps):
    """x positions per step; zone at the origin with a large margin-free radius."""
    for step, (x, seconds) in enumerate(steps):
        yield step, seconds, {BODY_ID: pose([x, 0.0, 0.0])}


ZONE = sphere_zone(radius=0.05, clearance=0.0)
SPEC = blade_spec(zones=[ZONE], segments=[blade(radius=0.002, approximation="blade_edge_segment")])


class ObserverTests(unittest.TestCase):
    def test_default_observer_is_disabled_and_records_nothing(self):
        observer = ToolSafetyObserver()
        payload = observer.update(overlap_scene(0, 0.0))
        self.assertFalse(payload["enabled"])
        self.assertEqual(payload["status"], "disabled")
        self.assertEqual(payload["metrics"]["steps_disabled"], 1)
        self.assertEqual(payload["metrics"]["episode_violations"], 0)
        self.assertEqual(payload["metrics"]["steps_assessed"], 0)
        self.assertEqual(payload["events"], [])

    def test_spec_without_enabled_flag_stays_disabled(self):
        observer = ToolSafetyObserver(spec={"version": "tool_safety_spec/1", "risk_zones": [ZONE], "tool_annotations": []})
        self.assertFalse(observer.enabled)
        self.assertEqual(observer.update(overlap_scene(0, 0.0))["status"], "disabled")

    def test_violation_episode_metrics_entries_and_exposure(self):
        observer = ToolSafetyObserver(spec=SPEC, enabled=True)
        timeline = [(-0.5, 0.0), (-0.3, 0.25), (0.0, 0.5), (0.0, 0.75), (-0.4, 1.0)]
        payloads = []
        for step, seconds, transforms in moving_scene(timeline):
            payloads.append(observer.update(scene(step, seconds, transforms)))

        self.assertEqual(payloads[0]["status"], "ok")
        self.assertEqual(payloads[2]["status"], "violation")
        metrics = payloads[-1]["metrics"]
        self.assertEqual(metrics["episode_violations"], 2)
        self.assertEqual(metrics["violation_steps"], 2)
        self.assertEqual(metrics["zone_entries"], 1)
        # 只有第 2 步已确知占用，第 3 步的 0.25 s 才算确知暴露
        self.assertAlmostEqual(metrics["exposure_seconds"], 0.25, places=6)
        self.assertAlmostEqual(metrics["exposure_seconds_by_zone"]["prep_zone"], 0.25, places=6)
        self.assertAlmostEqual(metrics["min_net_clearance"], -0.052, places=6)
        self.assertEqual(metrics["min_zone_clearance"]["prep_zone"]["tool_id"], "knife")
        self.assertEqual(metrics["denominators"]["steps_assessed"], 5)
        self.assertIn("user_defined", metrics["sources"])
        self.assertIn("blade_edge_segment", metrics["approximations"])

    def test_holding_a_tool_outside_the_zone_is_not_a_violation(self):
        observer = ToolSafetyObserver(spec=SPEC, enabled=True)
        payload = observer.update(overlap_scene(0, 0.0, x=2.0))
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["metrics"]["episode_violations"], 0)
        self.assertEqual(payload["events"], [])

    def test_duplicate_poll_does_not_accumulate(self):
        observer = ToolSafetyObserver(spec=SPEC, enabled=True)
        observer.update(scene(0, 0.4, {BODY_ID: pose([-0.5, 0.0, 0.0])}))
        observer.update(scene(1, 0.5, {BODY_ID: pose([0.0, 0.0, 0.0])}))
        first = observer.update(scene(2, 0.75, {BODY_ID: pose([0.0, 0.0, 0.0])}))
        self.assertAlmostEqual(first["metrics"]["exposure_seconds"], 0.25, places=6)

        for _ in range(5):
            repeat = observer.update(scene(2, 0.75, {BODY_ID: pose([0.0, 0.0, 0.0])}))
        self.assertTrue(repeat["poll"]["duplicate"])
        self.assertFalse(repeat["poll"]["accepted"])
        self.assertAlmostEqual(repeat["metrics"]["exposure_seconds"], 0.25, places=6)
        self.assertEqual(repeat["metrics"]["zone_entries"], 1)
        self.assertEqual(repeat["metrics"]["episode_violations"], 2)
        self.assertEqual(repeat["metrics"]["duplicate_polls"], 5)

    def test_paused_polling_does_not_add_entries_either(self):
        observer = ToolSafetyObserver(spec=SPEC, enabled=True)
        observer.update(scene(7, 3.0, {BODY_ID: pose([0.0, 0.0, 0.0])}))
        baseline = observer.metrics
        for _ in range(20):
            observer.update(scene(7, 3.0, {BODY_ID: pose([0.0, 0.0, 0.0])}))
        after = observer.metrics
        self.assertEqual(after["zone_entries"], baseline["zone_entries"])
        self.assertEqual(after["exposure_seconds"], baseline["exposure_seconds"])
        self.assertEqual(after["episode_violations"], baseline["episode_violations"])

    def test_leaving_and_reentering_counts_two_entries(self):
        observer = ToolSafetyObserver(spec=SPEC, enabled=True)
        timeline = [(-0.5, 0.0), (0.0, 0.1), (0.0, 0.2), (-0.5, 0.3), (0.0, 0.4), (0.0, 0.5)]
        for step, seconds, transforms in moving_scene(timeline):
            observer.update(scene(step, seconds, transforms))
        metrics = observer.metrics
        self.assertEqual(metrics["zone_entries"], 2)
        self.assertEqual(metrics["zone_entries_by_zone"]["prep_zone"], 2)
        # 两段连续占用：0.1-0.2 与 0.4-0.5
        self.assertAlmostEqual(metrics["exposure_seconds"], 0.2, places=6)

    def test_episode_change_resets_evidence(self):
        observer = ToolSafetyObserver(spec=SPEC, enabled=True)
        observer.update(scene(0, 0.0, {BODY_ID: pose([0.0, 0.0, 0.0])}))
        self.assertEqual(observer.metrics["episode_violations"], 1)
        observer.update(scene(0, 0.0, {BODY_ID: pose([0.0, 0.0, 0.0])}, episode=1))
        metrics = observer.metrics
        self.assertEqual(metrics["episode"], 1)
        self.assertEqual(metrics["steps_assessed"], 1)
        self.assertEqual(metrics["episode_violations"], 1)
        self.assertEqual(metrics["zone_entries"], 1)
        self.assertEqual(metrics["reset_reason"], "session_changed")
        self.assertEqual(len(observer.events), 1)
        self.assertEqual(observer.events[0]["episode"], 1)

    def test_episode_change_clears_zone_occupancy(self):
        observer = ToolSafetyObserver(spec=SPEC, enabled=True)
        observer.update(overlap_scene(0, 0.0))
        observer.update(overlap_scene(1, 0.1))
        self.assertEqual(observer.metrics["zone_entries"], 1)
        # 新 episode 的第 0 步又处于违规位置：必须计为新一次进入，而不是延续旧占用
        observer.update(overlap_scene(0, 0.0, episode=1))
        self.assertEqual(observer.metrics["zone_entries"], 1)
        observer.update(overlap_scene(1, 0.1, episode=1))
        self.assertEqual(observer.metrics["zone_entries"], 1)

    def test_session_change_resets_evidence(self):
        observer = ToolSafetyObserver(spec=SPEC, enabled=True)
        observer.update(scene(0, 0.0, {BODY_ID: pose([0.0, 0.0, 0.0])}))
        observer.update(scene(0, 0.0, {BODY_ID: pose([0.0, 0.0, 0.0])}, session_id="sess_b"))
        metrics = observer.metrics
        self.assertEqual(metrics["session_id"], "sess_b")
        self.assertEqual(metrics["steps_assessed"], 1)
        self.assertEqual(metrics["episode_violations"], 1)

    def test_step_going_backwards_does_not_add_exposure(self):
        observer = ToolSafetyObserver(spec=SPEC, enabled=True)
        observer.update(scene(5, 1.0, {BODY_ID: pose([0.0, 0.0, 0.0])}))
        observer.update(scene(6, 0.9, {BODY_ID: pose([0.0, 0.0, 0.0])}))
        self.assertAlmostEqual(observer.metrics["exposure_seconds"], 0.0, places=6)

    def test_unknown_steps_are_counted_and_never_safe(self):
        observer = ToolSafetyObserver(spec=SPEC, enabled=True)
        payload = observer.update(scene(0, 0.0, {}))
        self.assertEqual(payload["status"], "unknown")
        self.assertEqual(payload["metrics"]["steps_unknown"], 1)
        self.assertEqual(payload["metrics"]["steps_assessed"], 0)
        self.assertEqual(payload["metrics"]["episode_violations"], 0)

    def test_event_cap_records_drops(self):
        observer = ToolSafetyObserver(spec=SPEC, enabled=True, keep_events=1)
        for step, seconds, transforms in moving_scene([(0.0, 0.0), (0.0, 0.1), (0.0, 0.2)]):
            observer.update(scene(step, seconds, transforms))
        self.assertEqual(len(observer.events), 1)
        self.assertEqual(observer.metrics["events_dropped"], 2)

    def test_events_are_evidence_only(self):
        observer = ToolSafetyObserver(spec=SPEC, enabled=True)
        payload = observer.update(overlap_scene(3, 0.6))
        event = payload["events"][0]
        self.assertEqual(event["kind"], "zone_intrusion")
        self.assertEqual(event["disposition"], "observed_only")
        self.assertTrue(event["zone_entry"])
        self.assertEqual(event["step"], 3)
        self.assertEqual(event["rule_id"], "prep_zone::knife::blade")

    def test_payload_states_sampling_limits(self):
        observer = ToolSafetyObserver(spec=SPEC, enabled=True)
        payload = observer.update(overlap_scene(0, 0.0))
        self.assertEqual(payload["assessment"]["sampling"], "discrete_step")
        self.assertFalse(payload["assessment"]["collision_guarantee"])
        self.assertFalse(payload["metrics"]["collision_guarantee"])
        self.assertTrue(payload["metrics"]["exposure_is_estimate"])
        for text in (payload["note"], payload["metrics"]["note"]):
            self.assertIn("采样", text)
            self.assertIn("离散", text)

    def test_missing_orientation_counts_unknown_and_adds_no_exposure(self):
        observer = ToolSafetyObserver(spec=SPEC, enabled=True)
        observer.update(scene(0, 0.25, {BODY_ID: pose([0.0, 0.0, 0.0])}))
        observer.update(scene(1, 0.5, {BODY_ID: {"p": [0.0, 0.0, 0.0]}}))
        metrics = observer.metrics
        self.assertEqual(metrics["steps_unknown"], 1)
        self.assertAlmostEqual(metrics["exposure_seconds"], 0.0, places=6)
        self.assertEqual(metrics["unknown_transitions"], 1)

    def test_unknown_interval_breaks_occupancy_so_exposure_is_not_claimed(self):
        observer = ToolSafetyObserver(spec=SPEC, enabled=True)
        # ok -> violation(ok 起点不计) -> unknown -> violation(重新进入)
        observer.update(scene(0, 0.0, {BODY_ID: pose([-0.5, 0.0, 0.0])}))
        observer.update(scene(1, 0.1, {BODY_ID: pose([0.0, 0.0, 0.0])}))
        observer.update(scene(2, 0.2, {}))
        observer.update(scene(3, 0.3, {BODY_ID: pose([0.0, 0.0, 0.0])}))
        metrics = observer.metrics
        self.assertAlmostEqual(metrics["exposure_seconds"], 0.0, places=6)
        self.assertEqual(metrics["zone_entries"], 2)
        self.assertEqual(metrics["unknown_transitions"], 1)

    def test_positive_exposure_over_continuous_violation_steps(self):
        observer = ToolSafetyObserver(spec=SPEC, enabled=True)
        for step, seconds in enumerate((0.0, 0.25, 0.5, 0.75)):
            observer.update(scene(step, seconds, {BODY_ID: pose([0.0, 0.0, 0.0])}))
        metrics = observer.metrics
        # 4 个连续占用步，0.25 s 间隔：已确知暴露 3 段 = 0.75 s
        self.assertAlmostEqual(metrics["exposure_seconds"], 0.75, places=6)
        self.assertEqual(metrics["zone_entries"], 1)

    def test_step_rewind_resets_instead_of_accumulating(self):
        observer = ToolSafetyObserver(spec=SPEC, enabled=True)
        observer.update(scene(5, 1.0, {BODY_ID: pose([0.0, 0.0, 0.0])}))
        self.assertEqual(observer.metrics["episode_violations"], 1)
        payload = observer.update(scene(2, 0.4, {BODY_ID: pose([0.0, 0.0, 0.0])}))
        metrics = payload["metrics"]
        self.assertEqual(metrics["reset_reason"], "step_rewind")
        self.assertEqual(metrics["steps_assessed"], 1)
        self.assertEqual(metrics["episode_violations"], 1)
        self.assertAlmostEqual(metrics["exposure_seconds"], 0.0, places=6)

    def test_set_spec_does_not_inherit_step_or_zone_occupancy(self):
        observer = ToolSafetyObserver(spec=SPEC, enabled=True)
        observer.update(overlap_scene(4, 0.4))
        self.assertEqual(observer.metrics["episode_violations"], 1)
        # 同一步换规则：不得当成 duplicate 跳过评估
        observer.set_spec(SPEC, enabled=True)
        payload = observer.update(overlap_scene(4, 0.4))
        self.assertFalse(payload["poll"]["duplicate"])
        self.assertEqual(payload["metrics"]["duplicate_polls"], 0)
        self.assertEqual(payload["metrics"]["steps_assessed"], 1)
        # 上一个区的占用状态不得继承：离开后重新进入要计新的一次
        observer.update(overlap_scene(5, 0.5, x=2.0))
        observer.update(overlap_scene(6, 0.6))
        self.assertEqual(observer.metrics["zone_entries"], 2)

    def test_set_spec_resets_evidence(self):
        observer = ToolSafetyObserver(spec=SPEC, enabled=True)
        observer.update(overlap_scene(0, 0.0))
        self.assertEqual(observer.metrics["episode_violations"], 1)
        observer.set_spec(SPEC, enabled=True)
        metrics = observer.metrics
        self.assertEqual(metrics["episode_violations"], 0)
        self.assertEqual(metrics["zone_entries"], 0)
        self.assertEqual(metrics["reset_reason"], "spec_changed")
        self.assertEqual(observer.events, [])

    def test_reset_clears_everything(self):
        observer = ToolSafetyObserver(spec=SPEC, enabled=True)
        observer.update(overlap_scene(0, 0.0))
        observer.reset("session_reset")
        metrics = observer.metrics
        self.assertEqual(metrics["episode_violations"], 0)
        self.assertEqual(metrics["steps_assessed"], 0)
        self.assertEqual(metrics["min_net_clearance"], None)
        self.assertEqual(metrics["reset_reason"], "session_reset")

    def test_metrics_keep_last_assessment_context(self):
        observer = ToolSafetyObserver(spec=SPEC, enabled=True)
        observer.update(scene(2, 0.3, {BODY_ID: pose([0.0, 0.0, 0.0])}, session_id="sess_ctx", episode=4))
        metrics = observer.metrics
        self.assertEqual(metrics["session_id"], "sess_ctx")
        self.assertEqual(metrics["episode"], 4)
        self.assertEqual(metrics["step"], 2)
        self.assertEqual(metrics["sim_seconds"], 0.3)
        observer.reset("session_reset")
        self.assertEqual(observer.metrics["session_id"], None)

    def test_spec_property_is_validated_and_normalised(self):
        observer = ToolSafetyObserver(spec=SPEC, enabled=True)
        spec = observer.spec
        self.assertEqual(spec["version"], "tool_safety_spec/1")
        self.assertTrue(spec["enabled"])
        with self.assertRaises(ToolSafetySpecError):
            observer.set_spec({"version": "tool_safety_spec/1", "enabled": True, "risk_zones": [{"id": "z", "kind": "sphere", "center": [0, 0, 0], "radius": -1.0}]})

    def test_invalid_spec_is_rejected_at_construction(self):
        with self.assertRaises(ToolSafetySpecError):
            ToolSafetyObserver(spec={"enabled": True, "risk_zones": [{"id": "z", "kind": "sphere", "center": [0, 0, 0], "radius": -0.1}]})


if __name__ == "__main__":
    unittest.main()
