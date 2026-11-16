"""Regression tests for the evaluation sampling and paired-statistics fixes.

覆盖四项：
- run_group 在动作执行后记录 success/terminated/truncated，且输入帧侧的
  centroid_offset / defense_latency_ms 与被执行的动作保持对应；
- env 从创建起纳入 finally，早期异常也会 close；
- build_report 按 seed 与 clean 逐对计算，不拼接、不截断丢 seed；
- divergence 的 0.0 是有效值，与缺失 None 严格区分。

用 mock 环境真实调用 run_group，不启动 robocasa。
"""

import sys
import types
import unittest
from unittest import mock

import numpy as np

from server.domain.evaluation import (
    build_report,
    compute_gate,
    divergence,
    normalize_spec,
    summarize_group,
)
from server.runtime import evaluation_runner


CAMERA = "video.robot0_agentview_left"


class FakeEnv:
    """按脚本返回 success/terminated，并记录被调用次序。"""

    def __init__(self, success_at=None, terminate_at=None, fail_on_reset=False):
        self.success_at = success_at
        self.terminate_at = terminate_at
        self.fail_on_reset = fail_on_reset
        self.calls = 0
        self.closed = False
        self.seen_actions = []

    def reset(self, seed=None):
        if self.fail_on_reset:
            raise RuntimeError("模拟 reset 失败")
        self.calls = 0
        return {CAMERA: np.zeros((4, 4, 3), dtype=np.uint8)}, {"success": False}

    def step(self, action):
        self.calls += 1
        self.seen_actions.append(action)
        success = self.success_at is not None and self.calls >= self.success_at
        terminated = self.terminate_at is not None and self.calls >= self.terminate_at
        return (
            {CAMERA: np.zeros((4, 4, 3), dtype=np.uint8)},
            1.0 if success else 0.0,
            bool(terminated),
            False,
            {"success": success},
        )

    def close(self):
        self.closed = True


class FakePolicy:
    """每次调用返回递增的动作，并暴露 last_centroid / last_area。"""

    def __init__(self):
        self.calls = 0
        self.last_centroid = None
        self.last_area = 0.0

    def __call__(self, observation):
        self.calls += 1
        self.last_centroid = (self.calls, 2.0)
        self.last_area = float(self.calls) / 10.0
        return {"action.end_effector_position": np.full(7, float(self.calls))}


class FakeDefenseEnv:
    """模拟 DefenceWrapper：reset 与 step 都会写 last_latency_ms（与真实实现一致）。"""

    def __init__(self, env):
        self.env = env
        self.last_latency_ms = 10.0

    def reset(self, seed=None):
        observation, info = self.env.reset(seed=seed)
        self.last_latency_ms = 10.0
        return observation, info

    def step(self, action):
        result = self.env.step(action)
        self.last_latency_ms = 10.0 * (self.env.calls + 1)
        return result

    def close(self):
        self.env.close()


def fake_modules(env, policy):
    """构造要替换进 sys.modules 的假模块字典，不直接写入全局状态。"""
    gymnasium = types.ModuleType("gymnasium")
    gymnasium.make = lambda *a, **k: env
    robocasa = types.ModuleType("robocasa")

    experiments = types.ModuleType("experiments")
    attacks = types.ModuleType("experiments.attacks")
    observation = types.ModuleType("experiments.attacks.observation")
    digital_patch = types.ModuleType("experiments.attacks.digital_patch")
    adaptive = types.ModuleType("experiments.attacks.adaptive")
    defenses = types.ModuleType("experiments.defenses")
    filters = types.ModuleType("experiments.defenses.filters")
    sdk = types.ModuleType("experiments.sdk")
    registry = types.ModuleType("experiments.sdk.registry")

    # 让这些假模块被当成包，否则子模块 import 会直接失败。
    for module in (experiments, attacks, defenses, sdk):
        module.__path__ = []

    observation.DigitalPatchWrapper = lambda inner, patch, **k: inner
    digital_patch.make_noise_patch = lambda size, seed=1: np.zeros((size, size, 3), dtype=np.uint8)
    adaptive.JitteredPatchWrapper = lambda inner, patch, **k: inner
    adaptive.adaptive_patch = lambda seed: np.zeros((40, 40, 3), dtype=np.uint8)
    adaptive.adaptive_texture = lambda seed: np.zeros((40, 40, 3), dtype=np.uint8)
    filters.DefenseWrapper = lambda inner, defense, camera: FakeDefenseEnv(inner)
    filters.ImageDefense = lambda kind, kernel: object()
    registry.build_policy = lambda name, inner, options: policy

    return {
        "gymnasium": gymnasium,
        "robocasa": robocasa,
        "experiments": experiments,
        "experiments.attacks": attacks,
        "experiments.attacks.observation": observation,
        "experiments.attacks.digital_patch": digital_patch,
        "experiments.attacks.adaptive": adaptive,
        "experiments.defenses": defenses,
        "experiments.defenses.filters": filters,
        "experiments.sdk": sdk,
        "experiments.sdk.registry": registry,
    }


class MockedRunnerCase(unittest.TestCase):
    """需要注入假 experiments/gymnasium 的用例基类。

    sys.modules 用 patch.dict 注入并在 addCleanup 中严格恢复，
    包括本进程此前已有的真实模块；不允许把假模块泄漏给后续测试。
    """

    def install_mocks(self, env, policy=None):
        policy = policy or FakePolicy()
        # patch.dict 默认 clear=False：退出时会把这三个键恢复成原先的值，
        # 原先不存在的键会被删除，因此不会污染 test_robocasa_backend 的惰性 import。
        patcher = mock.patch.dict(sys.modules, fake_modules(env, policy))
        patcher.start()
        self.addCleanup(patcher.stop)
        return policy


SPEC = {
    "task": "robocasa/PickPlaceCounterToCabinet",
    "policy": "random",
    "attack": "none",
    "defense": "none",
    "seeds": [0],
    "steps": 5,
    "camera_key": CAMERA,
}
GROUP = {"name": "clean", "attack": "none", "defense": "none"}


def make_run(group, seed, series, success=False):
    rows = [
        {
            "step": index,
            "action": [value],
            "centroid_offset": [0.0, 0.0],
            "success": success,
            "defense_latency_ms": None,
        }
        for index, value in enumerate(series)
    ]
    return {
        "group": group,
        "seed": seed,
        "status": "ok",
        "success_observed": success,
        "rows": rows,
    }


class RunGroupSamplingTests(MockedRunnerCase):
    def test_success_on_the_terminating_step_is_recorded(self):
        env = FakeEnv(success_at=5, terminate_at=5)
        self.install_mocks(env)

        rows = evaluation_runner.run_group(SPEC, GROUP, 0)

        self.assertEqual(len(rows), 5)
        self.assertTrue(any(row["success"] for row in rows))
        self.assertTrue(rows[-1]["success"])
        self.assertTrue(rows[-1]["terminated"])

    def test_success_on_an_earlier_step_is_recorded_on_that_step(self):
        env = FakeEnv(success_at=2)
        self.install_mocks(env)

        rows = evaluation_runner.run_group(SPEC, GROUP, 0)

        self.assertEqual([row["success"] for row in rows], [False, True, True, True, True])

    def test_terminated_and_truncated_come_from_the_same_step_as_the_action(self):
        env = FakeEnv(success_at=3, terminate_at=3)
        self.install_mocks(env)

        rows = evaluation_runner.run_group(SPEC, GROUP, 0)

        self.assertEqual(len(rows), 3)
        for row in rows[:-1]:
            self.assertFalse(row["terminated"])
        self.assertTrue(rows[-1]["terminated"])
        self.assertFalse(rows[-1]["truncated"])

    def test_centroid_and_defense_latency_match_the_executed_action(self):
        env = FakeEnv()
        policy = self.install_mocks(env)

        rows = evaluation_runner.run_group(SPEC, GROUP, 0)

        for index, row in enumerate(rows):
            expected_action = float(index + 1)
            # 动作来自第 index+1 次策略调用
            self.assertAlmostEqual(row["action"][0], expected_action)
            # centroid_offset 取自同一帧的 last_centroid=(index+1, 2.0)
            self.assertAlmostEqual(row["area"], expected_action / 10.0)
            # 防御延迟属于执行该动作的那一次 step
            self.assertAlmostEqual(row["defense_latency_ms"], 10.0 * expected_action)
        self.assertEqual(policy.calls, 5)
        self.assertEqual(env.calls, 5)

    def test_step_count_matches_env_step_calls(self):
        env = FakeEnv(terminate_at=4)
        self.install_mocks(env)

        rows = evaluation_runner.run_group(SPEC, GROUP, 0)

        self.assertEqual(env.calls, 4)
        self.assertEqual(len(rows), 4)
        self.assertEqual([row["step"] for row in rows], [0, 1, 2, 3])


class EnvCleanupTests(MockedRunnerCase):
    def test_env_is_closed_when_reset_fails(self):
        env = FakeEnv(fail_on_reset=True)
        self.install_mocks(env)

        with self.assertRaises(RuntimeError):
            evaluation_runner.run_group(SPEC, GROUP, 0)

        self.assertTrue(env.closed, "reset 失败后 env 没有被关闭")

    def test_env_is_closed_when_policy_build_fails(self):
        env = FakeEnv()
        self.install_mocks(env)
        registry = sys.modules["experiments.sdk.registry"]

        def boom(*args, **kwargs):
            raise RuntimeError("模拟构建策略失败")

        registry.build_policy = boom
        with self.assertRaises(RuntimeError):
            evaluation_runner.run_group(SPEC, GROUP, 0)

        self.assertTrue(env.closed, "构建策略失败后 env 没有被关闭")

    def test_env_is_closed_on_the_normal_path(self):
        env = FakeEnv()
        self.install_mocks(env)

        evaluation_runner.run_group(SPEC, GROUP, 0)

        self.assertTrue(env.closed)


class PairedDivergenceTests(unittest.TestCase):
    def test_divergence_pairs_each_seed_with_its_own_clean_run(self):
        # seed0 上完全一致，seed1 上差异为 10。
        results = [
            make_run("clean", 0, [0.0] * 5),
            make_run("clean", 1, [1.0] * 5),
            make_run("attacked", 0, [0.0] * 5),
            make_run("attacked", 1, [11.0] * 5),
        ]

        report = build_report({**SPEC, "seeds": [0, 1]}, results, compute_gate(results))
        attacked = next(row for row in report["groups"] if row["group"] == "attacked")

        # per-seed: 0.0 与 10.0 -> 平均 5.0
        self.assertAlmostEqual(attacked["divergence_vs_clean"], 5.0)
        self.assertEqual(attacked["divergence_pairs"], 2)
        self.assertEqual(attacked["divergence_unpaired_seeds"], [])

    def test_different_step_counts_are_not_aligned_across_seeds(self):
        results = [
            make_run("clean", 0, [0.0] * 5),
            make_run("clean", 1, [1.0] * 5),
            make_run("attacked", 0, [0.0] * 3),
            make_run("attacked", 1, [11.0] * 5),
        ]

        report = build_report({**SPEC, "seeds": [0, 1]}, results, compute_gate(results))
        attacked = next(row for row in report["groups"] if row["group"] == "attacked")

        # seed0 只比 3 步 = 0，seed1 比满 5 步 = 10，平均 5.0
        self.assertAlmostEqual(attacked["divergence_vs_clean"], 5.0)

    def test_seed_without_a_clean_counterpart_is_reported_not_dropped_silently(self):
        results = [
            make_run("clean", 0, [0.0] * 5),
            make_run("attacked", 0, [0.0] * 5),
            make_run("attacked", 1, [5.0] * 5),
        ]

        report = build_report({**SPEC, "seeds": [0, 1]}, results, compute_gate(results))
        attacked = next(row for row in report["groups"] if row["group"] == "attacked")

        self.assertEqual(attacked["divergence_pairs"], 1)
        self.assertEqual(attacked["divergence_unpaired_seeds"], [1])
        self.assertAlmostEqual(attacked["divergence_vs_clean"], 0.0)

    def test_history_without_seed_fields_still_pairs(self):
        results = [
            {"group": "clean", "status": "ok", "success_observed": False, "rows": [],
             "action_x_series": [0.0, 0.0]},
            {"group": "attacked", "status": "ok", "success_observed": False, "rows": [],
             "action_x_series": [0.0, 1.0]},
        ]

        report = build_report(SPEC, results, compute_gate(results))
        attacked = next(row for row in report["groups"] if row["group"] == "attacked")

        self.assertAlmostEqual(attacked["divergence_vs_clean"], 0.5)
        self.assertEqual(attacked["divergence_pairs"], 1)

    def test_empty_rows_do_not_break_the_report(self):
        # 历史结果里存在 rows 为空的 ok 运行；summarize_group 必须返回完整键集。
        results = [
            {"group": "clean", "seed": 0, "status": "ok", "success_observed": False, "rows": []},
            {"group": "attacked", "seed": 0, "status": "ok", "success_observed": False, "rows": []},
        ]

        report = build_report(SPEC, results, compute_gate(results))

        for row in report["groups"]:
            self.assertIn("mean_defense_latency_ms", row)


class ZeroDivergenceTests(unittest.TestCase):
    def test_zero_divergence_is_a_value_not_missing_data(self):
        results = [
            make_run("clean", 0, [0.0] * 3),
            make_run("attacked", 0, [0.0] * 3),
        ]

        report = build_report({**SPEC, "defense": "median"}, results, compute_gate(results))
        attacked = next(row for row in report["groups"] if row["group"] == "attacked")

        self.assertIsNotNone(attacked["divergence_vs_clean"])
        self.assertEqual(attacked["divergence_vs_clean"], 0.0)

    def test_recovery_ratio_explains_zero_denominator_instead_of_silent_none(self):
        results = [
            make_run("clean", 0, [0.0] * 3),
            make_run("attacked", 0, [0.0] * 3),
            make_run("attacked_defended", 0, [1.0] * 3),
        ]

        report = build_report({**SPEC, "defense": "median"}, results, compute_gate(results))
        defended = next(row for row in report["groups"] if row["group"] == "attacked_defended")

        self.assertIsNone(defended["recovery_ratio"])
        self.assertIn("散度均为 0", defended["recovery_note"])

    def test_recovery_ratio_is_computed_when_denominator_is_positive(self):
        results = [
            make_run("clean", 0, [0.0] * 3),
            make_run("attacked", 0, [2.0] * 3),
            make_run("attacked_defended", 0, [0.5] * 3),
        ]

        report = build_report({**SPEC, "defense": "median"}, results, compute_gate(results))
        defended = next(row for row in report["groups"] if row["group"] == "attacked_defended")

        self.assertAlmostEqual(defended["recovery_ratio"], 0.75)
        self.assertIsNone(defended["recovery_note"])

    def test_missing_divergence_stays_none_and_says_so(self):
        results = [
            make_run("clean", 0, [0.0] * 3),
            make_run("attacked", 0, [2.0] * 3),
            make_run("attacked_defended", 1, [0.5] * 3),
        ]

        report = build_report({**SPEC, "defense": "median"}, results, compute_gate(results))
        defended = next(row for row in report["groups"] if row["group"] == "attacked_defended")

        self.assertIsNone(defended["recovery_ratio"])
        self.assertIn("没有共同有效 seed", defended["recovery_note"])

    def test_recovery_uses_only_seeds_shared_by_all_three_groups(self):
        # attacked 只有 seed0，attacked_defended 只有 seed1。
        # 两者均值相除会造出一个错位的“回收比例”，必须拒绝。
        results = [
            make_run("clean", 0, [0.0] * 3),
            make_run("clean", 1, [0.0] * 3),
            make_run("attacked", 0, [2.0] * 3),
            make_run("attacked_defended", 1, [1.0] * 3),
        ]

        report = build_report({**SPEC, "seeds": [0, 1], "defense": "median"}, results, compute_gate(results))
        defended = next(row for row in report["groups"] if row["group"] == "attacked_defended")

        self.assertIsNone(defended["recovery_ratio"])
        self.assertEqual(defended["recovery_pairs"], 0)
        self.assertIn("没有共同有效 seed", defended["recovery_note"])

    def test_recovery_uses_the_intersection_when_some_seeds_are_missing(self):
        # 共同 seed 只有 1；seed0 只在 attacked 侧存在，不得参与。
        results = [
            make_run("clean", 0, [0.0] * 3),
            make_run("clean", 1, [0.0] * 3),
            make_run("attacked", 0, [100.0] * 3),
            make_run("attacked", 1, [2.0] * 3),
            make_run("attacked_defended", 1, [0.5] * 3),
        ]

        report = build_report({**SPEC, "seeds": [0, 1], "defense": "median"}, results, compute_gate(results))
        defended = next(row for row in report["groups"] if row["group"] == "attacked_defended")

        # 只用 seed1：1 - 0.5/2 = 0.75；若误用 attacked 均值则会是另一个值。
        self.assertAlmostEqual(defended["recovery_ratio"], 0.75)
        self.assertEqual(defended["recovery_pairs"], 1)
        self.assertEqual(defended["recovery_missing_seeds"], [0])
        self.assertIn("缺少 seed", defended["recovery_note"])

    def test_recovery_coverage_survives_failed_runs(self):
        # failed 运行不进分母，但缺失的 seed 必须在 coverage 里可见。
        results = [
            make_run("clean", 0, [0.0] * 3),
            make_run("clean", 1, [0.0] * 3),
            make_run("attacked", 0, [2.0] * 3),
            {"group": "attacked", "seed": 1, "status": "failed", "success_observed": False, "rows": []},
            make_run("attacked_defended", 0, [0.5] * 3),
        ]

        report = build_report({**SPEC, "seeds": [0, 1], "defense": "median"}, results, compute_gate(results))
        attacked = next(row for row in report["groups"] if row["group"] == "attacked")
        defended = next(row for row in report["groups"] if row["group"] == "attacked_defended")

        self.assertEqual(attacked["failed_runs"], 1)
        # failed 运行不进 unpaired（它记在 failed_runs）；覆盖率靠 recovery_missing_seeds 体现。
        self.assertEqual(attacked["divergence_unpaired_seeds"], [])
        self.assertEqual(attacked["divergence_pairs"], 1)
        self.assertAlmostEqual(defended["recovery_ratio"], 0.75)
        self.assertEqual(defended["recovery_pairs"], 1)

    def test_divergence_helper_treats_empty_as_missing(self):
        self.assertIsNone(divergence([], [1.0]))
        self.assertEqual(divergence([0.0, 0.0], [0.0, 0.0]), 0.0)


class GateTransparencyTests(unittest.TestCase):
    def test_failed_runs_are_counted_but_do_not_change_the_protocol(self):
        results = [
            make_run("clean", 0, [0.0], success=True),
            {"group": "clean", "seed": 1, "status": "failed", "success_observed": False, "rows": []},
            {"group": "clean", "seed": 2, "status": "failed", "success_observed": False, "rows": []},
        ]

        gate = compute_gate(results)

        # 口径未改：仍按 ok 运行算 rate。
        self.assertEqual(gate["clean_success_rate"], 1.0)
        self.assertEqual(gate["clean_runs"], 1)
        # 但分布现在透明可查。
        self.assertEqual(gate["clean_failed_runs"], 2)
        self.assertEqual(gate["clean_attempts"], 3)
        self.assertIn("failed", gate["rate_denominator"])

    def test_summarize_group_reports_the_new_outcome_fields(self):
        rows = [
            {"step": 0, "action": [0.0], "centroid_offset": [0.0, 0.0], "success": True,
             "defense_latency_ms": None, "terminated": False, "truncated": False},
        ]

        summary = summarize_group(rows)

        self.assertEqual(summary["steps"], 1)
        self.assertTrue(summary["success_observed"])


if __name__ == "__main__":
    unittest.main()
