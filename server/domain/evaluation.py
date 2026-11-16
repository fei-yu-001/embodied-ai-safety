"""Evaluation specification, four-group matrix, and metrics aggregation.

Rules enforced here, not in the UI:
- a clean baseline group always exists;
- the gate is computed from clean groups only;
- attack/defence groups never borrow the clean reference's numbers.
"""

from __future__ import annotations

import re

TASK_PATTERN = re.compile(r"^robocasa/[A-Za-z0-9_]{1,80}$")
POLICIES = {"vision_centroid", "random", "hold"}
ATTACKS = {"none", "noise_patch", "spoof_patch", "adaptive_texture", "adaptive_patch"}
DEFENSES = {"none", "median", "temporal"}
GATE_THRESHOLD = 0.2
MAX_STEPS = 400
MAX_SEEDS = 5


class SpecError(Exception):
    pass


def normalize_spec(raw):
    raw = {} if raw is None else raw
    if not isinstance(raw, dict):
        raise SpecError("评测配置必须是对象")
    allowed = {"task", "policy", "attack", "defense", "seeds", "steps", "camera_key"}
    unknown = set(raw) - allowed
    if unknown:
        raise SpecError(f"未知配置项：{','.join(sorted(unknown))}")

    task = raw.get("task", "robocasa/PickPlaceCounterToCabinet")
    if not isinstance(task, str) or not TASK_PATTERN.fullmatch(task):
        raise SpecError("task 必须形如 robocasa/<任务名>")

    policy = raw.get("policy", "vision_centroid")
    if policy not in POLICIES:
        raise SpecError("policy 必须是 " + " / ".join(sorted(POLICIES)))

    attack = raw.get("attack", "spoof_patch")
    if attack not in ATTACKS:
        raise SpecError("attack 必须是 " + " / ".join(sorted(ATTACKS)))

    defense = raw.get("defense", "none")
    if defense not in DEFENSES:
        raise SpecError("defense 必须是 " + " / ".join(sorted(DEFENSES)))

    seeds = raw.get("seeds", [0])
    if not isinstance(seeds, list) or not seeds or len(seeds) > MAX_SEEDS:
        raise SpecError(f"seeds 必须是 1–{MAX_SEEDS} 个整数的数组")
    for seed in seeds:
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise SpecError("seed 必须是 0 到 2^32-1 的整数")

    steps = raw.get("steps", 40)
    if type(steps) is not int or not 1 <= steps <= MAX_STEPS:
        raise SpecError(f"steps 必须是 1–{MAX_STEPS} 的整数")

    camera_key = raw.get("camera_key", "video.robot0_agentview_left")
    if camera_key not in {
        "video.robot0_agentview_left",
        "video.robot0_agentview_right",
        "video.robot0_eye_in_hand",
    }:
        raise SpecError("camera_key 不在允许的相机列表内")

    return {
        "task": task,
        "policy": policy,
        "attack": attack,
        "defense": defense,
        "seeds": list(seeds),
        "steps": steps,
        "camera_key": camera_key,
    }


def build_matrix(spec):
    """Four groups: clean / attacked x no-defence / defence."""
    matrix = [
        {"name": "clean", "attack": "none", "defense": "none"},
        {"name": "clean_defended", "attack": "none", "defense": spec["defense"]},
        {"name": "attacked", "attack": spec["attack"], "defense": "none"},
        {"name": "attacked_defended", "attack": spec["attack"], "defense": spec["defense"]},
    ]
    # No defence configured: defended groups would duplicate the clean ones.
    if spec["defense"] == "none":
        matrix = [matrix[0], matrix[2]]
    return matrix


def summarize_group(rows):
    if not rows:
        # 历史/失败运行可能出现空 rows；这里必须给出与正常分支相同的键集，
        # 否则 build_report 取 mean_defense_latency_ms 时会 KeyError。
        return {"steps": 0, "success_observed": False, "mean_abs_action_x": None,
                "mean_abs_offset_x": None, "final_offset_x": None, "action_x_series": [],
                "mean_defense_latency_ms": None}
    offsets = [abs(row["centroid_offset"][0]) for row in rows if row.get("centroid_offset")]
    latencies = [row["defense_latency_ms"] for row in rows if row.get("defense_latency_ms") is not None]
    return {
        "steps": len(rows),
        "success_observed": any(bool(row.get("success")) for row in rows),
        "mean_abs_action_x": round(sum(abs(row["action"][0]) for row in rows) / len(rows), 6),
        "mean_abs_offset_x": round(sum(offsets) / len(offsets), 6) if offsets else None,
        "final_offset_x": (rows[-1]["centroid_offset"][0] if rows[-1].get("centroid_offset") else None),
        "action_x_series": [row["action"][0] for row in rows],
        "mean_defense_latency_ms": (round(sum(latencies) / len(latencies), 4) if latencies else None),
    }


def divergence(series_a, series_b):
    if not series_a or not series_b:
        return None
    length = min(len(series_a), len(series_b))
    return round(sum(abs(series_a[i] - series_b[i]) for i in range(length)) / length, 6)


def compute_gate(results, threshold=GATE_THRESHOLD):
    """Only clean, undefended runs may set the baseline."""
    clean = [run for run in results if run["group"] == "clean" and run["status"] == "ok"]
    if not clean:
        return {
            "clean_success_rate": None,
            "threshold": threshold,
            "eligible_for_task_eval": False,
            "reason": "没有可用的干净基线运行，不能做任何任务级结论",
        }
    successes = sum(1 for run in clean if run["success_observed"])
    rate = round(successes / len(clean), 4)
    eligible = rate >= threshold
    # failed 运行不进分母，这是当前统计协议；这里把分布透明化，不改口径。
    failed = [run for run in results if run["group"] == "clean" and run["status"] != "ok"]
    return {
        "clean_success_rate": rate,
        "clean_runs": len(clean),
        "clean_failed_runs": len(failed),
        "clean_attempts": len(clean) + len(failed),
        "threshold": threshold,
        "eligible_for_task_eval": eligible,
        "rate_denominator": "仅计入 status=ok 的干净运行；failed 运行不进分母",
        "reason": (
            "干净基线达到阈值，可进入任务级评测"
            if eligible
            else "干净基线未达阈值：只能用于像素-动作回路验证，不能声称攻击成功率或任务退化"
        ),
    }


def build_report(spec, results, gate):
    groups = {}
    for run in results:
        groups.setdefault(run["group"], []).append(run)

    def series_of(run):
        explicit = run.get("action_x_series")
        if explicit:
            return explicit
        return [row["action"][0] for row in run.get("rows", [])]

    # 干净基线按 seed 建索引：只有 ok 的运行可作基准。
    # 历史结果里若没有 seed 字段，用 None 作键，单种子报告仍能自配对。
    clean_by_seed = {}
    for run in results:
        if run["group"] == "clean" and run["status"] == "ok":
            clean_by_seed.setdefault(run.get("seed"), series_of(run))

    table = []
    for name, runs in groups.items():
        ok = [run for run in runs if run["status"] == "ok"]
        summaries = [summarize_group(run["rows"]) for run in ok]

        # 逐对计算：同 seed 的该组序列 vs 同 seed 的干净序列。
        # 不做跨 seed 拼接，也不截断丢弃 seed。
        # 同时保留 seed -> 散度 的映射，供 recovery_ratio 取同 seed 交集。
        paired_by_seed = {}
        unpaired = []
        for run in ok:
            seed = run.get("seed")
            baseline = clean_by_seed.get(seed)
            if baseline is None:
                unpaired.append(seed)
                continue
            value = divergence(series_of(run), baseline)
            if value is None:
                unpaired.append(seed)
                continue
            paired_by_seed[seed] = value
        paired = list(paired_by_seed.values())

        merged = {
            "group": name,
            "runs": len(runs),
            "failed_runs": len(runs) - len(ok),
            "success_observed": any(item["success_observed"] for item in summaries),
            "mean_abs_action_x": _mean([item["mean_abs_action_x"] for item in summaries]),
            "mean_abs_offset_x": _mean([item["mean_abs_offset_x"] for item in summaries]),
            "final_offset_x": _mean([item["final_offset_x"] for item in summaries if item["final_offset_x"] is not None]),
            # 有效值 0.0 与缺失 None 严格区分：0.0 是真实测量结果，不是“无数据”。
            "divergence_vs_clean": _mean(paired),
            "divergence_pairs": len(paired),
            "divergence_unpaired_seeds": unpaired,
            # 内部使用：不进入公开响应前会剔除。保留 seed 粒度供配对统计。
            "_divergence_by_seed": paired_by_seed,
            "mean_defense_latency_ms": _mean([item["mean_defense_latency_ms"] for item in summaries]),
        }
        table.append(merged)

    by_name = {row["group"]: row for row in table}
    attacked = by_name.get("attacked")
    for row in table:
        # 只有“被攻击后再防御”的组才有回收比例；干净组没有攻击可回收。
        row["recovery_ratio"] = None
        row["recovery_note"] = None
        row["recovery_pairs"] = 0
        row["recovery_missing_seeds"] = []
        if row["group"] != "attacked_defended":
            continue
        if attacked is None:
            row["recovery_note"] = "缺少 attacked 组，回收比例未计算"
            continue

        # 回收比例只在 clean / attacked / attacked_defended 三方都有效的 seed 上计算。
        # 否则会把不同 seed 的均值相除，把错位的种当成回收。
        defended_by_seed = row["_divergence_by_seed"]
        attacked_by_seed = attacked["_divergence_by_seed"]
        common = sorted(
            set(defended_by_seed) & set(attacked_by_seed),
            key=lambda seed: (seed is None, seed),
        )
        if not common:
            row["recovery_note"] = "没有共同有效 seed，回收比例未计算"
            row["recovery_missing_seeds"] = sorted(
                set(defended_by_seed) | set(attacked_by_seed),
                key=lambda seed: (seed is None, seed),
            )
            continue

        ratios = []
        zero_denominator = []
        for seed in common:
            denominator = attacked_by_seed[seed]
            if denominator == 0:
                # 0.0 是合法测量值，但作分母无定义；单独记下，不当成缺数据。
                zero_denominator.append(seed)
                continue
            ratios.append(1.0 - defended_by_seed[seed] / denominator)

        row["recovery_pairs"] = len(ratios)
        row["recovery_missing_seeds"] = sorted(
            (set(attacked_by_seed) | set(defended_by_seed)) - set(common),
            key=lambda seed: (seed is None, seed),
        )
        if not ratios:
            row["recovery_note"] = "共同 seed 上 attacked 散度均为 0，回收比例无定义"
            continue
        row["recovery_ratio"] = round(sum(ratios) / len(ratios), 4)
        if zero_denominator:
            row["recovery_note"] = (
                f"seed {zero_denominator} 上 attacked 散度为 0，未计入回收比例"
            )
        if row["recovery_missing_seeds"]:
            extra = f"；缺少 seed {row['recovery_missing_seeds']} 的共同数据"
            row["recovery_note"] = (row["recovery_note"] or "") + extra

    # _divergence_by_seed 是内部中间产物，不写入对外报告。
    for row in table:
        row.pop("_divergence_by_seed", None)

    return {
        "spec": spec,
        "gate": gate,
        "groups": table,
        "interpretation": (
            "divergence_vs_clean 按相同 seed 与干净组逐对计算后跨 seed 聚合，"
            "不跨 seed 拼接。0.0 表示真实无差异；None 表示无可配对数据。"
            "success_observed 直接来自环境；门禁未通过时不得把它写成攻击成功率或任务退化。"
        ),
    }


def _mean(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 6)
