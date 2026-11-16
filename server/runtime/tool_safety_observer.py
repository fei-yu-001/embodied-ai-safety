"""Runtime observer that turns one scene_state poll into tool-safety evidence.

The observer is the only piece that holds state. It accepts the same snapshot
the 3D view polls (`body_transforms` plus the static `geometry` object
dictionary) and never reads or writes MuJoCo: the adapter decides when to call
`update` and how to surface the result.

State rules:
- evidence is keyed by `(session_id, episode)`; a new session or episode clears
  everything, including exposure time and entry counts;
- within one key, a repeated `(session_id, episode, step)` is a duplicate poll:
  it is reported as such and never accumulates exposure again, so pausing and
  polling in a loop does not inflate exposure or entries;
- a step that goes backwards without an episode change means a rewound or
  replaced run: the observer clears evidence and says why instead of adding a
  new episode's numbers onto the old ones;
- exposure comes from sampled steps only. It is a discrete estimate: the
  interval since the previous step, and only when both steps agree that the
  zone is occupied. An unknown step is not known occupancy, so it never adds
  exposure;
- only a violation event is recorded, never a "safe" claim on missing data.
"""

from __future__ import annotations

from server.domain.tool_safety import (
    FRAME,
    NOTE,
    STATUS_DISABLED,
    STATUS_UNKNOWN,
    TOOL_SAFETY_SPEC_VERSION,
    assess_tool_safety,
    normalize_tool_safety_spec,
)

DEFAULT_KEEP_EVENTS = 200

SAMPLING_NOTE = (
    "按采样步评估，不构成连续扫掠碰撞保证：两个采样步之间的穿越不会被发现，"
    "exposure_seconds 是由相邻采样步推得的离散估计；未知步不累积暴露时间。"
)


class ToolSafetyObserver:
    """Disabled by default; enabling requires an explicit spec or flag."""

    def __init__(self, spec=None, enabled=None, keep_events=DEFAULT_KEEP_EVENTS):
        # 存原始 spec 而不是归一化副本：assess 每步都会完整重验，规范化副本
        # 等于绕开校验（NaN、未知字段、重复 id 都能藏在“看起来规范”的 dict 里）。
        self._spec = normalize_tool_safety_spec(spec)
        self.keep_events = int(keep_events)
        self._enabled_override = None if enabled is None else bool(enabled)
        self._key = None
        self._last_step = None
        self._last_sim_seconds = None
        self._active_zones = set()
        self._context = {}
        self._reset_state("never_run")

    # -- configuration -----------------------------------------------------

    @property
    def spec(self):
        return normalize_tool_safety_spec(self._spec)

    @property
    def enabled(self):
        if self._enabled_override is not None:
            return self._enabled_override
        return bool(self._spec.get("enabled"))

    def _effective_spec(self):
        spec = self._spec
        if self._enabled_override is None:
            return spec
        return {**spec, "enabled": self._enabled_override}

    def set_spec(self, raw, enabled=None):
        """Replace the spec and clear all evidence.

        A full reset, not just a metrics reset: keeping `_key`/`_last_step` would
        let a new rule set inherit the previous zone occupancy or treat the same
        step as a duplicate and silently skip assessment.
        """
        self._spec = normalize_tool_safety_spec(raw)
        self._enabled_override = None if enabled is None else bool(enabled)
        self.reset(reason="spec_changed")
        return self.spec

    # -- state -------------------------------------------------------------

    def _reset_state(self, reason):
        self.reset_reason = reason
        self.events = []
        self.events_dropped = 0
        self._event_seq = 0
        self.zone_entries = {}
        self.exposure_seconds_by_zone = {}
        self.min_zone_clearance = {}
        self.steps_assessed = 0
        self.steps_unknown = 0
        self.steps_disabled = 0
        self.violation_steps = set()
        self.episode_violations = 0
        self.duplicate_polls = 0
        self.unknown_transitions = 0
        self.min_net_clearance = None
        self.min_net_clearance_step = None
        self.checks_total = 0
        self.unknown_steps = 0
        self.sources = set()
        self.approximations = set()

    def reset(self, reason="manual"):
        """Clear all accumulated evidence. Called on session/episode change."""
        self._key = None
        self._last_step = None
        self._last_sim_seconds = None
        self._active_zones = set()
        self._context = {}
        self._reset_state(reason)

    # -- polling -----------------------------------------------------------

    def _payload(self, assessment, status, poll):
        return {
            "spec_version": TOOL_SAFETY_SPEC_VERSION,
            "enabled": bool(assessment.get("enabled")),
            "status": status,
            "assessment": assessment,
            "metrics": self._metrics(assessment),
            "events": list(self.events),
            "overlays": assessment.get("overlays", {}),
            "poll": poll,
            "frame": dict(FRAME),
            "note": NOTE + SAMPLING_NOTE,
        }

    def _metrics(self, assessment):
        episode_violations = self.episode_violations
        return {
            "session_id": assessment.get("session_id"),
            "episode": assessment.get("episode"),
            "step": assessment.get("step"),
            "sim_seconds": assessment.get("sim_seconds"),
            "steps_assessed": self.steps_assessed,
            "steps_unknown": self.steps_unknown,
            "steps_disabled": self.steps_disabled,
            "duplicate_polls": self.duplicate_polls,
            "episode_violations": episode_violations,
            "violation_steps": len(self.violation_steps),
            "zone_entries": sum(self.zone_entries.values()),
            "zone_entries_by_zone": dict(sorted(self.zone_entries.items())),
            "exposure_seconds": round(sum(self.exposure_seconds_by_zone.values()), 6),
            "exposure_seconds_by_zone": {
                zone_id: round(value, 6)
                for zone_id, value in sorted(self.exposure_seconds_by_zone.items())
            },
            "min_net_clearance": self.min_net_clearance,
            "min_net_clearance_step": self.min_net_clearance_step,
            "min_zone_clearance": {
                zone_id: dict(row) for zone_id, row in sorted(self.min_zone_clearance.items())
            },
            "checks_total": self.checks_total,
            "sources": sorted(self.sources),
            "approximations": sorted(self.approximations),
            "events_recorded": len(self.events),
            "events_dropped": self.events_dropped,
            "reset_reason": self.reset_reason,
            "unknown_transitions": self.unknown_transitions,
            "denominators": {
                "steps_assessed": self.steps_assessed,
                "steps_observed": self.steps_assessed + self.steps_unknown + self.steps_disabled,
                "checks_evaluated": self.checks_total,
            },
            "sampling": "discrete_step",
            "collision_guarantee": False,
            "exposure_is_estimate": True,
            "note": "指标分母与单位：step 计数、秒、米；未评估数据不计入安全结论。" + SAMPLING_NOTE,
        }

    def update(self, scene_state):
        """Assess one poll and fold it into the episode metrics."""
        scene_state = scene_state if isinstance(scene_state, dict) else {}
        session_id = scene_state.get("session_id")
        episode = scene_state.get("episode")
        step = scene_state.get("step")
        key = (session_id, episode)
        spec = self._effective_spec()

        if key != self._key:
            self.reset(reason="session_changed" if self._key is not None else "first_observation")
            self._key = key

        if self._last_step is not None and step == self._last_step:
            self.duplicate_polls += 1
            assessment = assess_tool_safety(spec, scene_state)
            self._remember_context(assessment)
            return self._payload(
                assessment,
                assessment["status"],
                {"accepted": False, "duplicate": True, "reason": "duplicate_step"},
            )

        # 同 episode 内步数回退：不是新 episode，也不能把退后的步当成正常前进。
        # 判定为运行被回退/重置，清空指标并以当前步重新开始，避免旧数据被累加。
        if (
            self._last_step is not None
            and isinstance(step, int)
            and isinstance(self._last_step, int)
            and step < self._last_step
        ):
            self.reset(reason="step_rewind")
            self._key = key

        previous_time = self._last_sim_seconds
        previous_zones = set(self._active_zones)
        self._last_step = step
        self._last_sim_seconds = scene_state.get("sim_seconds")

        assessment = assess_tool_safety(spec, scene_state)
        self._remember_context(assessment)
        status = assessment["status"]

        if status == STATUS_DISABLED:
            self.steps_disabled += 1
            previous_zones = set()
        elif status == STATUS_UNKNOWN:
            self.steps_unknown += 1
            self.unknown_steps += 1
            # 未知不是“已知占用”：上一段的占用状态到此结束，下一段重新计进入
            self.unknown_transitions += 1
            previous_zones = set()
        else:
            self.steps_assessed += 1

        delta = 0.0
        if (
            status != STATUS_UNKNOWN
            and previous_time is not None
            and isinstance(self._last_sim_seconds, (int, float))
            and not isinstance(self._last_sim_seconds, bool)
            and self._last_sim_seconds >= previous_time
        ):
            delta = float(self._last_sim_seconds - previous_time)

        checks = assessment.get("checks") or []
        self.checks_total += len(checks)
        for check in checks:
            clearance = check.get("net_clearance")
            if clearance is None:
                continue
            if self.min_net_clearance is None or clearance < self.min_net_clearance:
                self.min_net_clearance = clearance
                self.min_net_clearance_step = step
            zone_id = check["zone_id"]
            row = self.min_zone_clearance.get(zone_id)
            if row is None or clearance < row["min_net_clearance"]:
                self.min_zone_clearance[zone_id] = {
                    "min_net_clearance": clearance,
                    "step": step,
                    "tool_id": check["tool_id"],
                    "segment_id": check["segment_id"],
                    "source": check["zone_source"],
                }
            self.sources.add(check["zone_source"])
            self.sources.add(check["tool_source"])
            self.approximations.add(check["approximation"])

        violations = assessment.get("violations") or []
        violating_zones = {check["zone_id"] for check in violations}
        entries = violating_zones - previous_zones
        for zone_id in entries:
            self.zone_entries[zone_id] = self.zone_entries.get(zone_id, 0) + 1
        # 连续占用才计暴露：只有上一步也判定为占用，这一段时长才算已确认占用
        if delta > 0.0:
            for zone_id in violating_zones & previous_zones:
                self.exposure_seconds_by_zone[zone_id] = (
                    self.exposure_seconds_by_zone.get(zone_id, 0.0) + delta
                )
        if violations:
            self.violation_steps.add(step)
            self.episode_violations += len(violations)
            self._record(violations, assessment, entries)
        self._active_zones = violating_zones

        return self._payload(
            assessment,
            status,
            {"accepted": True, "duplicate": False, "reason": None},
        )

    def _record(self, violations, assessment, entries):
        """Append one event per violating (zone, segment) pair.

        Events are evidence of a geometric condition, not a control action: the
        observer never stops the robot and does not claim it did.
        """
        for check in violations:
            self._event_seq += 1
            event = {
                "event_id": f"tsv_{self._event_seq:06d}",
                "kind": "zone_intrusion",
                "rule_id": check["rule_id"],
                "session_id": assessment.get("session_id"),
                "episode": assessment.get("episode"),
                "step": assessment.get("step"),
                "sim_seconds": assessment.get("sim_seconds"),
                "zone_id": check["zone_id"],
                "zone_source": check["zone_source"],
                "tool_id": check["tool_id"],
                "segment_id": check["segment_id"],
                "tool_source": check["tool_source"],
                "approximation": check["approximation"],
                "distance": check["distance"],
                "surface_distance": check["surface_distance"],
                "margin": check["margin"],
                "net_clearance": check["net_clearance"],
                "zone_entry": check["zone_id"] in entries,
                "disposition": "observed_only",
                "note": "仅几何判定，未执行停止、避让或伤害评估",
            }
            if len(self.events) < self.keep_events:
                self.events.append(event)
            else:
                self.events_dropped += 1

    def _remember_context(self, assessment):
        """Keep the newest assessment identity so `metrics` is not context-free."""
        self._context = {
            "session_id": assessment.get("session_id"),
            "episode": assessment.get("episode"),
            "step": assessment.get("step"),
            "sim_seconds": assessment.get("sim_seconds"),
        }

    @property
    def metrics(self):
        return self._metrics(self._context)

    @property
    def context(self):
        return dict(self._context)
