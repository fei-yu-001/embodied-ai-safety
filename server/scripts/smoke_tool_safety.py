"""Real-kitchen tool safety smoke on GatherCuttingTools.

Runs in its own backend process (no server, no existing session touched). It
loads the registered knife task, finds the real knife body from the live model,
builds a spec from the observed world pose, and checks that:

- a zone placed on the knife's current world position reports a violation,
- a zone far away reports ok,
- a stale/renamed tool body reports unknown (never safe),
- a disabled spec adds no observer and no overlays.

Slow by nature (scene load ~30-60 s on CPU). Nothing here writes MuJoCo state
beyond the ordinary session steps, and no third-party source is modified.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

from server.runtime.adapters.robocasa_backend import RoboCasaBackend

TASK = "robocasa/GatherCuttingTools"


def find_knife(backend):
    """Return (body_name, world_position) for the task's knife object."""
    import numpy as np

    task = backend.env.unwrapped.env
    sim = task.sim
    objects = getattr(task, "objects", None) or []
    if isinstance(objects, dict):
        objects = list(objects.values())
    for obj in objects:
        name = str(getattr(obj, "name", ""))
        root = getattr(obj, "root_body", None)
        if not root or "knife" not in name.lower():
            continue
        try:
            body_id = int(sim.model.body_name2id(str(root)))
        except (KeyError, ValueError):
            continue
        return str(root), [float(v) for v in np.asarray(sim.data.body_xpos[body_id], dtype=float).reshape(3)]
    return None, None


def spec_with_zone(body, center, radius=0.15, clearance=0.0, enabled=True):
    return {
        "version": "tool_safety_spec/1",
        "enabled": enabled,
        "risk_zones": [{
            "id": "prep_zone",
            "kind": "sphere",
            "center": [float(v) for v in center],
            "radius": float(radius),
            "clearance": float(clearance),
            "label": "刀具附近虚拟保护区",
        }],
        "tool_annotations": [{
            "tool_id": "knife",
            "body": body,
            "approximation": "user_local_segment",
            "source": "user_annotation",
            "segments": [{
                "id": "seg1",
                "kind": "segment",
                "start": [-0.09, 0.0, 0.0],
                "end": [0.09, 0.0, 0.0],
                "radius": 0.002,
                "approximation": "user_local_segment",
            }],
        }],
    }


def report(label, backend):
    payload = backend.tool_safety_payload
    if payload is None:
        return {"label": label, "payload": None}
    metrics = payload.get("metrics") or {}
    return {
        "label": label,
        "status": payload.get("status"),
        "unknown_reasons": (payload.get("assessment") or {}).get("unknown_reasons"),
        "violations": len((payload.get("assessment") or {}).get("violations") or []),
        "min_net_clearance": metrics.get("min_net_clearance"),
        "zone_entries": metrics.get("zone_entries"),
        "exposure_seconds": metrics.get("exposure_seconds"),
        "steps_assessed": metrics.get("steps_assessed"),
        "events": len(payload.get("events") or []),
        "overlay_zones": len(backend._overlays().get("risk_zones") or []),
        "world_tool_segments": len(backend._overlays().get("world_tool_segments") or []),
        "overlay_source": backend._overlays().get("source"),
    }


def main():
    frames = Path(tempfile.mkdtemp(prefix="prism-tool-real-"))
    backend = RoboCasaBackend(frames)

    # 1) 注册与资产可用性
    started = time.perf_counter()
    info = backend.load("sess_tool_real", {"seed": 0, "control_freq": 20, "env_id": TASK})
    print(json.dumps({
        "phase": "load",
        "env_id": info["env_id"],
        "seconds": round(time.perf_counter() - started, 1),
        "step": info["step"],
        "tool_safety": info.get("tool_safety"),
    }, ensure_ascii=False))

    body, position = find_knife(backend)
    if body is None:
        print(json.dumps({"phase": "knife", "found": False, "note": "任务里没有找到 knife 物体"}, ensure_ascii=False))
        return 2
    print(json.dumps({"phase": "knife", "found": True, "body": body, "world_position": [round(v, 4) for v in position]}, ensure_ascii=False))

    results = []

    # 2) 未启用：没有 observer，overlays 明确为空
    print(json.dumps({"phase": "disabled", "observer": backend.tool_safety is None,
                      "overlays": backend._overlays()}, ensure_ascii=False))

    # 3) 启用，保护区就放在刀具当前位置 -> 应 violation
    backend.config = {**backend.config, "tool_safety": spec_with_zone(body, position)}
    backend._tool_safety_bind()
    backend._tool_safety_tick()
    results.append(report("zone_on_knife", backend))

    # 4) 远离刀具的保护区 -> 应 ok
    backend.config = {**backend.config, "tool_safety": spec_with_zone(body, [position[0] + 3.0, position[1], position[2]])}
    backend._tool_safety_bind()
    backend._tool_safety_tick()
    results.append(report("zone_far_away", backend))

    # 5) 未知工具 body -> 应 unknown，机器人不停
    backend.config = {**backend.config, "tool_safety": spec_with_zone("no_such_body", position)}
    backend._tool_safety_bind()
    stepped = backend.apply("step")
    results.append(report("unknown_body", backend))
    results[-1]["step_after"] = stepped["step"]
    results[-1]["state_after"] = stepped["state"]
    results[-1]["summary_tool_safety"] = stepped.get("tool_safety")

    # 6) 真实推进几步，确认逐物理步评估（不是等轮询）
    backend.config = {**backend.config, "tool_safety": spec_with_zone(body, position)}
    backend._tool_safety_bind()
    backend._tool_safety_tick()
    before = backend.tool_safety.metrics["steps_assessed"]
    for _ in range(3):
        backend.apply("step")
    after = backend.tool_safety.metrics["steps_assessed"]
    results.append({"label": "advance_cadence", "steps_assessed_before": before,
                    "after_3_steps": after, "delta": after - before})

    print(json.dumps({"phase": "assessments", "results": results}, ensure_ascii=False, indent=2))

    # 7) 真实 scene_state：必须带 tool_safety 与 overlays，且不因轮询重复评估
    scene_before = backend.tool_safety.metrics["steps_assessed"]
    state = backend.scene_state({"have_geometry_key": "stale"})
    scene_after = backend.tool_safety.metrics["steps_assessed"]
    scene_report = {
        "phase": "scene_state",
        "has_tool_safety": isinstance(state.get("tool_safety"), dict),
        "tool_safety_status": (state.get("tool_safety") or {}).get("status"),
        "overlay_source": (state.get("overlays") or {}).get("source"),
        "overlay_zone_count": len((state.get("overlays") or {}).get("risk_zones") or []),
        "overlay_world_segments": len((state.get("overlays") or {}).get("world_tool_segments") or []),
        "assessed_unchanged_by_poll": scene_before == scene_after,
        "geometry_included": state.get("geometry_included"),
    }
    print(json.dumps(scene_report, ensure_ascii=False))

    ok = (
        results[0]["status"] == "violation"
        and results[1]["status"] == "ok"
        and results[2]["status"] == "unknown"
        and results[2]["step_after"] >= 1
        and results[3]["delta"] == 3
        and scene_report["has_tool_safety"]
        and scene_report["tool_safety_status"] == "violation"
        and scene_report["overlay_zone_count"] >= 1
        and scene_report["overlay_world_segments"] >= 1
        and scene_report["assessed_unchanged_by_poll"]
    )
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
