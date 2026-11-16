"""End-to-end smoke check for the M1 control plane."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

import httpx

BASE = "http://127.0.0.1:8766"
TOKEN = {"X-Console-Token": os.environ.get("PRISM_TOKEN", "smoke-token")}
REAL = os.environ.get("SMOKE_REAL") == "1"


def main():
    client = httpx.Client(base_url=BASE, headers=TOKEN, timeout=300)
    payload = {"config": {"seed": 0, "control_freq": 20}} if REAL else None
    created = client.post("/api/v1/sessions", json=payload)
    created.raise_for_status()
    session = created.json()
    session_id = session["id"]
    print("created", session_id, session["state"], "backend", session.get("backend"))

    for _ in range(100):
        current = client.get(f"/api/v1/sessions/{session_id}").json()
        if current["state"] == "READY":
            break
        if current["state"] in {"FAILED", "INTERRUPTED"}:
            print("worker failed", current)
            return 1
        import time
        time.sleep(0.2)
    print("ready", current["state"], "version", current["state_version"], "loads_physics", current.get("loads_physics"))

    now = datetime.now(timezone.utc)
    body = {
        "command_id": "smoke-step-1",
        "fencing_token": current["fencing_token"],
        "command": "step",
        "issued_at": now.isoformat(),
        "expires_at": (now + timedelta(seconds=30)).isoformat(),
        "expected_state_version": current["state_version"],
    }
    first = client.post(f"/api/v1/sessions/{session_id}/commands", json=body)
    print("submit", first.status_code, first.json()["acknowledgement"])

    duplicate = client.post(f"/api/v1/sessions/{session_id}/commands", json={**body, "command": "reset"})
    print("duplicate", duplicate.status_code, duplicate.json()["acknowledgement"])

    for _ in range(100):
        events = client.get(f"/api/v1/sessions/{session_id}/events", params={"after": 0}).json()["events"]
        if any(e.get("command_id") == "smoke-step-1" and e["type"] == "command_applied" for e in events):
            break
        import time
        time.sleep(0.2)

    applied = [e for e in events if e.get("command_id") == "smoke-step-1"]
    print("command events", [(e["type"], e["seq"]) for e in applied])
    final = client.get(f"/api/v1/sessions/{session_id}").json()
    print("final", final["state"], "applied_steps", final["applied_steps"])
    ok = (
        final["state"] == "PAUSED"
        and final["applied_steps"] == 1
        and [e["type"] for e in applied] == ["command_accepted", "command_applied"]
    )
    if REAL:
        frame = _data_root() / session_id / "frames" / "latest.jpg"
        size = frame.stat().st_size if frame.is_file() else 0
        print("real_frame", frame, "bytes", size, "loads_physics", final.get("loads_physics"))
        ok = ok and bool(final.get("loads_physics")) and size > 1000
    if REAL:
        now = datetime.now(timezone.utc)
        current = client.get(f"/api/v1/sessions/{session_id}").json()
        jog_body = {
            "command_id": "smoke-jog-1",
            "fencing_token": current["fencing_token"],
            "command": "jog",
            "issued_at": now.isoformat(),
            "expires_at": (now + timedelta(seconds=60)).isoformat(),
            "expected_state_version": current["state_version"],
            "args": {"delta_m": [0.02, 0.0, 0.0], "gripper": "close", "repeat": 2},
        }
        jog = client.post(f"/api/v1/sessions/{session_id}/commands", json=jog_body)
        print("jog", jog.status_code, jog.json()["acknowledgement"])
        for _ in range(240):
            events_now = client.get(f"/api/v1/sessions/{session_id}/events", params={"after": 0}).json()["events"]
            if any(e.get("command_id") == "smoke-jog-1" and e["type"] == "command_applied" for e in events_now):
                break
            import time as _t
            _t.sleep(0.25)
        detail = client.get(f"/api/v1/sessions/{session_id}").json()
        print("jog_detail", detail.get("last_command"))
        ok = ok and detail.get("last_command", {}).get("jog", {}).get("gripper_state") == "close"

    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _data_root():
    from server.main import DATA_ROOT

    return DATA_ROOT


if __name__ == "__main__":
    sys.exit(main())
