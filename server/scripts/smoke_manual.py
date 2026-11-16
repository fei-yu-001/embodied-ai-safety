"""Real-kitchen manual motion: hold, keepalive, and stop when heartbeats stop.

Proves the loop lives in the worker, not in the browser: after the last
heartbeat the arm must stop by itself and the session must report why.
"""

from __future__ import annotations

import os
import sys
import time

import httpx

BASE = "http://127.0.0.1:8766"
TOKEN = {"X-Console-Token": os.environ.get("PRISM_TOKEN", "smoke-token")}
TTL_MS = 1500


def main():
    client = httpx.Client(base_url=BASE, headers=TOKEN, timeout=120)
    session = client.post("/api/v1/sessions", json={"config": {"seed": 0, "control_freq": 20}}).json()
    session_id = session["id"]
    print("session", session_id, session.get("backend"), "loads_physics", session.get("loads_physics"))

    lease = client.post(f"/api/v1/sessions/{session_id}/lease", json={"holder": "smoke-operator", "ttl_ms": TTL_MS}).json()
    print("lease", lease["lease_id"], "expires", lease["expires_at"])

    version = client.get(f"/api/v1/sessions/{session_id}").json()["state_version"]
    body = {
        "command_id": "manual-1",
        "fencing_token": session["fencing_token"],
        "command": "jog_start",
        "issued_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "expires_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 30)),
        "expected_state_version": version,
        "args": {"lease_id": lease["lease_id"], "delta_m": [0.01, 0.0, 0.0], "keepalive_ms": TTL_MS},
    }
    started = client.post(f"/api/v1/sessions/{session_id}/commands", json=body)
    print("jog_start", started.status_code, started.json()["acknowledgement"])

    for _ in range(3):
        time.sleep(0.4)
        renewed = client.post(
            f"/api/v1/sessions/{session_id}/lease/{lease['lease_id']}/renew", json={"ttl_ms": TTL_MS}
        )
        print("renew", renewed.status_code, renewed.json().get("revision"))

    running = client.get(f"/api/v1/sessions/{session_id}").json()
    print("during", running["state"], "mode", running["mode"])

    print("stopping heartbeats ...")
    deadline = time.time() + 20
    stop_event = None
    while time.time() < deadline:
        events = client.get(f"/api/v1/sessions/{session_id}/events", params={"after": 0}).json()["events"]
        found = [e for e in events if e["type"] == "manual_stopped"]
        if found:
            stop_event = found[-1]
            break
        time.sleep(0.3)

    final = client.get(f"/api/v1/sessions/{session_id}").json()
    print("after", final["state"], "mode", final["mode"])
    print("stop_event", None if stop_event is None else {k: stop_event[k] for k in ("reason", "steps")})

    frame_dir = _session_dir(session_id) / "frames"
    frames = sorted(frame_dir.glob("frame_*.jpg")) if frame_dir.is_dir() else []
    print("frames", len(frames), "latest_bytes", (frame_dir / "latest.jpg").stat().st_size if (frame_dir / "latest.jpg").is_file() else 0)

    ok = (
        stop_event is not None
        and stop_event["reason"] == "lease_expired"
        and stop_event["steps"] > 1
        and final["state"] == "PAUSED"
        and final["mode"] is None
        and len(frames) > 2
    )
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _session_dir(session_id):
    from server.main import DATA_ROOT

    return DATA_ROOT / session_id


if __name__ == "__main__":
    sys.exit(main())
