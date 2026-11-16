"""Real-kitchen smoke test for the RoboCasa backend.

Loads the actual MuJoCo kitchen, applies a few control steps, and checks that a
real camera frame reached disk. Slow by nature (scene load ~30-60 s on CPU).
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

from server.runtime.adapters.robocasa_backend import RoboCasaBackend


def main():
    frames = Path(tempfile.mkdtemp(prefix="prism-robocasa-"))
    backend = RoboCasaBackend(frames)
    started = time.perf_counter()
    info = backend.load("sess_real", {"seed": 0, "control_freq": 20})
    load_seconds = time.perf_counter() - started
    print(f"load_seconds={load_seconds:.1f}")
    print("loaded:", {k: info[k] for k in ("env_id", "policy", "policy_uses_vision", "step", "frame_seq")})
    print("task:", info["task_description"])

    for index in range(3):
        result = backend.apply("step")
        print(
            f"step={result['step']} sim_seconds={result['sim_seconds']:.3f} "
            f"reward={result['reward']:.3f} terminated={result['terminated']} "
            f"frame_seq={result['frame_seq']} error={result.get('last_error')}"
        )
        if result.get("rejected"):
            print("REJECTED:", result["rejected"])

    latest = frames / "latest.jpg"
    numbered = sorted(frames.glob("frame_*.jpg"))
    size = latest.stat().st_size if latest.is_file() else 0
    print(f"frames_dir={frames}")
    print(f"latest_bytes={size} numbered_frames={len(numbered)}")

    ok = (
        backend.step == 3
        and info["policy_uses_vision"] is False
        and size > 1000
        and len(numbered) >= 4
        and backend.last_error is None
    )
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
