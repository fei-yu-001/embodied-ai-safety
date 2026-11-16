"""Real-kitchen jog check: does a 5 cm request move the end effector?

Loads the actual kitchen, reads the end-effector position from the observation,
applies one jog, and reports the measured displacement. Also confirms an
oversized request is refused without advancing physics.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from server.runtime.adapters.robocasa_backend import RoboCasaBackend

EEF_KEY = "state.end_effector_position_relative"


def _eef(backend):
    value = backend.observation.get(EEF_KEY)
    return None if value is None else np.asarray(value, dtype=float).reshape(-1)


def main():
    backend = RoboCasaBackend(Path(tempfile.mkdtemp(prefix="prism-jog-real-")))
    info = backend.load("sess_jog_real", {"seed": 0, "control_freq": 20})
    print("capabilities:", info["capabilities"])

    start = _eef(backend)
    if start is None:
        print("EEF_KEY 不在观测里，无法测量位移；不宣称点动有效")
        return 1

    result = backend.apply("jog", {"delta_m": [0.05, 0.0, 0.0], "repeat": 8})
    after = _eef(backend)
    moved = after - start
    print(f"step={result['step']} rejected={result.get('rejected')} sim_seconds={result['sim_seconds']:.3f}")
    print("delta_requested_m:", [0.05, 0.0, 0.0], "repeat=8")
    print("delta_measured_m :", np.round(moved, 4).tolist())
    print(f"norm_measured={np.linalg.norm(moved):.4f}")

    before_reject = backend.step
    rejected = backend.apply("jog", {"delta_m": [0.5, 0.0, 0.0]})
    print("oversized rejected:", rejected.get("rejected"), "steps_advanced:", backend.step - before_reject)

    ok = (
        result["step"] == 8
        and rejected.get("rejected") is not None
        and backend.step == 8
        and np.linalg.norm(moved) > 0.02
        and abs(moved[0]) > abs(moved[1]) + abs(moved[2])
    )
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
