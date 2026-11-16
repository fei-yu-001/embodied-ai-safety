"""Real-kitchen scene edit check: paused-only, validated, and observable."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from server.runtime.adapters.robocasa_backend import RoboCasaBackend


def main():
    backend = RoboCasaBackend(Path(tempfile.mkdtemp(prefix="prism-edit-real-")))
    info = backend.load("sess_edit_real", {"seed": 0})
    print("capabilities:", info["capabilities"])

    scene = backend.scene_state()
    print("objects:", [(item["name"], item["editable"]) for item in scene["tree"]][:4])
    movable = next((item for item in scene["tree"] if "position" in item["editable"]), None)
    if movable is None:
        print("没有可移动对象，不宣称编辑有效")
        return 1

    start = list(movable["position"])
    target = [start[0] + 0.05, start[1], start[2]]
    moved = backend.apply("edit", {"target": movable["name"], "kind": "position", "values": target})
    after = backend.scene_state()
    found = next(item for item in after["tree"] if item["name"] == movable["name"])

    mass_before = found["mass"]
    scaled = backend.apply("edit", {"target": movable["name"], "kind": "mass", "values": 1.5})
    after_mass = next(item for item in backend.scene_state()["tree"] if item["name"] == movable["name"])

    before_refusal = next(item for item in backend.scene_state()["tree"] if item["name"] == movable["name"])
    revision_before = backend.scene_state().get("scene_revision")
    refused = backend.apply("edit", {"target": movable["name"], "kind": "position", "values": [99.0, 0.0, 1.0]})
    after_refusal = backend.scene_state()
    still = next(item for item in after_refusal["tree"] if item["name"] == movable["name"])

    print("position_before:", start)
    print("position_after :", found["position"], "rejected:", moved.get("rejected"))
    print("mass_before:", mass_before, "->", after_mass["mass"], "rejected:", scaled.get("rejected"))
    print("out_of_workspace rejected:", refused.get("rejected"))
    print("scene_revision:", after.get("scene_revision"))
    print("unchanged_after_refusal:", still["position"] == before_refusal["position"])
    print("revision_after_refusal:", after_refusal.get("scene_revision"), "before:", revision_before)

    ok = (
        moved.get("rejected") is None
        and abs(found["position"][0] - target[0]) < 1e-3
        and scaled.get("rejected") is None
        and abs(after_mass["mass"] - 1.5) < 1e-6
        and refused.get("rejected") is not None
        and still["position"] == before_refusal["position"]
        and after_refusal.get("scene_revision") == revision_before
        and after.get("scene_revision", 0) >= 1
    )
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
