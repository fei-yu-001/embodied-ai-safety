"""Scene query contract: full geometry once, then poses only.

Measures real encoded bytes so "polling does not resend static geometry" is a
number, not a claim.
"""

from __future__ import annotations

import json
import sys
import tempfile

from server.runtime.adapters.robocasa_backend import RoboCasaBackend


def size(payload):
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def main():
    backend = RoboCasaBackend(tempfile.mkdtemp(prefix="prism-scene-"))
    backend.load("sess_contract", {"seed": 0})

    first = backend.scene_state()
    first_size = size(first)
    key = first["geometry_key"]
    print(f"first call: geometry_included={first['geometry_included']} bytes={first_size}")

    second = backend.scene_state({"have_geometry_key": key})
    second_size = size(second)
    print(f"second call (same key): geometry_included={second['geometry_included']} "
          f"geometry={second['geometry']} bytes={second_size}")

    third = backend.scene_state({"have_geometry_key": key})
    third_size = size(third)
    print(f"third call: bytes={third_size}")

    stale = backend.scene_state({"have_geometry_key": "g_stale_key"})
    stale_size = size(stale)
    print(f"stale key: geometry_included={stale['geometry_included']} bytes={stale_size}")

    poses_only = size({"transforms": second["transforms"]})
    ratio = second_size / first_size
    print(f"payload reduction: {first_size} -> {second_size} ({ratio:.3%} of first)")
    print(f"in-loop cells: eef={len(second['eef']['position'])} base={len(second['base']['position'])} "
          f"joints={len(second['joints'])} transforms={len(second['transforms'])}")
    print(f"overlays source: {second['overlays']['source']}")

    checks = {
        "first_includes_geometry": first["geometry"] is not None and first["geometry_included"] is True,
        "second_omits_geometry": second["geometry"] is None and second["geometry_included"] is False,
        "second_still_has_transforms": bool(second["transforms"]) and len(second["transforms"]) > 0,
        "second_has_key": second["geometry_key"] == key,
        "repeat_is_same_size": second_size == third_size,
        "stale_key_resends": stale["geometry_included"] is True,
        "reduction_below_20_percent": ratio < 0.20,
        "under_worker_limit": second_size < 1_000_000,
        "poses_are_small": poses_only < second_size,
    }
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'} {name}")
    ok = all(checks.values())
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
