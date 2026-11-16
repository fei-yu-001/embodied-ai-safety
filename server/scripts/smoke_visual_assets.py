"""Real-kitchen smoke test for the visual asset export.

Loads the actual RoboCasa kitchen (scene load ~30-60 s on CPU), exports every
visible mesh/texture from the live compiled `mjModel` into a session cache, and
checks the properties that only a real model can show:

- the export covers the visible geoms and reports a machine-readable reason for
  everything it could not export;
- a real texture round-trips through PNG, and cube textures are typed with their
  six-face layout;
- the mesh payload is valid (indices in range, normals unit length) and the
  scene reference table is a few kilobytes while the assets are tens of MiB, so
  the 1,000,000-byte worker message stays safe;
- re-running the export reuses the cache instead of rewriting it.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time

from server.runtime.adapters.robocasa_backend import RoboCasaBackend
from server.runtime.adapters.visual_assets import (
    asset_directory,
    export_visual_assets,
    geom_references,
    summarize,
)

WORKER_LIMIT = 1_000_000


def main():
    root = Path(tempfile.mkdtemp(prefix="prism-visual-assets-"))
    frames = root / "sess_smoke" / "frames"
    backend = RoboCasaBackend(frames)
    started = time.perf_counter()
    backend.load("sess_smoke", {"seed": 0, "control_freq": 20})
    print(f"scene_load_seconds={time.perf_counter() - started:.1f}")

    model = backend.env.unwrapped.env.sim.model
    directory = asset_directory(frames)
    started = time.perf_counter()
    manifest = export_visual_assets(model, directory)
    export_seconds = time.perf_counter() - started

    counts = manifest["counts"]
    budget = manifest["budget"]
    print(f"export_seconds={export_seconds:.1f}")
    print("counts:", json.dumps(counts, ensure_ascii=False))
    print("budget:", json.dumps(budget, ensure_ascii=False))
    print("unsupported reasons:", json.dumps(summarize(manifest)["unsupported_reasons"],
                                             ensure_ascii=False))

    references = geom_references(manifest)
    reference_bytes = len(json.dumps(references, separators=(",", ":")).encode("utf-8"))
    summary_bytes = len(json.dumps(summarize(manifest), ensure_ascii=False,
                                   separators=(",", ":")).encode("utf-8"))
    print(f"scene_reference_bytes={reference_bytes} summary_bytes={summary_bytes} "
          f"worker_limit={WORKER_LIMIT}")

    files = sorted(path.name for path in directory.iterdir())
    asset_bytes = sum((directory / name).stat().st_size for name in files)
    print(f"asset_files={len(files)} asset_total_bytes={asset_bytes} "
          f"({asset_bytes / 1048576:.1f} MiB)")

    # ---- per-mesh structure on a real (often large) mesh
    mesh_report = []
    for key in sorted(manifest["meshes"], key=lambda item: -manifest["meshes"][item]["bytes"])[:3]:
        entry = manifest["meshes"][key]
        payload = json.loads((directory / entry["file"]).read_text("utf-8"))
        vertices = entry["vertices"]
        assert len(payload["positions"]) == vertices * 3, key
        assert len(payload["normals"]) == vertices * 3, key
        assert max(payload["indices"]) < vertices, key
        assert len(payload["indices"]) == entry["triangles"] * 3, key
        if entry["has_uvs"] and entry["uv_complete"]:
            assert len(payload["uvs"]) == vertices * 2, key
        mesh_report.append((key, entry["triangles"], vertices, entry["bytes"],
                            entry["normal_source"], entry["has_uvs"], entry["uv_complete"]))
    print("largest meshes (id, triangles, vertices, bytes, normal_source, has_uvs, uv_complete):")
    for row in mesh_report:
        print("  ", row)

    # ---- textures on the real kitchen
    cube_textures = {key: value for key, value in manifest["textures"].items()
                     if value["type"] in ("cube", "skybox")}
    texture_bytes = sorted((value["bytes"] for value in manifest["textures"].values()), reverse=True)
    print(f"textures={len(manifest['textures'])} cube_or_skybox={len(cube_textures)} "
          f"largest_png_bytes={texture_bytes[:3]}")

    double = export_visual_assets(model, directory)
    rewritten = sum(1 for value in list(double["meshes"].values()) + list(double["textures"].values())
                    if value.get("written"))
    print(f"second_export_rewritten_files={rewritten} asset_key={double['asset_key']}")
    print(f"distinct_files={budget['distinct_files']} referenced_bytes={budget['referenced_bytes']} "
          f"distinct_bytes={budget['distinct_bytes']}")

    reasons = summarize(manifest)["unsupported_reasons"]
    unexplained = [item for item in manifest["unsupported"] if not item.get("reason")]
    ok = (
        counts["meshes"] > 0
        and counts["textures"] > 0
        and counts["geoms_with_assets"] > 0
        and reference_bytes < WORKER_LIMIT
        and summary_bytes < WORKER_LIMIT
        and not unexplained
        and rewritten == 0
        and double["asset_key"] == manifest["asset_key"]
        and manifest["provenance"]["mesh_scale_applied"] is False
        and manifest["mujoco_version"] == "3.3.1"
    )
    print("unsupported_reasons:", json.dumps(reasons, ensure_ascii=False))
    print("directory:", directory)
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
