"""Prepare real image assets for a memory-constrained software renderer."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
from PIL import Image


def prepare(root, backup, max_side=1024, manifest=None):
    root, backup = Path(root).resolve(), Path(backup).resolve()
    files = [root / name for name in json.loads(Path(manifest).read_text())] if manifest else root.rglob("*")
    changed = 0
    for path in files:
        if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        with Image.open(path) as image:
            if max(image.size) <= max_side:
                continue
            original = backup / path.relative_to(root)
            if not original.exists():
                original.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, original)
            fmt = image.format
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            temporary = path.with_name(path.name + ".prism-tmp")
            image.save(temporary, format=fmt)
        temporary.replace(path)
        changed += 1
    print(json.dumps({"resized": changed, "max_side": max_side, "originals": str(backup)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("backup")
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--manifest")
    args = parser.parse_args()
    if args.max_side < 128:
        parser.error("max-side must be at least 128")
    prepare(args.root, args.backup, args.max_side, args.manifest)
