"""Frame concurrency smoke: a reader holding the displayed frame cannot stop publishing.

Reproduces the reported WinError 5 situation with real handles and the real
FrameStore used by the adapter and API:

1. publish frames 1..N,
2. hold the current newest frame open (as a browser/API read would),
3. keep publishing and reading concurrently,
4. confirm every write succeeds, every read is a complete JPEG, and the newest
   frame advances while the held handle is still open,
5. confirm a held `latest.jpg` only yields a warning.

Run with either interpreter; needs no MuJoCo.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.infrastructure.frame_store import FrameStore, looks_like_jpeg  # noqa: E402


def jpeg(tag):
    return b"\xff\xd8" + tag + b"\xff\xd9"


def main():
    root = Path(tempfile.mkdtemp(prefix="prism-frames-smoke-"))
    store = FrameStore(root / "frames")
    report = {"dir": str(root / "frames"), "published": 0, "errors": [], "reads": 0, "torn": 0}

    # 1) 先发布一批，模拟已有历史帧
    for seq in range(1, 6):
        result = store.publish(jpeg(f"pre-{seq}".encode()), seq)
        if result["error"]:
            report["errors"].append(result)
        report["published"] += 1

    newest = store.latest()
    held_path = newest["path"]
    holder = open(held_path, "rb")  # 读者持有正在显示的那张
    held_bytes = holder.read()
    report["held_seq"] = newest["seq"]
    report["held_is_jpeg"] = looks_like_jpeg(held_bytes)

    stop = threading.Event()

    def reader():
        while not stop.is_set():
            resolved = store.latest()
            if resolved is None:
                time.sleep(0.001)
                continue
            try:
                snapshot = store.read_bytes(resolved["path"])
            except OSError as error:
                report["errors"].append({"read": str(error)})
                continue
            report["reads"] += 1
            if not looks_like_jpeg(snapshot):
                report["torn"] += 1

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        # 2) 读者仍持有旧帧时持续发布新帧
        for seq in range(6, 60):
            result = store.publish(jpeg(f"live-{seq}".encode()), seq)
            if result["error"]:
                report["errors"].append(result)
            else:
                report["published"] += 1
    finally:
        stop.set()
        thread.join()

    # 3) 旧帧内容对持有者不变（不可变）
    holder.seek(0)
    report["held_unchanged"] = holder.read() == held_bytes
    holder.close()

    report["final_latest_seq"] = (store.latest() or {}).get("seq")
    report["advanced_while_held"] = report["final_latest_seq"] > report["held_seq"]

    # 4) latest.jpg 被占用只产生 warning
    holder = open(store.frame_dir / "latest.jpg", "rb+")
    try:
        warn = store.publish(jpeg(b"compat"), 60)
    finally:
        holder.close()
    report["compat_warning"] = warn["warning"] is not None
    report["compat_error"] = warn["error"]
    report["compat_status"] = warn["status"]

    print(json.dumps(report, ensure_ascii=False, indent=2))
    ok = (
        not report["errors"]
        and report["torn"] == 0
        and report["held_unchanged"]
        and report["advanced_while_held"]
        and report["compat_error"] is None
        and report["compat_warning"]
    )
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
