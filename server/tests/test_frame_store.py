"""Frame publishing under Windows read/write contention.

The bug these guard against: the original publisher renamed `latest.tmp.jpg`
onto `latest.jpg`, which fails with WinError 5 / WinError 32 on Windows while a
reader (the API, for every browser poll) holds that exact path open. A reader
showing the old frame could therefore break the worker's publish.

The fix is versioned, immutable numbered frames plus a derived "latest": nothing
is ever renamed onto a frame a reader may hold. These tests reproduce the real
contention, not a mock of it, and confirm:

- a reader holding the old file still lets a new frame publish,
- continuous interleaved reads and writes stay consistent,
- a held `latest.jpg` copy degrades to a reported warning, never a false
  failure and never a false "live",
- latest resolution comes from the directory, not a mutable pointer,
- the API reads a byte snapshot and names the frame source.
"""

import os
from pathlib import Path
import tempfile
import threading
import time
import unittest

from server.infrastructure.frame_store import (
    FRAME_PATTERN,
    FrameStore,
    frame_name,
    looks_like_jpeg,
    parse_frame_name,
)


def jpeg(payload=b"body"):
    """A minimal byte string shaped like a JPEG (start + end markers)."""
    return b"\xff\xd8" + payload + b"\xff\xd9"


class FrameStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="prism-frames-")
        self.dir = Path(self.temporary.name) / "frames"

    def tearDown(self):
        self.temporary.cleanup()

    def test_publish_writes_numbered_frame_and_legacy_latest(self):
        store = FrameStore(self.dir)
        result = store.publish(jpeg(b"one"), 1)
        self.assertIsNone(result["error"])
        self.assertTrue((self.dir / frame_name(1)).is_file())
        self.assertTrue((self.dir / "latest.jpg").is_file())
        self.assertEqual((self.dir / frame_name(1)).read_bytes(), jpeg(b"one"))

    def test_publish_rejects_empty_and_reports_error(self):
        store = FrameStore(self.dir)
        empty = store.publish(b"", 1)
        self.assertEqual(empty["error"], "empty_frame")
        self.assertFalse((self.dir / frame_name(1)).exists())

    def test_reader_holding_old_file_does_not_block_new_publish(self):
        """核心回归：读者持有旧编号帧，worker 仍能发布新帧。"""
        store = FrameStore(self.dir)
        store.publish(jpeg(b"old"), 1)
        old_path = self.dir / frame_name(1)
        holder = open(old_path, "rb")
        try:
            self.assertEqual(holder.read(), jpeg(b"old"))
            # 读者持有旧帧时继续发布：必须成功，且不依赖 latest.jpg
            result = store.publish(jpeg(b"new"), 2)
            self.assertIsNone(result["error"], result["error"])
            self.assertEqual((self.dir / frame_name(2)).read_bytes(), jpeg(b"new"))
            # 旧帧内容对读者不变（不可变）
            self.assertEqual(holder.read(), b"")
            holder.seek(0)
            self.assertEqual(holder.read(), jpeg(b"old"))
        finally:
            holder.close()
        self.assertEqual(store.latest()["seq"], 2)

    @unittest.skipUnless(os.name == "nt", "Windows denies replacing an open file")
    def test_held_latest_copy_is_reported_as_warning_not_failure(self):
        """latest.jpg 被占用：编号帧照发，只记 warning，不谎报失败。"""
        store = FrameStore(self.dir)
        store.publish(jpeg(b"first"), 1)
        held = open(self.dir / "latest.jpg", "rb+")
        try:
            result = store.publish(jpeg(b"second"), 2)
        finally:
            held.close()
        self.assertIsNone(result["error"], "编号帧发布不得因 latest.jpg 被占用而失败")
        self.assertEqual(result["status"], "published")
        self.assertFalse(result["latest_written"])
        self.assertIsNotNone(result["warning"])
        self.assertTrue((self.dir / frame_name(2)).is_file())
        # latest 由目录推导，不依赖 latest.jpg 是否更新
        self.assertEqual(store.latest()["seq"], 2)
        self.assertEqual(store.latest()["source"], "numbered")

    def test_duplicate_seq_with_same_bytes_is_idempotent(self):
        store = FrameStore(self.dir)
        first = store.publish(jpeg(b"same"), 5)
        self.assertEqual(first["status"], "published")
        repeat = store.publish(jpeg(b"same"), 5)
        self.assertEqual(repeat["status"], "idempotent")
        self.assertIsNone(repeat["error"])
        self.assertEqual((self.dir / frame_name(5)).read_bytes(), jpeg(b"same"))

    def test_duplicate_seq_with_different_bytes_is_refused_not_overwritten(self):
        store = FrameStore(self.dir)
        store.publish(jpeg(b"original"), 5)
        conflict = store.publish(jpeg(b"different"), 5)
        self.assertEqual(conflict["status"], "conflict")
        self.assertEqual(conflict["error"], "frame_seq_conflict")
        # 已存在的帧绝不被覆盖
        self.assertEqual((self.dir / frame_name(5)).read_bytes(), jpeg(b"original"))

    def test_duplicate_seq_is_refused_while_a_reader_holds_that_seq(self):
        """已存在的同 seq 被读者持有：不得覆盖，返回冲突。"""
        store = FrameStore(self.dir)
        store.publish(jpeg(b"held"), 3)
        held = open(self.dir / frame_name(3), "rb")
        try:
            result = store.publish(jpeg(b"new"), 3)
        finally:
            held.close()
        self.assertEqual(result["error"], "frame_seq_conflict")
        self.assertEqual((self.dir / frame_name(3)).read_bytes(), jpeg(b"held"))

    def test_two_writers_do_not_collide_on_temp_names(self):
        store = FrameStore(self.dir)
        results = []
        errors = []

        def writer(seq):
            try:
                results.append(store.publish(jpeg(f"frame-{seq}".encode()), seq))
            except Exception as error:  # pragma: no cover - failure path
                errors.append(error)

        threads = [threading.Thread(target=writer, args=(seq,)) for seq in range(1, 9)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertTrue(all(row["error"] is None for row in results), results)
        self.assertEqual(len(list(self.dir.glob("frame_*.jpg"))), 8)
        self.assertEqual(store.latest()["seq"], 8)
        # 临时文件不得残留
        self.assertEqual(list(self.dir.glob("*.tmp")), [])

    def test_continuous_interleaved_read_and_write_stays_consistent(self):
        """连续读写：发布 N 帧，每帧发布后立即快照读，内容必须始终完整。"""
        store = FrameStore(self.dir)
        seen = []
        for seq in range(1, 41):
            payload = jpeg(f"frame-{seq}".encode() * (seq % 7 + 1))
            result = store.publish(payload, seq)
            self.assertIsNone(result["error"], result["error"])
            snapshot = store.read_bytes(store.latest()["path"])
            self.assertTrue(looks_like_jpeg(snapshot))
            self.assertEqual(snapshot, payload)
            seen.append(snapshot)
        self.assertEqual(store.latest()["seq"], 40)
        self.assertEqual(len(seen), 40)

    def test_realistic_reader_writer_race(self):
        """真实竞争：一个线程持续读最新帧，另一个持续发布；读到的每张都完整。"""
        store = FrameStore(self.dir)
        stop = threading.Event()
        problems = []
        reads = {"count": 0, "jpeg": 0}

        def reader():
            while not stop.is_set():
                resolved = store.latest()
                if resolved is None:
                    time.sleep(0.001)
                    continue
                try:
                    snapshot = store.read_bytes(resolved["path"])
                except OSError as error:
                    problems.append(f"read failed: {error}")
                    continue
                reads["count"] += 1
                if looks_like_jpeg(snapshot):
                    reads["jpeg"] += 1
                else:
                    problems.append(f"torn read: {snapshot[:12]!r}")

        thread = threading.Thread(target=reader)
        thread.start()
        try:
            for seq in range(1, 121):
                result = store.publish(jpeg(f"r-{seq}".encode()), seq)
                self.assertIsNone(result["error"], result["error"])
        finally:
            stop.set()
            thread.join()
        self.assertEqual(problems, [])
        self.assertGreater(reads["count"], 0)
        self.assertEqual(reads["count"], reads["jpeg"], "每次读取都必须是完整 JPEG")

    def test_retention_bounds_disk_use(self):
        store = FrameStore(self.dir, retain=5)
        for seq in range(1, 21):
            store.publish(jpeg(b"x"), seq)
        kept = sorted(
            seq for seq in (parse_frame_name(p.name) for p in self.dir.glob("frame_*.jpg"))
            if seq is not None
        )
        self.assertEqual(kept, [16, 17, 18, 19, 20])
        self.assertEqual(store.latest()["seq"], 20)

    def test_latest_is_derived_from_directory_not_a_pointer_file(self):
        store = FrameStore(self.dir)
        store.publish(jpeg(b"a"), 1)
        store.publish(jpeg(b"b"), 2)
        # 删掉兼容副本后，latest 仍从编号帧推导
        (self.dir / "latest.jpg").unlink()
        self.assertEqual(store.latest()["seq"], 2)

    def test_legacy_latest_only_session_still_resolves(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "latest.jpg").write_bytes(jpeg(b"legacy"))
        store = FrameStore(self.dir)
        resolved = store.latest()
        self.assertEqual(resolved["source"], "legacy_latest")
        self.assertTrue(resolved["legacy"])
        self.assertIsNone(resolved["seq"])

    def test_describe_by_seq_and_missing(self):
        store = FrameStore(self.dir)
        store.publish(jpeg(b"a"), 1)
        store.publish(jpeg(b"b"), 2)
        self.assertEqual(store.describe(1)["seq"], 1)
        self.assertEqual(store.describe(2)["path"].read_bytes(), jpeg(b"b"))
        self.assertIsNone(store.describe(99))
        self.assertEqual(store.describe(None)["seq"], 2)

    def test_frame_name_and_parse_round_trip(self):
        self.assertEqual(frame_name(7), "frame_000007.jpg")
        self.assertEqual(frame_name(1234567), "frame_1234567.jpg")
        self.assertEqual(parse_frame_name("frame_000007.jpg"), 7)
        # 6 位是最小宽度：更大 seq 的文件名更长，仍要能解析
        self.assertEqual(parse_frame_name("frame_1234567.jpg"), 1234567)
        self.assertEqual(parse_frame_name("frame_999999.jpg"), 999999)
        self.assertIsNone(parse_frame_name("latest.jpg"))
        self.assertIsNone(parse_frame_name("frame_7.jpg"))
        with self.assertRaises(ValueError):
            frame_name(-1)

    def test_large_seq_publishes_and_resolves(self):
        store = FrameStore(self.dir)
        result = store.publish(jpeg(b"big"), 1_000_001)
        self.assertIsNone(result["error"])
        self.assertEqual(result["name"], "frame_1000001.jpg")
        self.assertEqual(store.latest()["seq"], 1_000_001)
        self.assertEqual(store.describe(1_000_001)["seq"], 1_000_001)

    def test_looks_like_jpeg(self):
        self.assertTrue(looks_like_jpeg(jpeg()))
        self.assertFalse(looks_like_jpeg(b"not a jpeg"))
        self.assertFalse(looks_like_jpeg(b"\xff\xd8 truncated"))


if __name__ == "__main__":
    unittest.main()
