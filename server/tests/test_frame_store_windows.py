"""Real Windows share-mode contention, using CreateFileW exactly like the API.

The reviewer reproduced the original bug by opening a file with a share mode
that omits FILE_SHARE_DELETE: Windows then refuses `os.replace` onto that path
with WinError 5. This test uses the same real handle (via ctypes on Windows) to
verify the new publisher, not a mock:

- a reader holding the *old* numbered frame cannot block a new numbered frame,
- a reader holding `latest.jpg` only skips the compatibility copy (a warning),
- once the reader releases, the next publish refreshes `latest.jpg`.

On non-Windows platforms the share-mode denial does not exist, so the handle
tests skip; the pure-Python contention tests in `test_frame_store.py` still run.
"""

import ctypes
from pathlib import Path
import sys
import tempfile
import unittest

from server.infrastructure.frame_store import FrameStore, frame_name

GENERIC_READ = 0x80000000
FILE_SHARE_READ = 0x00000001
# 故意不含 FILE_SHARE_DELETE / FILE_SHARE_WRITE：与 API 打开方式不同的“更强”
# 占用，才能触发出 WinError 5 的替换失败
OPEN_EXISTING = 3
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

IS_WINDOWS = sys.platform == "win32"


def jpeg(payload=b"body"):
    return b"\xff\xd8" + payload + b"\xff\xd9"


if IS_WINDOWS:
    import ctypes.wintypes as wintypes

    KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
    KERNEL32.CreateFileW.restype = wintypes.HANDLE
    KERNEL32.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    ]
    KERNEL32.CloseHandle.argtypes = [wintypes.HANDLE]


class Win32Handle:
    """A real file handle with FILE_SHARE_READ only (no share-delete)."""

    def __init__(self, path):
        self.path = str(path)
        self.handle = None

    def __enter__(self):
        handle = KERNEL32.CreateFileW(
            self.path, GENERIC_READ, FILE_SHARE_READ, None, OPEN_EXISTING, 0, None,
        )
        if handle == INVALID_HANDLE_VALUE:
            raise ctypes.WinError(ctypes.get_last_error())
        self.handle = handle
        return self

    def __exit__(self, *exc):
        if self.handle:
            KERNEL32.CloseHandle(self.handle)
            self.handle = None
        return False


@unittest.skipUnless(IS_WINDOWS, "FILE_SHARE_DELETE contention is Windows-only")
class Win32ContentionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="prism-frames-win-")
        self.dir = Path(self.temporary.name) / "frames"

    def tearDown(self):
        self.temporary.cleanup()

    def test_replace_onto_held_path_really_fails(self):
        """先证明机制真实：持有目标路径时 os.replace 失败（复现 WinError 5）。"""
        self.dir.mkdir(parents=True, exist_ok=True)
        target = self.dir / "latest.jpg"
        target.write_bytes(jpeg(b"old"))
        source = self.dir / "source.jpg"
        source.write_bytes(jpeg(b"new"))
        with Win32Handle(target):
            with self.assertRaises(PermissionError):
                source.replace(target)

    def test_new_numbered_frame_publishes_while_reader_holds_old_one(self):
        store = FrameStore(self.dir)
        store.publish(jpeg(b"old"), 1)
        with Win32Handle(self.dir / frame_name(1)):
            result = store.publish(jpeg(b"new"), 2)
        self.assertIsNone(result["error"], result["error"])
        self.assertEqual(result["status"], "published")
        self.assertEqual((self.dir / frame_name(2)).read_bytes(), jpeg(b"new"))
        self.assertEqual((self.dir / frame_name(1)).read_bytes(), jpeg(b"old"))

    def test_held_latest_only_skips_compat_copy(self):
        store = FrameStore(self.dir)
        store.publish(jpeg(b"first"), 1)
        with Win32Handle(self.dir / "latest.jpg"):
            result = store.publish(jpeg(b"second"), 2)
        self.assertIsNone(result["error"], "编号帧必须发布成功")
        self.assertEqual(result["status"], "published")
        self.assertIsNotNone(result["warning"])
        self.assertFalse(result["latest_written"])
        self.assertEqual(store.latest()["seq"], 2)

    def test_released_handle_lets_latest_refresh_next_publish(self):
        store = FrameStore(self.dir)
        store.publish(jpeg(b"first"), 1)
        with Win32Handle(self.dir / "latest.jpg"):
            store.publish(jpeg(b"second"), 2)
        # 句柄释放后，下一次发布应恢复兼容副本
        third = store.publish(jpeg(b"third"), 3)
        self.assertIsNone(third["warning"])
        self.assertTrue(third["latest_written"])
        self.assertEqual((self.dir / "latest.jpg").read_bytes(), jpeg(b"third"))

    def test_held_duplicate_seq_is_refused_not_overwritten(self):
        store = FrameStore(self.dir)
        store.publish(jpeg(b"held"), 4)
        with Win32Handle(self.dir / frame_name(4)):
            result = store.publish(jpeg(b"other"), 4)
        self.assertEqual(result["status"], "conflict")
        self.assertEqual(result["error"], "frame_seq_conflict")
        self.assertEqual((self.dir / frame_name(4)).read_bytes(), jpeg(b"held"))


if __name__ == "__main__":
    unittest.main()
