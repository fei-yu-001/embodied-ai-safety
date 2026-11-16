"""Immutable, versioned camera frames with a contention-free publisher.

Why this exists
---------------
The first implementation published each frame by renaming `latest.tmp.jpg` onto
`latest.jpg`. On Windows a rename *onto* a path fails with `WinError 5` /
`WinError 32` while any other process holds that path open for reading, and the
API reads `latest.jpg` for every browser poll. So a reader displaying the old
frame could make the worker's publish fail.

Design
------
- A frame is written once under its own numbered name `frame_%06d.jpg` and the
  final name is claimed with `os.link`, which is atomic and fails if the name
  already exists. Nothing is ever renamed *onto* a frame a reader may hold, so a
  reader holding an old frame can never block a new publish, and a reader never
  sees a half-written file.
- A duplicate sequence is never silently overwritten: identical bytes are
  idempotent (a retry), different bytes are reported as `frame_seq_conflict`.
- `os.link` gives create-if-absent semantics on both POSIX and Windows, so
  concurrent writers cannot clobber each other's numbered frame.
- "Latest" is *derived*, not stored: it is the highest numbered frame present.
  No mutable pointer file means no shared file for readers and writers to fight
  over, and the worker and the API process stay independent.
- `latest.jpg` is still written for backward compatibility with existing scripts
  and old sessions, but only as a single best-effort copy with no retry, no
  sleep and no fsync (this runs on a 20 Hz path). The numbered frame is the
  source of truth; if the copy cannot be replaced, that is reported as a frame
  *warning*, never as a publish failure and never as a false "live".
- Disk use is bounded: after each publish, numbered frames beyond `retain` are
  removed, best-effort (a held file is retried on a later publish).
- A reader gets a byte snapshot: the file is read fully into memory and the
  handle released before the response is built.

Historical cache compatibility: sessions that only contain a legacy
`latest.jpg` (no numbered frames) are still resolvable.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import uuid

FRAME_NAME = "frame_{seq:06d}.jpg"
# 6 位是最小宽度：seq >= 1_000_000 时文件名会更长，解析必须继续认
FRAME_PATTERN = re.compile(r"^frame_(\d{6,})\.jpg$")
LEGACY_LATEST = "latest.jpg"
DEFAULT_RETAIN = 400
# 兼容副本不重试、不 sleep：20 Hz 路径上不让 latest.jpg 的占用拖慢发布
DEFAULT_LATEST_SINGLE_TRY = 1

JPEG_START = b"\xff\xd8"
JPEG_END = b"\xff\xd9"


def frame_name(seq):
    if type(seq) is not int or seq < 0:
        raise ValueError("frame seq 必须是非负整数")
    return FRAME_NAME.format(seq=seq)


def parse_frame_name(name):
    match = FRAME_PATTERN.match(name)
    return int(match.group(1)) if match else None


def looks_like_jpeg(payload):
    """True when `payload` starts and ends with the JPEG markers."""
    if not isinstance(payload, (bytes, bytearray)) or len(payload) < 4:
        return False
    return payload[:2] == JPEG_START and payload[-2:] == JPEG_END


class FrameStore:
    """Publishes immutable numbered JPEG frames and resolves the newest one.

    Pure filesystem logic: no cv2, no MuJoCo. Safe to call from the worker
    process while another process reads the same directory.
    """

    def __init__(self, frame_dir, retain=DEFAULT_RETAIN, latest_attempts=DEFAULT_LATEST_SINGLE_TRY):
        self.frame_dir = Path(frame_dir)
        self.retain = int(retain)
        self.latest_attempts = int(latest_attempts)

    # ---------------------------------------------------------------- publish
    def publish(self, jpeg_bytes, seq):
        """Write one complete frame. Never overwrites an existing numbered frame.

        Returns:
          status:      "published" | "idempotent" | "conflict" | "error"
          error:       set only for a real failure / conflict on the numbered frame
          warning:     set when the numbered frame is fine but the compat copy
                       could not be replaced (a held `latest.jpg`)
          latest_written: whether the compat copy was refreshed
        """
        result = {"seq": seq, "name": None, "path": None, "status": "error",
                  "error": None, "warning": None, "latest_written": False}
        if not isinstance(jpeg_bytes, (bytes, bytearray)) or len(jpeg_bytes) == 0:
            result["error"] = "empty_frame"
            return result
        try:
            name = frame_name(seq)
        except ValueError:
            result["error"] = "invalid_seq"
            return result

        target = self.frame_dir / name
        payload = bytes(jpeg_bytes)
        result["name"] = name
        result["path"] = str(target)
        try:
            self.frame_dir.mkdir(parents=True, exist_ok=True)
            status = self._claim_numbered(target, name, payload)
        except OSError as error:
            result["error"] = self._format_error(error)
            return result

        if status == "conflict":
            result["status"] = "conflict"
            result["error"] = "frame_seq_conflict"
            return result
        result["status"] = status  # published | idempotent
        # 兼容副本：单次 best-effort，占用只记 warning
        result["warning"] = self._write_latest(payload)
        result["latest_written"] = result["warning"] is None
        self._prune(keep_seq=seq)
        return result

    def _claim_numbered(self, target, base_name, payload):
        """Atomically claim `target` with `payload`, never overwriting."""
        temp = self.frame_dir / f".{base_name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
        try:
            with open(temp, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            # link 是“不存在才成功”的原子操作：不会覆盖已存在的帧，
            # 也不会让读者看到半张图
            try:
                os.link(temp, target)
                return "published"
            except FileExistsError:
                return "idempotent" if self._read(target) == payload else "conflict"
        finally:
            self._remove_quietly(temp)

    def _write_latest(self, payload):
        """Single best-effort `latest.jpg` copy. No retry, no sleep, no fsync.

        Returns an error string when the copy could not be replaced (for example
        a legacy reader holds it). This is a compatibility copy; the numbered
        frame has already been published and is the source of truth.
        """
        target = self.frame_dir / LEGACY_LATEST
        temp = self.frame_dir / f".{LEGACY_LATEST}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
        try:
            with open(temp, "wb") as handle:
                handle.write(payload)
            os.replace(temp, target)
            return None
        except OSError as error:
            return self._format_error(error)
        finally:
            self._remove_quietly(temp)

    def _prune(self, keep_seq):
        """Drop numbered frames older than the retention window, best-effort."""
        low = keep_seq - self.retain + 1
        if low <= 0:
            return
        for seq, path in self._numbered():
            if seq >= low:
                break
            self._remove_quietly(path)

    def _read(self, path):
        try:
            return path.read_bytes()
        except OSError:
            return None

    def _remove_quietly(self, path):
        try:
            path.unlink()
        except OSError:
            # 文件被读者持有或已不存在；下次发布再清，不吞掉其他错误路径
            pass

    @staticmethod
    def _format_error(error):
        return f"{type(error).__name__}:{getattr(error, 'winerror', '')}:{error}"

    # ------------------------------------------------------------------ read
    def _numbered(self):
        """[(seq, path)] sorted ascending, from directory listing only."""
        rows = []
        try:
            entries = list(os.scandir(self.frame_dir))
        except OSError:
            return rows
        for entry in entries:
            seq = parse_frame_name(entry.name)
            if seq is not None:
                try:
                    if entry.is_file():
                        rows.append((seq, self.frame_dir / entry.name))
                except OSError:
                    continue
        rows.sort(key=lambda row: row[0])
        return rows

    def latest(self):
        """Resolve the newest frame without a mutable pointer file.

        Returns `{seq, path, source, legacy}` or None. `source` is `numbered`
        for a published frame and `legacy_latest` for a session that only has
        the old `latest.jpg`.
        """
        rows = self._numbered()
        if rows:
            seq, path = rows[-1]
            return {"seq": seq, "path": path, "source": "numbered", "legacy": False}
        legacy = self.frame_dir / LEGACY_LATEST
        if legacy.is_file():
            return {"seq": None, "path": legacy, "source": "legacy_latest", "legacy": True}
        return None

    def describe(self, seq=None):
        """Path for a requested seq, or the latest when seq is None."""
        if seq is None:
            return self.latest()
        try:
            name = frame_name(seq)
        except ValueError:
            return None
        path = self.frame_dir / name
        if not path.is_file():
            return None
        return {"seq": seq, "path": path, "source": "numbered", "legacy": False}

    @staticmethod
    def read_bytes(path):
        """Byte snapshot: one full read with the handle released immediately."""
        return Path(path).read_bytes()
