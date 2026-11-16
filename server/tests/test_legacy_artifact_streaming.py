"""End-to-end download behavior over a real Uvicorn process and real sockets."""

import hashlib
import http.client
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.parse import quote, unquote

try:
    import psutil
except ImportError:
    psutil = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOKEN = "stream-token"
RUN_ID = "run_stream"
CHUNK = 16 * 1024
LARGE_MB = 48


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def patterned_file(path, size):
    block = bytes(range(256))
    with path.open("wb") as handle:
        remaining = size
        while remaining:
            piece = (block * min(1024, remaining // 256 + 1))[:remaining]
            handle.write(piece)
            remaining -= len(piece)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for piece in iter(lambda: handle.read(1 << 20), b""):
            digest.update(piece)
    return digest.hexdigest()


class RealServer:
    """启动一个真实的 Uvicorn 子进程，并用真实 socket 访问它。"""

    def __init__(self, runs_dir, output_dir, session_dir, log_path):
        self.runs_dir = runs_dir
        self.output_dir = output_dir
        self.session_dir = session_dir
        self.log_path = log_path
        self.process = None
        self.port = None
        self._log = None

    def start(self, attempts=3):
        last_error = None
        for _ in range(attempts):
            self.port = free_port()
            self._log = self.log_path.open("wb")
            environment = dict(os.environ)
            environment.update(
                {
                    "PRISM_RUNS": str(self.runs_dir),
                    "PRISM_OUT": str(self.output_dir),
                    "PRISM_SESSION_DIR": str(self.session_dir),
                    "PRISM_PORT": str(self.port),
                    "PRISM_TOKEN": TOKEN,
                }
            )
            self.process = subprocess.Popen(
                [sys.executable, "-X", "utf8", "-m", "server.tests.net_probe_app"],
                cwd=str(PROJECT_ROOT),
                env=environment,
                stdout=subprocess.DEVNULL,
                stderr=self._log,
            )
            try:
                self.wait_until_ready()
                return
            except RuntimeError as error:
                last_error = error
                self.stop()
                time.sleep(0.2)
        raise RuntimeError(f"Uvicorn 启动失败: {last_error}\n{self.read_log()}")

    def wait_until_ready(self, timeout=30.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("进程已退出")
            connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=1)
            try:
                connection.request("GET", "/api/v1/health")
                response = connection.getresponse()
                response.read()
                if response.status == 200:
                    return
            except OSError:
                pass
            finally:
                connection.close()
            time.sleep(0.05)
        raise RuntimeError("等待超时")

    def read_log(self):
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def stop(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=10)
            self.process = None
        if self._log is not None:
            self._log.close()
            self._log = None

    def connect(self, timeout=10):
        return http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)

    def request(self, path, headers=None, read=True):
        connection = self.connect()
        try:
            merged = {"X-Console-Token": TOKEN}
            merged.update(headers or {})
            connection.request("GET", path, headers=merged)
            response = connection.getresponse()
            body = response.read() if read else b""
            return response, body
        finally:
            connection.close()


class LegacyArtifactStreamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="prism-stream-")
        cls.root = Path(cls.temporary.name)
        cls.runs_dir = cls.root / "runs"
        cls.output_dir = cls.root / "output"
        cls.session_dir = cls.root / "sessions"
        cls.runs_dir.mkdir()
        cls.output_dir.mkdir()
        cls.session_dir.mkdir()

        cls.run_dir = cls.runs_dir / RUN_ID
        write_json(
            cls.run_dir / "job.json",
            {"id": RUN_ID, "status": "completed", "created_at": "2026-11-13T09:00:00+08:00"},
        )
        write_json(
            cls.run_dir / "summary.json",
            {"experiment": "stream_probe", "created_at": "2026-11-13T09:00:00+08:00"},
        )
        patterned_file(cls.run_dir / "clean.mp4", 256 * 1024)
        (cls.run_dir / "对比 报告1.md").write_text("# 报告", encoding="utf-8")

        cls.server = RealServer(
            cls.runs_dir, cls.output_dir, cls.session_dir, cls.root / "uvicorn.log"
        )
        cls.server.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()
        cls.temporary.cleanup()

    def url(self, name, query=""):
        return f"/api/v1/legacy-runs/{RUN_ID}/artifacts/{name}{query}"

    def test_download_headers_and_full_body(self):
        path = self.run_dir / "clean.mp4"
        size = path.stat().st_size

        response, body = self.server.request(self.url("clean.mp4", "?download=1"))

        self.assertEqual(response.status, 200)
        self.assertEqual(int(response.getheader("content-length")), size)
        self.assertEqual(response.getheader("accept-ranges"), "bytes")
        self.assertEqual(response.getheader("x-content-type-options"), "nosniff")
        self.assertTrue(response.getheader("content-disposition").startswith("attachment"))
        self.assertEqual(len(body), size)
        self.assertEqual(hashlib.sha256(body).hexdigest(), hashlib.sha256(path.read_bytes()).hexdigest())

    def test_inline_response_is_not_an_attachment(self):
        size = (self.run_dir / "clean.mp4").stat().st_size

        response, body = self.server.request(self.url("clean.mp4"))

        self.assertEqual(response.status, 200)
        self.assertEqual(int(response.getheader("content-length")), size)
        self.assertIsNone(response.getheader("content-disposition"))
        self.assertEqual(len(body), size)

    def test_body_is_read_in_bounded_chunks(self):
        size = (self.run_dir / "clean.mp4").stat().st_size
        connection = self.server.connect()
        try:
            connection.request(
                "GET", self.url("clean.mp4", "?download=1"), headers={"X-Console-Token": TOKEN}
            )
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            chunks = []
            while True:
                piece = response.read(CHUNK)
                if not piece:
                    break
                chunks.append(len(piece))
        finally:
            connection.close()

        self.assertGreater(len(chunks), 4)
        self.assertLessEqual(max(chunks), CHUNK)
        self.assertEqual(sum(chunks), size)

    def test_range_request_returns_partial_content(self):
        path = self.run_dir / "clean.mp4"
        size = path.stat().st_size

        response, body = self.server.request(
            self.url("clean.mp4"), headers={"Range": "bytes=100-199"}
        )

        self.assertEqual(response.status, 206)
        self.assertEqual(response.getheader("content-range"), f"bytes 100-199/{size}")
        self.assertEqual(int(response.getheader("content-length")), 100)
        self.assertEqual(body, path.read_bytes()[100:200])

    def test_unsatisfiable_range_returns_416(self):
        size = (self.run_dir / "clean.mp4").stat().st_size

        response, _body = self.server.request(
            self.url("clean.mp4"), headers={"Range": f"bytes={size + 10}-"}
        )

        self.assertEqual(response.status, 416)
        self.assertEqual(response.getheader("content-range"), f"bytes */{size}")

    def test_head_request_is_rejected_because_only_get_is_declared(self):
        connection = self.server.connect()
        try:
            connection.request(
                "HEAD", self.url("clean.mp4", "?download=1"), headers={"X-Console-Token": TOKEN}
            )
            response = connection.getresponse()
            body = response.read()
        finally:
            connection.close()

        self.assertEqual(response.status, 405)
        self.assertEqual(body, b"")

    def test_abrupt_client_disconnect_does_not_break_the_server(self):
        path = self.url("clean.mp4", "?download=1")
        raw = socket.create_connection(("127.0.0.1", self.server.port), timeout=10)
        try:
            raw.sendall(
                (
                    f"GET {path} HTTP/1.1\r\n"
                    "Host: 127.0.0.1\r\n"
                    f"X-Console-Token: {TOKEN}\r\n"
                    "Connection: close\r\n\r\n"
                ).encode("ascii")
            )
            self.assertTrue(raw.recv(8192))
            raw.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        finally:
            raw.close()

        time.sleep(0.3)
        response, body = self.server.request(self.url("clean.mp4"))
        self.assertEqual(response.status, 200)
        self.assertEqual(len(body), (self.run_dir / "clean.mp4").stat().st_size)
        self.assertIsNone(self.server.process.poll(), "服务器在客户端中断后退出")

    def test_non_ascii_filename_uses_rfc5987_disposition(self):
        name = "对比 报告1.md"

        response, body = self.server.request(self.url(quote(name), "?download=1"))

        self.assertEqual(response.status, 200)
        disposition = response.getheader("content-disposition")
        disposition.encode("ascii")
        self.assertIn("filename*=utf-8''", disposition)
        encoded = disposition.split("''", 1)[1]
        self.assertEqual(unquote(encoded), name)
        self.assertEqual(body.decode("utf-8"), "# 报告")

    def test_large_file_is_streamed_without_buffering_it(self):
        if psutil is None:
            self.skipTest("psutil 不可用，无法观测服务端内存")
        size = LARGE_MB * 1024 * 1024
        big = self.run_dir / "large.mp4"
        patterned_file(big, size)
        expected = file_sha256(big)

        try:
            baseline = psutil.Process(self.server.process.pid).memory_info().rss
            connection = self.server.connect(timeout=30)
            peak = baseline
            total = 0
            digest = hashlib.sha256()
            try:
                connection.request(
                    "GET", self.url("large.mp4", "?download=1"), headers={"X-Console-Token": TOKEN}
                )
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(int(response.getheader("content-length")), size)
                while True:
                    piece = response.read(256 * 1024)
                    if not piece:
                        break
                    total += len(piece)
                    digest.update(piece)
                    peak = max(peak, psutil.Process(self.server.process.pid).memory_info().rss)
                    time.sleep(0.004)
            finally:
                connection.close()
        finally:
            big.unlink()

        self.assertEqual(total, size)
        self.assertEqual(digest.hexdigest(), expected, "下载内容与磁盘不一致")
        grown = peak - baseline
        self.assertLess(
            grown,
            size // 2,
            f"服务端内存增长 {grown} 字节，接近文件大小 {size}，疑似整文件读入内存",
        )


if __name__ == "__main__":
    unittest.main()
