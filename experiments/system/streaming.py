"""Live JPEG streams from the actual MuJoCo worker, including its paused scene."""

from time import monotonic, sleep

from .jobs import _inside_file, _record, TERMINAL

VIEWS = {
    "orbit": "orbit.jpg", "scene": "scene.jpg", "clean": "clean.jpg", "attacked": "attacked.jpg",
    "defended": "defended.jpg", "comparison": "preview.jpg",
    "left": "left.jpg", "right": "right.jpg", "wrist": "wrist.jpg",
}


def stream_frames(folder, filename):
    """Yield snapshots as MJPEG; state advances only in the simulator process.

    Sending an occasional unchanged frame keeps paused sessions and connection
    teardown responsive without advancing physics or fabricating new frames.
    """
    path = folder / filename
    last_signature = None
    sent_at = 0.0
    started = monotonic()
    while True:
        data = None
        if _inside_file(folder, path):
            try:
                stat = path.stat()
                signature = (stat.st_mtime_ns, stat.st_size)
                if signature != last_signature or monotonic() - sent_at > 1.0:
                    data = path.read_bytes()
                    last_signature = signature
            except OSError:
                pass  # Worker atomically replaces the snapshot between frames.
        if data:
            sent_at = monotonic()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                   + str(len(data)).encode("ascii") + b"\r\n\r\n" + data + b"\r\n")
        status = _record(folder, "job.json").get("status")
        if status in TERMINAL:
            break
        # A broken startup must not leave an image request pending indefinitely.
        if last_signature is None and monotonic() - started > 180:
            break
        sleep(0.08)
