"""Start the integrated local console on http://127.0.0.1:8765."""

import argparse
from pathlib import Path
import sys
import subprocess
import threading
import webbrowser

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from werkzeug.serving import make_server
from experiments.system.app import create_app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be in 1..65535")
    app = create_app()
    server = make_server("127.0.0.1", args.port, app, threaded=True)
    # Recover stale tasks only after this process owns the bound port. This avoids
    # a second server that fails to bind marking the first server's live task as interrupted.
    manager = app.extensions["run_manager"]
    recovered = manager.recover_interrupted()
    print(f"Embodied Safety console: http://127.0.0.1:{args.port}", flush=True)
    if recovered:
        print(f"Recovered interrupted runs: {len(recovered)}", flush=True)
    if args.open_browser:
        browser_timer = threading.Timer(0.3, webbrowser.open, args=(f"http://127.0.0.1:{args.port}",))
        browser_timer.daemon = True
        browser_timer.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        manager.shutdown()
        worker = manager.process
        if worker is not None:
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.terminate()
                try:
                    worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait(timeout=5)
        server.server_close()
