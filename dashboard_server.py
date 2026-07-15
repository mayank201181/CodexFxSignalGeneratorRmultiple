"""Local server for the FX dashboard with a Bloomberg refresh endpoint."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from datetime import datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
REFRESH_LOCK = threading.Lock()


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(RESULTS_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self) -> None:
        if self.path != "/refresh":
            self.send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Unknown endpoint"})
            return

        if not REFRESH_LOCK.acquire(blocking=False):
            self.send_json(
                HTTPStatus.CONFLICT,
                {"ok": False, "error": "A refresh is already running"},
            )
            return

        try:
            completed = self.run_refresh()
            self.send_json(HTTPStatus.OK, {"ok": True, "completed_at": completed})
        except Exception as exc:
            self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(exc)})
        finally:
            REFRESH_LOCK.release()

    def run_refresh(self) -> str:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stdout_path = RESULTS_DIR / "button_refresh_stdout.log"
        stderr_path = RESULTS_DIR / "button_refresh_stderr.log"
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            proc = subprocess.run(
                [sys.executable, "run_fx_dashboard.py", "--refresh"],
                cwd=ROOT,
                env=env,
                stdout=stdout,
                stderr=stderr,
                text=True,
                check=False,
            )
        if proc.returncode != 0:
            detail = stderr_path.read_text(encoding="utf-8", errors="replace")[-1200:]
            raise RuntimeError(f"Refresh failed with exit code {proc.returncode}: {detail}")
        return datetime.now().strftime("%Y-%m-%d %H:%M")

    def send_json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Serving FX dashboard on http://{args.host}:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
