"""
Minimal local dashboard server (no extra dependencies): serves a static
cockpit-style status page and a JSON status endpoint that reads a session
directory. Purely observational — it never writes into the session
directory (REQ-57).
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .status import read_status

STATIC_DIR = Path(__file__).parent / "static"

_STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
}


def make_handler(session_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # keep the console quiet; REQ-39 logging is the recorder's job, not this viewer's

        def do_GET(self):
            if self.path == "/api/status":
                self._send_json(read_status(session_dir))
                return
            entry = _STATIC_FILES.get(self.path)
            if entry:
                filename, content_type = entry
                data = (STATIC_DIR / filename).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_response(404)
            self.end_headers()

        def _send_json(self, payload: dict):
            data = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

    return Handler


def serve(session_dir: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(Path(session_dir)))
    print(f"CLASSISM dashboard: http://{host}:{port}  (session: {session_dir})")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
