"""HTTP frontend server for the obstacle avoidance viewer."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable


StatusPayloadFn = Callable[[], dict[str, object]]
LatestFrameFn = Callable[[], bytes | None]
ShouldStopFn = Callable[[], bool]
LogFn = Callable[[dict[str, object]], None]


@dataclass(frozen=True)
class WebAppContext:
    """Runtime callbacks and settings needed by the frontend server."""

    frontend_dir: Path
    get_status_payload: StatusPayloadFn
    get_latest_frame: LatestFrameFn
    should_stop: ShouldStopFn
    log_json: LogFn
    stream_fps: int


class FrontendHandler(BaseHTTPRequestHandler):
    """Serve static frontend files, status JSON, and MJPEG depth video."""

    protocol_version = "HTTP/1.1"
    context: WebAppContext

    def do_GET(self) -> None:
        """Route HTTP GET requests."""

        if self.path in {"/", "/index.html"}:
            self._serve_file(self.context.frontend_dir / "index.html", "text/html")
        elif self.path == "/styles.css":
            self._serve_file(self.context.frontend_dir / "styles.css", "text/css")
        elif self.path == "/app.js":
            self._serve_file(self.context.frontend_dir / "app.js", "application/javascript")
        elif self.path == "/api/status":
            self._serve_status()
        elif self.path == "/stream/depth.mjpg":
            self._serve_mjpeg()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, _format: str, *_args: object) -> None:
        """Suppress default HTTP request logs."""

    def _serve_file(self, path: Path, content_type: str) -> None:
        """Serve one static frontend asset."""

        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        payload = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _serve_status(self) -> None:
        """Serve the latest point-cloud navigation state."""

        data = json.dumps(self.context.get_status_payload(), separators=(",", ":")).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_mjpeg(self) -> None:
        """Serve the latest rendered depth frame as MJPEG."""

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        while not self.context.should_stop():
            frame = self.context.get_latest_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
            self.wfile.write(frame)
            self.wfile.write(b"\r\n")
            time.sleep(1 / self.context.stream_fps)


def serve_frontend(context: WebAppContext) -> None:
    """Start the frontend server and keep it alive until shutdown."""

    port = int(os.getenv("OAKAPP_STATIC_FRONTEND_PORT", os.getenv("PORT", "8080")))
    handler = type("ConfiguredFrontendHandler", (FrontendHandler,), {"context": context})
    server = ThreadingHTTPServer(("0.0.0.0", port), handler)
    context.log_json({"event": "frontend_ready", "url": f"http://0.0.0.0:{port}"})
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
