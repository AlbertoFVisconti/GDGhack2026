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
from urllib.parse import urlparse


StatusPayloadFn = Callable[[], dict[str, object]]
LatestFrameFn = Callable[[str], bytes | None]
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

        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._serve_file(self.context.frontend_dir / "index.html", "text/html")
        elif path == "/styles.css":
            self._serve_file(self.context.frontend_dir / "styles.css", "text/css")
        elif path == "/app.js":
            self._serve_file(self.context.frontend_dir / "app.js", "application/javascript")
        elif path == "/api/status":
            self._serve_status()
        elif path == "/stream/depth.mjpg":
            self._serve_mjpeg("depth")
        elif path == "/stream/rgb.mjpg":
            self._serve_mjpeg("rgb")
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

    def _serve_mjpeg(self, stream_name: str) -> None:
        """Serve the latest rendered frame as MJPEG."""

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.context.log_json({"event": "stream_opened", "stream": stream_name})
        last_wait_log_at = 0.0
        while not self.context.should_stop():
            frame = self.context.get_latest_frame(stream_name)
            if frame is None:
                now = time.time()
                if now - last_wait_log_at >= 2.0:
                    last_wait_log_at = now
                    self.context.log_json({"event": "stream_waiting_for_frame", "stream": stream_name})
                time.sleep(0.1)
                continue
            try:
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                self.context.log_json({"event": "stream_closed", "stream": stream_name})
                break
            except OSError as exc:
                self.context.log_json({"event": "stream_write_failed", "stream": stream_name, "error": repr(exc)})
                break
            time.sleep(1 / self.context.stream_fps)


def serve_frontend(context: WebAppContext) -> None:
    """Start the frontend server and keep it alive until shutdown."""

    port = int(os.getenv("OAKAPP_STATIC_FRONTEND_PORT", os.getenv("PORT", "8080")))
    bind_host = os.getenv("FRONTEND_BIND_HOST", "0.0.0.0")
    public_host = os.getenv("FRONTEND_PUBLIC_HOST", bind_host)
    public_port = int(os.getenv("FRONTEND_PUBLIC_PORT", str(port)))
    public_url = f"http://{public_host}:{public_port}/"
    handler = type("ConfiguredFrontendHandler", (FrontendHandler,), {"context": context})
    server = ThreadingHTTPServer((bind_host, port), handler)
    context.log_json(
        {
            "event": "frontend_ready",
            "url": public_url,
            "bind_host": bind_host,
            "bind_port": port,
        }
    )
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
