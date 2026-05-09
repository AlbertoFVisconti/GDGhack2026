#!/usr/bin/env python3
"""OAK 4 D Pro point-cloud obstacle avoidance app with a debug frontend."""

from __future__ import annotations

import json
import os
import signal
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Final

import depthai as dai
import numpy as np


LEFT_STEREO_SOCKET: Final = dai.CameraBoardSocket.CAM_B
RIGHT_STEREO_SOCKET: Final = dai.CameraBoardSocket.CAM_C

OBSTACLE_LABEL: Final = "obstacle"
MIN_RANGE_MM: Final = 250
OBSTACLE_DISTANCE_MM: Final = 1_500
CAUTION_DISTANCE_MM: Final = 2_200
MAX_NAVIGATION_RANGE_MM: Final = 3_500
MIN_LANE_POINTS: Final = 80
LANE_ANGLE_TAN: Final = 0.18
VERTICAL_LIMIT_MM: Final = 1_200
DEFAULT_FRAME_WIDTH: Final = 640
DEFAULT_FRAME_HEIGHT: Final = 400
DEFAULT_STREAM_FPS: Final = 18
DEFAULT_JPEG_QUALITY: Final = 76
DEFAULT_POINT_SAMPLE_LIMIT: Final = 1_200

FRONTEND_DIR: Final = Path(__file__).resolve().parent / "frontend"

_should_stop = False
_state_lock = threading.Lock()
_latest_state: "NavigationState | None" = None
_latest_frame: bytes | None = None
_pipeline_status = "starting"
_pipeline_error: str | None = None
_cv2: Any | None = None


def env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    """Read a bounded integer from the environment."""

    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


FRAME_SIZE: Final = (
    env_int("DEBUG_FRAME_WIDTH", DEFAULT_FRAME_WIDTH, 320, 1280),
    env_int("DEBUG_FRAME_HEIGHT", DEFAULT_FRAME_HEIGHT, 240, 720),
)
STREAM_FPS: Final = env_int("DEBUG_STREAM_FPS", DEFAULT_STREAM_FPS, 1, 30)
JPEG_QUALITY: Final = env_int("DEBUG_JPEG_QUALITY", DEFAULT_JPEG_QUALITY, 35, 95)
POINT_SAMPLE_LIMIT: Final = env_int("POINT_SAMPLE_LIMIT", DEFAULT_POINT_SAMPLE_LIMIT, 150, 5_000)


@dataclass(frozen=True)
class NavigationState:
    """Navigation result derived from one point-cloud frame."""

    label: str
    detected: bool
    recommended_path: str
    nearest_obstacle_mm: int | None
    lane_clearance_mm: dict[str, int | None]
    confidence: float
    reason: str
    sample_points: list[list[float]]
    timestamp: float


def log_json(payload: dict[str, object]) -> None:
    """Write one structured log line."""

    print(json.dumps(payload, separators=(",", ":")), flush=True)


def handle_shutdown_signal(_signum: int, _frame: object) -> None:
    """Request a clean shutdown when the OAK app container is stopped."""

    global _should_stop
    _should_stop = True


def create_pipeline() -> tuple[dai.Pipeline, dict[str, Any]]:
    """Create a DepthAI v3 stereo depth and point-cloud pipeline."""

    pipeline = dai.Pipeline()
    mono_left = pipeline.create(dai.node.Camera).build(LEFT_STEREO_SOCKET)
    mono_right = pipeline.create(dai.node.Camera).build(RIGHT_STEREO_SOCKET)
    stereo = pipeline.create(dai.node.StereoDepth)
    point_cloud = pipeline.create(dai.node.PointCloud)

    mono_left.requestFullResolutionOutput().link(stereo.left)
    mono_right.requestFullResolutionOutput().link(stereo.right)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.ROBOTICS)
    stereo.setRectification(True)
    stereo.setExtendedDisparity(True)
    stereo.setLeftRightCheck(True)
    stereo.initialConfig.postProcessing.thresholdFilter.minRange = MIN_RANGE_MM
    stereo.initialConfig.postProcessing.thresholdFilter.maxRange = MAX_NAVIGATION_RANGE_MM

    stereo.depth.link(point_cloud.inputDepth)

    queues = {
        "points": point_cloud.outputPointCloud.createOutputQueue(),
        "depth": stereo.depth.createOutputQueue(),
    }
    return pipeline, queues


def points_to_array(point_message: Any) -> np.ndarray:
    """Convert a DepthAI point-cloud message to an Nx3 float array."""

    points = np.asarray(point_message.getPoints(), dtype=np.float32)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    return points.reshape((-1, 3))


def navigation_points(points: np.ndarray) -> np.ndarray:
    """Filter point-cloud data to plausible forward navigation geometry."""

    if points.size == 0:
        return points

    x_values = points[:, 0]
    y_values = points[:, 1]
    z_values = points[:, 2]
    mask = (
        np.isfinite(points).all(axis=1)
        & (z_values >= MIN_RANGE_MM)
        & (z_values <= MAX_NAVIGATION_RANGE_MM)
        & (np.abs(y_values) <= VERTICAL_LIMIT_MM)
        & (np.abs(x_values) <= z_values)
    )
    return points[mask]


def lane_clearance(points: np.ndarray, lane: str) -> int | None:
    """Estimate free distance for one left, center, or right lane."""

    if points.size == 0:
        return None

    x_over_z = points[:, 0] / np.maximum(points[:, 2], 1.0)
    if lane == "left":
        lane_points = points[x_over_z < -LANE_ANGLE_TAN]
    elif lane == "right":
        lane_points = points[x_over_z > LANE_ANGLE_TAN]
    else:
        lane_points = points[np.abs(x_over_z) <= LANE_ANGLE_TAN]

    if lane_points.shape[0] < MIN_LANE_POINTS:
        return None
    return int(np.percentile(lane_points[:, 2], 10))


def choose_path(clearance: dict[str, int | None]) -> tuple[str, str]:
    """Choose a simple path suggestion from lane clearance estimates."""

    center = clearance["center"]
    left = clearance["left"]
    right = clearance["right"]

    if center is None:
        return "slow_or_stop", "center depth is uncertain"
    if center >= CAUTION_DISTANCE_MM:
        return "forward", "center lane is clear"
    if center >= OBSTACLE_DISTANCE_MM:
        return "forward_slow", "object ahead but outside obstacle threshold"

    side_options = {
        lane: distance
        for lane, distance in {"left": left, "right": right}.items()
        if distance is not None and distance > center + 250
    }
    if not side_options:
        return "stop", "obstacle ahead and no safer side lane found"

    best_lane = max(side_options, key=lambda lane: side_options[lane] or 0)
    return f"go_{best_lane}", f"obstacle ahead; {best_lane} lane has more clearance"


def sample_points(points: np.ndarray, max_points: int = POINT_SAMPLE_LIMIT) -> list[list[float]]:
    """Downsample navigation points for browser-side top-down visualization."""

    if points.shape[0] == 0:
        return []

    step = max(1, points.shape[0] // max_points)
    sampled = points[::step][:max_points, [0, 2]]
    sampled[:, 0] = np.round(sampled[:, 0] / 1000, 3)
    sampled[:, 1] = np.round(sampled[:, 1] / 1000, 3)
    return sampled.tolist()


def analyze_point_cloud(points: np.ndarray) -> NavigationState:
    """Identify an obstacle and suggest a coarse path around it."""

    filtered = navigation_points(points)
    clearance = {
        "left": lane_clearance(filtered, "left"),
        "center": lane_clearance(filtered, "center"),
        "right": lane_clearance(filtered, "right"),
    }
    recommended_path, reason = choose_path(clearance)
    center_clearance = clearance["center"]
    detected = center_clearance is not None and center_clearance < OBSTACLE_DISTANCE_MM
    populated_lanes = sum(distance is not None for distance in clearance.values())
    confidence = round(populated_lanes / len(clearance), 2)

    return NavigationState(
        label=OBSTACLE_LABEL,
        detected=detected,
        recommended_path=recommended_path,
        nearest_obstacle_mm=center_clearance,
        lane_clearance_mm=clearance,
        confidence=confidence,
        reason=reason,
        sample_points=sample_points(filtered),
        timestamp=time.time(),
    )


def render_depth_frame(depth_frame: np.ndarray, state: NavigationState) -> bytes | None:
    """Create a JPEG depth visualization with lane and path overlays."""

    cv2 = get_cv2()
    if cv2 is None:
        return None

    valid = np.where(depth_frame > 0, depth_frame, MAX_NAVIGATION_RANGE_MM)
    clipped = np.clip(valid, MIN_RANGE_MM, MAX_NAVIGATION_RANGE_MM)
    normalized = ((MAX_NAVIGATION_RANGE_MM - clipped) * 255 / MAX_NAVIGATION_RANGE_MM).astype(np.uint8)
    resized = cv2.resize(normalized, FRAME_SIZE)
    frame = cv2.applyColorMap(resized, cv2.COLORMAP_TURBO)

    height, width = frame.shape[:2]
    left_x = width // 3
    right_x = (width * 2) // 3
    color = (0, 0, 255) if state.detected else (0, 180, 0)

    cv2.line(frame, (left_x, 0), (left_x, height), (255, 255, 255), 1)
    cv2.line(frame, (right_x, 0), (right_x, height), (255, 255, 255), 1)
    cv2.putText(frame, state.label.upper(), (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, state.recommended_path, (18, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, state.reason, (18, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return encoded.tobytes() if success else None


def get_cv2() -> Any | None:
    """Import OpenCV only when a frame needs JPEG rendering."""

    global _cv2
    if _cv2 is not None:
        return _cv2

    try:
        import cv2
    except BaseException as exc:
        set_pipeline_status("opencv_unavailable", repr(exc))
        return None

    _cv2 = cv2
    return _cv2


def update_runtime_state(state: NavigationState, frame: bytes | None) -> None:
    """Store the latest navigation and visualization outputs."""

    global _latest_frame, _latest_state
    with _state_lock:
        _latest_state = state
        if frame is not None:
            _latest_frame = frame
    log_json(asdict(state))


def set_pipeline_status(status: str, error: str | None = None) -> None:
    """Record the current backend pipeline status for the frontend."""

    global _pipeline_error, _pipeline_status
    with _state_lock:
        _pipeline_status = status
        _pipeline_error = error
    log_json({"event": "pipeline_status", "status": status, "error": error})


def pipeline_worker() -> None:
    """Run DepthAI point-cloud processing in a background thread."""

    try:
        set_pipeline_status("creating")
        pipeline, queues = create_pipeline()
        set_pipeline_status("starting")
        with pipeline:
            pipeline.start()
            set_pipeline_status("running")
            while pipeline.isRunning() and not _should_stop:
                point_message = queues["points"].get()
                state = analyze_point_cloud(points_to_array(point_message))

                depth_message = queues["depth"].tryGet()
                frame = None
                if isinstance(depth_message, dai.ImgFrame):
                    frame = render_depth_frame(depth_message.getFrame(), state)

                update_runtime_state(state, frame)

            set_pipeline_status("stopping")
            pipeline.stop()
            pipeline.wait()
    except BaseException as exc:
        set_pipeline_status("error", repr(exc))


class FrontendHandler(BaseHTTPRequestHandler):
    """Serve static frontend files, status JSON, and MJPEG depth video."""

    def do_GET(self) -> None:
        """Route HTTP GET requests."""

        if self.path in {"/", "/index.html"}:
            self._serve_file(FRONTEND_DIR / "index.html", "text/html")
        elif self.path == "/styles.css":
            self._serve_file(FRONTEND_DIR / "styles.css", "text/css")
        elif self.path == "/app.js":
            self._serve_file(FRONTEND_DIR / "app.js", "application/javascript")
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

        with _state_lock:
            state = asdict(_latest_state) if _latest_state else None
            payload = {
                "pipeline_status": _pipeline_status,
                "pipeline_error": _pipeline_error,
                "state": state,
                "thresholds": {
                    "obstacle_mm": OBSTACLE_DISTANCE_MM,
                    "caution_mm": CAUTION_DISTANCE_MM,
                    "max_range_mm": MAX_NAVIGATION_RANGE_MM,
                },
                "viewer": {
                    "frame_width": FRAME_SIZE[0],
                    "frame_height": FRAME_SIZE[1],
                    "stream_fps": STREAM_FPS,
                    "jpeg_quality": JPEG_QUALITY,
                    "point_sample_limit": POINT_SAMPLE_LIMIT,
                },
            }
        data = json.dumps(payload, separators=(",", ":")).encode()
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
        while not _should_stop:
            with _state_lock:
                frame = _latest_frame
            if frame is None:
                time.sleep(0.1)
                continue
            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
            self.wfile.write(frame)
            self.wfile.write(b"\r\n")
            time.sleep(1 / STREAM_FPS)


def serve_frontend() -> None:
    """Start the frontend server and keep it alive until shutdown."""

    port = int(os.getenv("OAKAPP_STATIC_FRONTEND_PORT", os.getenv("PORT", "8080")))
    server = ThreadingHTTPServer(("0.0.0.0", port), FrontendHandler)
    log_json({"event": "frontend_ready", "url": f"http://0.0.0.0:{port}"})
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()


def main() -> None:
    """Start the frontend and point-cloud processing worker."""

    log_json({"event": "app_boot"})
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)
    worker = threading.Thread(target=pipeline_worker, daemon=True)
    worker.start()
    serve_frontend()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        log_json(
            {
                "event": "fatal_error",
                "error_type": type(exc).__name__,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        raise
