#!/usr/bin/env python3
"""OAK 4 D Pro point-cloud obstacle avoidance app with a debug frontend."""

from __future__ import annotations

import json
import os
import queue
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
OBSTACLE_DISTANCE_MM: Final = 1_000
CAUTION_DISTANCE_MM: Final = 2_000
MAX_NAVIGATION_RANGE_MM: Final = 5_500
MIN_LANE_POINTS: Final = 80
FOV_FRACTION: Final = 0.90
LANE_ANGLE_TAN: Final = FOV_FRACTION / 6
FOV_HALF_TAN: Final = FOV_FRACTION / 2
VERTICAL_LIMIT_MM: Final = 1_200
DEFAULT_FRAME_WIDTH: Final = 640
DEFAULT_FRAME_HEIGHT: Final = 400
DEFAULT_STREAM_FPS: Final = 18
DEFAULT_JPEG_QUALITY: Final = 76
DEFAULT_POINT_SAMPLE_LIMIT: Final = 1_200
DEFAULT_TTS_MIN_INTERVAL_SECONDS: Final = 4.0
DEFAULT_TTS_STABLE_FRAMES: Final = 3
DEFAULT_WALL_STOP_FRAMES: Final = 16

FRONTEND_DIR: Final = Path(__file__).resolve().parent / "frontend"

_should_stop = False
_state_lock = threading.Lock()
_latest_state: "NavigationState | None" = None
_latest_frame: bytes | None = None
_pipeline_status = "starting"
_pipeline_error: str | None = None
_cv2: Any | None = None
_tts_events: list[dict[str, object]] = []


def env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    """Read a bounded integer from the environment."""

    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    """Read a bounded float from the environment."""

    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def env_url(name: str) -> str | None:
    """Read a non-empty URL from the environment."""

    value = os.getenv(name, "").strip()
    return value or None


FRAME_SIZE: Final = (
    env_int("DEBUG_FRAME_WIDTH", DEFAULT_FRAME_WIDTH, 320, 1280),
    env_int("DEBUG_FRAME_HEIGHT", DEFAULT_FRAME_HEIGHT, 240, 720),
)
STREAM_FPS: Final = env_int("DEBUG_STREAM_FPS", DEFAULT_STREAM_FPS, 1, 30)
JPEG_QUALITY: Final = env_int("DEBUG_JPEG_QUALITY", DEFAULT_JPEG_QUALITY, 35, 95)
POINT_SAMPLE_LIMIT: Final = env_int("POINT_SAMPLE_LIMIT", DEFAULT_POINT_SAMPLE_LIMIT, 150, 5_000)
VISION_WS_URL: Final = env_url("VISION_WS_URL")
TTS_MIN_INTERVAL_SECONDS: Final = env_float(
    "TTS_MIN_INTERVAL_SECONDS", DEFAULT_TTS_MIN_INTERVAL_SECONDS, 1.0, 30.0
)
TTS_STABLE_FRAMES: Final = env_int("TTS_STABLE_FRAMES", DEFAULT_TTS_STABLE_FRAMES, 1, 20)
WALL_STOP_FRAMES: Final = env_int("WALL_STOP_FRAMES", DEFAULT_WALL_STOP_FRAMES, 4, 120)


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


class TtsNotifier:
    """Notify the vision frontend WebSocket when guidance state changes."""

    def __init__(
        self,
        vision_ws_url: str | None,
        min_interval_seconds: float,
        stable_frames: int,
    ) -> None:
        """Create a notifier for the frontend /api/camera WebSocket."""

        self.vision_ws_url = vision_ws_url
        self.min_interval_seconds = min_interval_seconds
        self.stable_frames = stable_frames
        self._camera_connected_sent = False
        self._candidate_path: str | None = None
        self._candidate_count = 0
        self._last_spoken_path: str | None = None
        self._last_spoken_at = 0.0
        self._ws: Any | None = None
        self._ws_lock = threading.Lock()
        self._listener_started = False
        self._outbox: "queue.Queue[dict[str, object]]" = queue.Queue()

    def notify_camera_connected(self) -> None:
        """Send camera connection state once after the pipeline is running."""

        if self._camera_connected_sent:
            return
        self._camera_connected_sent = True
        payload = {"type": "status", "cameraConnected": True}
        record_tts_event("camera_status", payload)
        self.send(payload)

    def maybe_describe(self, state: NavigationState) -> None:
        """Send a description hint only after a stable recommendation change."""

        path = state.recommended_path
        if path == self._candidate_path:
            self._candidate_count += 1
        else:
            self._candidate_path = path
            self._candidate_count = 1

        now = time.monotonic()
        if self._candidate_count < self.stable_frames:
            return
        if path == self._last_spoken_path:
            return
        if now - self._last_spoken_at < self.min_interval_seconds:
            return

        text = speech_hint(path)
        self._last_spoken_path = path
        self._last_spoken_at = now
        payload = {"type": "description", "text": text}
        record_tts_event("description", payload)
        self.send(payload)

    def send(self, payload: dict[str, object]) -> None:
        """Queue one JSON payload for the /api/camera WebSocket."""

        if not self.vision_ws_url:
            log_json({"event": "vision_ws_skipped", "payload": payload, "reason": "url_not_configured"})
            return

        self._outbox.put(payload)
        self._start_listener()

    def _start_listener(self) -> None:
        """Start the persistent WebSocket loop once."""

        if not self.vision_ws_url or self._listener_started:
            return
        self._listener_started = True
        thread = threading.Thread(target=self._websocket_loop, daemon=True)
        thread.start()

    def _websocket_loop(self) -> None:
        """Keep the WebSocket open for outgoing and incoming messages."""

        while not _should_stop:
            ws = self._ensure_websocket()
            if ws is None:
                time.sleep(2.0)
                continue

            self._drain_outbox(ws)
            try:
                ws.settimeout(0.25)
                incoming = ws.recv()
            except TimeoutError:
                continue
            except BaseException as exc:
                if is_websocket_timeout(exc):
                    continue
                log_json({"event": "vision_ws_receive_failed", "error": repr(exc)})
                with self._ws_lock:
                    self._close_websocket_locked()
                time.sleep(1.0)
                continue

            if incoming:
                self._handle_incoming_message(incoming)

    def _drain_outbox(self, ws: Any) -> None:
        """Send all queued messages on the current WebSocket connection."""

        while not self._outbox.empty():
            payload = self._outbox.get()
            try:
                ws.send(json.dumps(payload, separators=(",", ":")))
                log_json({"event": "vision_ws_sent", "payload": payload})
            except BaseException as exc:
                log_json({"event": "vision_ws_failed", "error": repr(exc), "payload": payload})
                self._outbox.put(payload)
                with self._ws_lock:
                    self._close_websocket_locked()
                return

    def _handle_incoming_message(self, raw_message: object) -> None:
        """Handle frontend requests sent over the camera WebSocket."""

        try:
            if isinstance(raw_message, bytes):
                message = json.loads(raw_message.decode("utf-8"))
            else:
                message = json.loads(str(raw_message))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            log_json({"event": "vision_ws_bad_message", "error": repr(exc), "raw": str(raw_message)})
            return

        log_json({"event": "vision_ws_received", "payload": message})
        if is_description_request(message):
            payload = {"type": "description", "text": describe_surroundings()}
            record_tts_event("description", payload)
            self._outbox.put(payload)

    def _ensure_websocket(self) -> Any | None:
        """Open the WebSocket once and reuse it for later messages."""

        if self._ws is not None and self._ws.connected:
            return self._ws

        with self._ws_lock:
            if self._ws is not None and self._ws.connected:
                return self._ws
            try:
                import websocket

                self._ws = websocket.create_connection(self.vision_ws_url, timeout=2.0)
                log_json({"event": "vision_ws_connected", "url": self.vision_ws_url})
                return self._ws
            except BaseException as exc:
                log_json({"event": "vision_ws_connect_failed", "error": repr(exc), "url": self.vision_ws_url})
                self._ws = None
                return None

    def _close_websocket_locked(self) -> None:
        """Close the current WebSocket connection while holding the lock."""

        if self._ws is None:
            return
        try:
            self._ws.close()
        except BaseException:
            pass
        self._ws = None


def speech_hint(recommended_path: str) -> str:
    """Convert internal path labels into short spoken guidance."""

    hints = {
        "forward": "Path clear. Keep going forward.",
        "forward_slow": "Obstacle ahead. Move forward slowly.",
        "go_left": "Obstacle ahead. Go left.",
        "go_right": "Obstacle ahead. Go right.",
        "stop": "Stop. Obstacle ahead.",
        "slow_or_stop": "Slow down. Path unclear.",
        "scan_left": "Blocked ahead. Turn left slowly and scan for an opening.",
        "scan_right": "Blocked ahead. Turn right slowly and scan for an opening.",
    }
    return hints.get(recommended_path, recommended_path.replace("_", " "))


def record_tts_event(event_type: str, payload: dict[str, object]) -> None:
    """Record a would-send TTS payload for the local viewer."""

    event = {
        "type": event_type,
        "payload": payload,
        "timestamp": time.time(),
    }
    with _state_lock:
        _tts_events.append(event)
        del _tts_events[:-30]
    log_json({"event": "tts_event_recorded", **event})


def is_description_request(message: dict[str, object]) -> bool:
    """Return whether an incoming WebSocket message requests scene description."""

    message_type = str(message.get("type", "")).lower()
    action = str(message.get("action", "")).lower()
    return message_type in {"describe", "description_request", "request_description"} or action in {
        "describe",
        "request_description",
    }


def is_websocket_timeout(exc: BaseException) -> bool:
    """Return whether a WebSocket receive exception is just an idle timeout."""

    return type(exc).__name__ in {"WebSocketTimeoutException", "TimeoutError", "TimeoutException"}


def describe_surroundings() -> str:
    """Return a placeholder scene description until object recognition is wired in."""

    with _state_lock:
        state = _latest_state

    if state is None:
        return "The camera is starting. I do not have a scene description yet."

    nearest = (
        f"{state.nearest_obstacle_mm / 1000:.1f} metres"
        if state.nearest_obstacle_mm is not None
        else "an unknown distance"
    )
    path = state.recommended_path.replace("_", " ")
    return f"Navigation view active. The current recommendation is {path}. The nearest center obstacle is about {nearest} away."


class SearchRouteAdvisor:
    """Escalate repeated stop states into turn-and-scan guidance."""

    def __init__(self, wall_stop_frames: int) -> None:
        """Create a wall/blocked-path advisor."""

        self.wall_stop_frames = wall_stop_frames
        self._stop_count = 0
        self._turn_direction = "left"

    def apply(self, state: NavigationState) -> NavigationState:
        """Return adjusted guidance when the user appears stopped at a wall."""

        if state.recommended_path not in {"stop", "slow_or_stop"}:
            self._stop_count = 0
            return state

        self._stop_count += 1
        if self._stop_count < self.wall_stop_frames:
            return state

        clearance = state.lane_clearance_mm
        left = clearance.get("left") or 0
        right = clearance.get("right") or 0
        if abs(left - right) > 250:
            self._turn_direction = "left" if left > right else "right"

        path = f"scan_{self._turn_direction}"
        return NavigationState(
            label=state.label,
            detected=state.detected,
            recommended_path=path,
            nearest_obstacle_mm=state.nearest_obstacle_mm,
            lane_clearance_mm=state.lane_clearance_mm,
            confidence=state.confidence,
            reason=f"blocked ahead; turn {self._turn_direction} slowly to search for an exit route",
            sample_points=state.sample_points,
            timestamp=state.timestamp,
        )


tts_notifier = TtsNotifier(
    VISION_WS_URL,
    TTS_MIN_INTERVAL_SECONDS,
    TTS_STABLE_FRAMES,
)
search_route_advisor = SearchRouteAdvisor(WALL_STOP_FRAMES)


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
        & (np.abs(x_values) <= z_values * FOV_HALF_TAN)
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
    fov_left_x = int(width * (1 - FOV_FRACTION) / 2)
    fov_right_x = width - fov_left_x
    lane_width = fov_right_x - fov_left_x
    left_x = fov_left_x + lane_width // 3
    right_x = fov_left_x + (lane_width * 2) // 3
    color = (0, 0, 255) if state.detected else (0, 180, 0)

    overlay = frame.copy()
    cv2.rectangle(overlay, (fov_left_x, 0), (fov_right_x, height), (255, 255, 255), -1)
    frame = cv2.addWeighted(overlay, 0.07, frame, 0.93, 0)
    cv2.line(frame, (fov_left_x, 0), (fov_left_x, height), (255, 255, 255), 1)
    cv2.line(frame, (fov_right_x, 0), (fov_right_x, height), (255, 255, 255), 1)
    cv2.line(frame, (left_x, 0), (left_x, height), (255, 255, 255), 1)
    cv2.line(frame, (right_x, 0), (right_x, height), (255, 255, 255), 1)
    cv2.putText(frame, state.label.upper(), (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, state.recommended_path, (18, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, state.reason, (18, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return encoded.tobytes() if success else None


def draw_suggested_path(
    cv2_module: Any,
    frame: np.ndarray,
    recommended_path: str,
    fov_left_x: int,
    left_x: int,
    right_x: int,
    fov_right_x: int,
) -> None:
    """Draw a corridor line through the currently recommended clear path."""

    height, width = frame.shape[:2]
    if recommended_path in {"stop", "slow_or_stop"}:
        cv2_module.line(frame, (width // 2 - 42, height - 88), (width // 2 + 42, height - 28), (0, 0, 255), 6)
        cv2_module.line(frame, (width // 2 + 42, height - 88), (width // 2 - 42, height - 28), (0, 0, 255), 6)
        return

    target_centers = {
        "go_left": (fov_left_x + left_x) // 2,
        "forward": width // 2,
        "forward_slow": width // 2,
        "go_right": (right_x + fov_right_x) // 2,
        "scan_left": fov_left_x,
        "scan_right": fov_right_x,
    }
    target_x = target_centers.get(recommended_path, width // 2)
    path_color = (0, 220, 0)
    if recommended_path == "forward_slow":
        path_color = (0, 200, 255)

    y_values = np.linspace(height - 26, max(46, int(height * 0.24)), 18)
    t_values = np.linspace(0.0, 1.0, len(y_values))
    x_values = (1 - t_values) ** 2 * (width / 2) + (1 - (1 - t_values) ** 2) * target_x
    centerline = np.column_stack([x_values, y_values]).astype(np.int32)

    corridor_width_bottom = max(42, int(width * 0.12))
    corridor_width_top = max(12, int(width * 0.035))
    left_edge = []
    right_edge = []
    for point, progress in zip(centerline, t_values):
        half_width = int((corridor_width_bottom * (1 - progress) + corridor_width_top * progress) / 2)
        left_edge.append([point[0] - half_width, point[1]])
        right_edge.append([point[0] + half_width, point[1]])

    corridor = np.array(left_edge + right_edge[::-1], dtype=np.int32)
    overlay = frame.copy()
    cv2_module.fillPoly(overlay, [corridor], path_color)
    cv2_module.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    cv2_module.polylines(frame, [centerline], False, path_color, 5, lineType=cv2_module.LINE_AA)
    cv2_module.circle(frame, tuple(centerline[-1]), 8, path_color, -1, lineType=cv2_module.LINE_AA)


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
    tts_notifier.maybe_describe(state)
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
            tts_notifier.notify_camera_connected()
            while pipeline.isRunning() and not _should_stop:
                point_message = queues["points"].get()
                state = search_route_advisor.apply(analyze_point_cloud(points_to_array(point_message)))

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

    protocol_version = "HTTP/1.1"

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
            tts_events = list(_tts_events[-12:])
            payload = {
                "pipeline_status": _pipeline_status,
                "pipeline_error": _pipeline_error,
                "state": state,
                "tts_events": tts_events,
                "thresholds": {
                    "obstacle_mm": OBSTACLE_DISTANCE_MM,
                    "caution_mm": CAUTION_DISTANCE_MM,
                    "max_range_mm": MAX_NAVIGATION_RANGE_MM,
                    "fov_fraction": FOV_FRACTION,
                    "lane_angle_tan": LANE_ANGLE_TAN,
                    "fov_half_tan": FOV_HALF_TAN,
                },
                "viewer": {
                    "frame_width": FRAME_SIZE[0],
                    "frame_height": FRAME_SIZE[1],
                    "stream_fps": STREAM_FPS,
                    "jpeg_quality": JPEG_QUALITY,
                    "point_sample_limit": POINT_SAMPLE_LIMIT,
                    "wall_stop_frames": WALL_STOP_FRAMES,
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
