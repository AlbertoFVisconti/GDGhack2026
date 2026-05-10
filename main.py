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
from pathlib import Path
from typing import Any, Final

import depthai as dai
import numpy as np

from tts import TtsNotifier
from webapp import WebAppContext, serve_frontend


LEFT_STEREO_SOCKET: Final = dai.CameraBoardSocket.CAM_B
RIGHT_STEREO_SOCKET: Final = dai.CameraBoardSocket.CAM_C
RGB_SOCKET: Final = dai.CameraBoardSocket.CAM_A

OBSTACLE_LABEL: Final = "obstacle"
YOLO_MODEL: Final = "luxonis/yolov10-nano:coco-512x288"
COCO_LABELS: Final = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
MIN_RANGE_MM: Final = 250
OBSTACLE_DISTANCE_MM: Final = 1_000
CAUTION_DISTANCE_MM: Final = 2_000
MAX_NAVIGATION_RANGE_MM: Final = 2_500
MIN_LANE_POINTS: Final = 80
FOV_FRACTION: Final = 1.10
LANE_ANGLE_TAN: Final = FOV_FRACTION / 6
FOV_HALF_TAN: Final = FOV_FRACTION / 2
VERTICAL_LIMIT_MM: Final = 1_800
DEFAULT_FRAME_WIDTH: Final = 800
DEFAULT_FRAME_HEIGHT: Final = 500
DEFAULT_STEREO_WIDTH: Final = 640
DEFAULT_STEREO_HEIGHT: Final = 400
DEFAULT_STREAM_FPS: Final = 18
DEFAULT_JPEG_QUALITY: Final = 76
DEFAULT_POINT_SAMPLE_LIMIT: Final = 2_000
DEFAULT_TTS_MIN_INTERVAL_SECONDS: Final = 8.0
DEFAULT_TTS_STABLE_FRAMES: Final = 3
DEFAULT_WALL_STOP_FRAMES: Final = 16
DEFAULT_YOLO_CONFIDENCE: Final = 0.35
DEFAULT_YOLO_CENTER_FRACTION: Final = 0.40
DEFAULT_OBJECT_NOTIFY_COOLDOWN_SECONDS: Final = 5.0
OBJECT_DEPTH_PERCENTILE: Final = 50
OBJECT_MOTION_WINDOW: Final = 6
OBJECT_MOTION_MM_PER_SECOND: Final = 180

FRONTEND_DIR: Final = Path(__file__).resolve().parent / "frontend"

_should_stop = False
_state_lock = threading.Lock()
_latest_state: "NavigationState | None" = None
_latest_scene_detections: list["ObjectDetection"] = []
_latest_frame: bytes | None = None
_latest_rgb_frame: bytes | None = None
_pipeline_status = "starting"
_pipeline_error: str | None = None
_cv2: Any | None = None
_tts_events: list[dict[str, object]] = []
_last_runtime_log_at = 0.0


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
STEREO_SIZE: Final = (
    env_int("STEREO_FRAME_WIDTH", DEFAULT_STEREO_WIDTH, 128, 1280),
    env_int("STEREO_FRAME_HEIGHT", DEFAULT_STEREO_HEIGHT, 80, 800),
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
YOLO_CONFIDENCE: Final = env_float("YOLO_CONFIDENCE", DEFAULT_YOLO_CONFIDENCE, 0.05, 0.95)
YOLO_CENTER_FRACTION: Final = env_float("YOLO_CENTER_FRACTION", DEFAULT_YOLO_CENTER_FRACTION, 0.15, 1.0)
OBJECT_NOTIFY_COOLDOWN_SECONDS: Final = env_float(
    "OBJECT_NOTIFY_COOLDOWN_SECONDS", DEFAULT_OBJECT_NOTIFY_COOLDOWN_SECONDS, 1.0, 30.0
)
OBJECT_TRIGGER_RANGE_MM: Final = env_int("OBJECT_TRIGGER_RANGE_MM", OBSTACLE_DISTANCE_MM, 500, 10_000)


@dataclass(frozen=True)
class ObjectDetection:
    """YOLO object detection enriched with aligned stereo depth."""

    label: str
    confidence: float
    distance_mm: int | None
    bbox: list[float]
    center: list[float]
    motion: str
    lateral_position: str


@dataclass(frozen=True)
class SpaceEstimate:
    """Coarse estimate of surrounding space shape from the point cloud."""

    kind: str
    width_m: float | None
    depth_m: float | None
    left_edge_m: float | None
    right_edge_m: float | None
    confidence: float
    wall_confidence: float
    description: str


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
    object_detections: list[ObjectDetection]
    space: SpaceEstimate
    timestamp: float


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


def describe_surroundings() -> str:
    """Return a placeholder scene description until object recognition is wired in."""

    with _state_lock:
        state = _latest_state
        scene_detections = list(_latest_scene_detections)

    if state is None and not scene_detections:
        return "The camera is starting. I do not have a scene description yet."
    if state is None:
        return f"I see {describe_detections(scene_detections[:5])}."

    nearest = (
        f"{state.nearest_obstacle_mm / 1000:.1f} metres"
        if state.nearest_obstacle_mm is not None
        else "an unknown distance"
    )
    path = state.recommended_path.replace("_", " ")
    space = state.space.description
    if scene_detections:
        return f"I see {describe_detections(scene_detections[:5])}. The space looks like {space}. The current recommendation is {path}."
    if state.object_detections:
        return f"I see {describe_detections(state.object_detections[:3])}. The space looks like {space}. The current recommendation is {path}."

    return f"Navigation view active. The space looks like {space}. The current recommendation is {path}. The nearest center obstacle is about {nearest} away."


def describe_detections(detections: list[ObjectDetection]) -> str:
    """Format YOLO detections for a spoken scene description."""

    described = []
    for detection in detections:
        distance = (
            f"{detection.distance_mm / 1000:.1f} metres"
            if detection.distance_mm is not None
            else "unknown distance"
        )
        motion = "" if detection.motion in {"unknown", "stationary"} else f", {detection.motion.replace('_', ' ')}"
        described.append(f"{detection.label} {detection.lateral_position} at {distance}{motion}")
    return "; ".join(described)


def object_speech_text(detection: ObjectDetection) -> str:
    """Create a short speech message for one YOLO object detection."""

    distance = (
        f"{detection.distance_mm / 1000:.1f} metres"
        if detection.distance_mm is not None
        else "an unknown distance"
    )
    motion = "" if detection.motion in {"unknown", "stationary"} else f", {detection.motion.replace('_', ' ')}"
    return f"{detection.label} detected {detection.lateral_position}, {distance} away{motion}."


def announce_nearby_objects(state: NavigationState) -> None:
    """Send debounced center-object description messages to the speech frontend."""

    for detection in object_announcement_cooldown.pending_announcements(state.object_detections):
        payload = {"type": "description", "text": object_speech_text(detection)}
        record_tts_event("description", payload)
        tts_notifier.send(payload)
        log_json(
            {
                "event": "object_announcement",
                "label": detection.label,
                "distance_mm": detection.distance_mm,
                "cooldown_seconds": OBJECT_NOTIFY_COOLDOWN_SECONDS,
                "detection_region": "center",
            }
        )


def center_frame_detections(detections: list[ObjectDetection]) -> list[ObjectDetection]:
    """Keep only YOLO detections whose center is in the horizontal navigation band."""

    half_width = YOLO_CENTER_FRACTION / 2
    left = 0.5 - half_width
    right = 0.5 + half_width
    return [detection for detection in detections if left <= detection.center[0] <= right]


def store_scene_detections(detections: list[ObjectDetection]) -> None:
    """Cache full-frame detections for explicit room description requests."""

    with _state_lock:
        _latest_scene_detections.clear()
        _latest_scene_detections.extend(detections[:12])


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
            object_detections=state.object_detections,
            space=state.space,
            timestamp=state.timestamp,
        )


class ObjectMotionTracker:
    """Estimate simple relative motion from per-object distance history."""

    def __init__(self, window_size: int, threshold_mm_per_second: int) -> None:
        """Create an object motion tracker."""

        self.window_size = window_size
        self.threshold_mm_per_second = threshold_mm_per_second
        self._history: dict[str, list[tuple[float, int]]] = {}

    def update(self, detection: ObjectDetection, timestamp: float) -> ObjectDetection:
        """Return a detection with relative motion filled from recent distance trend."""

        if detection.distance_mm is None:
            return detection

        key = self._track_key(detection)
        history = self._history.setdefault(key, [])
        history.append((timestamp, detection.distance_mm))
        del history[:-self.window_size]

        motion = "unknown"
        if len(history) >= 3:
            elapsed = max(0.001, history[-1][0] - history[0][0])
            velocity = (history[-1][1] - history[0][1]) / elapsed
            if velocity < -self.threshold_mm_per_second:
                motion = "approaching"
            elif velocity > self.threshold_mm_per_second:
                motion = "moving_away"
            else:
                motion = "stationary"

        return ObjectDetection(
            label=detection.label,
            confidence=detection.confidence,
            distance_mm=detection.distance_mm,
            bbox=detection.bbox,
            center=detection.center,
            motion=motion,
            lateral_position=detection.lateral_position,
        )

    def _track_key(self, detection: ObjectDetection) -> str:
        """Create a coarse identity key for short-term object motion."""

        x_bin = int(detection.center[0] * 4)
        return f"{detection.label}:{x_bin}"


class ObjectAnnouncementCooldown:
    """Announce objects only when they newly enter the blocking center corridor."""

    def __init__(self, cooldown_seconds: float, trigger_range_mm: int) -> None:
        """Create an object announcement debounce helper."""

        self.cooldown_seconds = cooldown_seconds
        self.trigger_range_mm = trigger_range_mm
        self._last_announced_at: dict[str, float] = {}
        self._active_blocking_labels: set[str] = set()

    def pending_announcements(self, detections: list[ObjectDetection]) -> list[ObjectDetection]:
        """Return newly blocking detections that should be announced now."""

        now = time.time()
        announcements = []
        current_blocking_labels = set()
        for detection in detections:
            if detection.distance_mm is None or detection.distance_mm > self.trigger_range_mm:
                continue
            current_blocking_labels.add(detection.label)
            if detection.label in self._active_blocking_labels:
                continue
            last_announced_at = self._last_announced_at.get(detection.label, 0.0)
            if now - last_announced_at < self.cooldown_seconds:
                continue
            self._last_announced_at[detection.label] = now
            announcements.append(detection)
        self._active_blocking_labels = current_blocking_labels
        return announcements


def log_json(payload: dict[str, object]) -> None:
    """Write one structured log line."""

    print(json.dumps(payload, separators=(",", ":")), flush=True)


tts_notifier = TtsNotifier(
    VISION_WS_URL,
    TTS_MIN_INTERVAL_SECONDS,
    TTS_STABLE_FRAMES,
    record_tts_event,
    describe_surroundings,
    lambda: _should_stop,
    log_json,
)
search_route_advisor = SearchRouteAdvisor(WALL_STOP_FRAMES)
object_motion_tracker = ObjectMotionTracker(OBJECT_MOTION_WINDOW, OBJECT_MOTION_MM_PER_SECOND)
object_announcement_cooldown = ObjectAnnouncementCooldown(
    OBJECT_NOTIFY_COOLDOWN_SECONDS,
    OBJECT_TRIGGER_RANGE_MM,
)


def handle_shutdown_signal(_signum: int, _frame: object) -> None:
    """Request a clean shutdown when the OAK app container is stopped."""

    global _should_stop
    _should_stop = True


def create_pipeline() -> tuple[dai.Pipeline, dict[str, Any]]:
    """Create a DepthAI v3 stereo depth and point-cloud pipeline."""

    pipeline = dai.Pipeline()
    color = pipeline.create(dai.node.Camera).build(RGB_SOCKET)
    mono_left = pipeline.create(dai.node.Camera).build(LEFT_STEREO_SOCKET)
    mono_right = pipeline.create(dai.node.Camera).build(RIGHT_STEREO_SOCKET)
    stereo = pipeline.create(dai.node.StereoDepth)
    point_cloud = pipeline.create(dai.node.PointCloud)
    rgb_output = color.requestOutput(FRAME_SIZE)
    yolo = pipeline.create(dai.node.DetectionNetwork).build(color, dai.NNModelDescription(YOLO_MODEL))

    mono_left.requestOutput(STEREO_SIZE, type=dai.ImgFrame.Type.GRAY8).link(stereo.left)
    mono_right.requestOutput(STEREO_SIZE, type=dai.ImgFrame.Type.GRAY8).link(stereo.right)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
    stereo.setDepthAlign(RGB_SOCKET)
    stereo.setRectification(True)
    stereo.setExtendedDisparity(True)
    stereo.setLeftRightCheck(True)
    stereo.initialConfig.postProcessing.thresholdFilter.minRange = MIN_RANGE_MM
    stereo.initialConfig.postProcessing.thresholdFilter.maxRange = MAX_NAVIGATION_RANGE_MM

    stereo.depth.link(point_cloud.inputDepth)

    queues = {
        "points": point_cloud.outputPointCloud.createOutputQueue(),
        "depth": stereo.depth.createOutputQueue(),
        "yolo": yolo.out.createOutputQueue(),
        "rgb": rgb_output.createOutputQueue(),
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

    if (left is None or left < CAUTION_DISTANCE_MM) and (right is None or right < CAUTION_DISTANCE_MM):
        return "stop", "broad surface or wall ahead; no side lane is safely open"

    side_options = {
        lane: distance
        for lane, distance in {"left": left, "right": right}.items()
        if distance is not None and distance >= CAUTION_DISTANCE_MM and distance > center + 500
    }
    if not side_options:
        return "stop", "object ahead and no safer side lane found"

    best_lane = max(side_options, key=lambda lane: side_options[lane] or 0)
    return f"go_{best_lane}", f"object ahead; {best_lane} lane has more clearance"


def sample_points(points: np.ndarray, max_points: int = POINT_SAMPLE_LIMIT) -> list[list[float]]:
    """Downsample navigation points for browser-side top-down visualization."""

    if points.shape[0] == 0:
        return []

    step = max(1, points.shape[0] // max_points)
    sampled = points[::step][:max_points, [0, 2]]
    sampled[:, 0] = np.round(sampled[:, 0] / 1000, 3)
    sampled[:, 1] = np.round(sampled[:, 1] / 1000, 3)
    return sampled.tolist()


def estimate_space(points: np.ndarray, clearance: dict[str, int | None]) -> SpaceEstimate:
    """Estimate whether the current geometry looks open, narrow, corridor-like, or blocked."""

    if points.shape[0] < MIN_LANE_POINTS:
        return SpaceEstimate(
            kind="unknown",
            width_m=None,
            depth_m=None,
            left_edge_m=None,
            right_edge_m=None,
            confidence=0.0,
            wall_confidence=0.0,
            description="space shape uncertain",
        )

    near_band = points[(points[:, 2] >= MIN_RANGE_MM) & (points[:, 2] <= MAX_NAVIGATION_RANGE_MM)]
    if near_band.shape[0] < MIN_LANE_POINTS:
        near_band = points

    left_edge = float(np.percentile(near_band[:, 0], 5)) / 1000
    right_edge = float(np.percentile(near_band[:, 0], 95)) / 1000
    width = max(0.0, right_edge - left_edge)
    depth = float(np.percentile(near_band[:, 2], 90)) / 1000
    center = clearance.get("center") or 0
    left = clearance.get("left") or 0
    right = clearance.get("right") or 0
    wall_confidence = estimate_wall_confidence(near_band, clearance)

    if wall_confidence >= 0.68:
        kind = "wall"
        description = f"wall ahead; visible span about {width:.1f} metres wide"
    elif center < OBSTACLE_DISTANCE_MM:
        kind = "blocked_or_near_wall"
        description = f"blocked ahead; visible space about {width:.1f} metres wide"
    elif width < 1.2:
        kind = "narrow_passage"
        description = f"narrow passage; visible width about {width:.1f} metres"
    elif left >= CAUTION_DISTANCE_MM and right >= CAUTION_DISTANCE_MM and center >= CAUTION_DISTANCE_MM:
        kind = "open_area"
        description = f"open area; visible space about {width:.1f} metres wide"
    elif abs(left - right) < 450 and center >= CAUTION_DISTANCE_MM and width < 2.6:
        kind = "corridor_like"
        description = f"corridor-like space; visible width about {width:.1f} metres"
    else:
        kind = "partly_open"
        description = f"partly open space; visible width about {width:.1f} metres"

    confidence = round(min(1.0, near_band.shape[0] / max(MIN_LANE_POINTS * 6, 1)), 2)
    return SpaceEstimate(
        kind=kind,
        width_m=round(width, 2),
        depth_m=round(depth, 2),
        left_edge_m=round(left_edge, 2),
        right_edge_m=round(right_edge, 2),
        confidence=confidence,
        wall_confidence=wall_confidence,
        description=description,
    )


def estimate_wall_confidence(points: np.ndarray, clearance: dict[str, int | None]) -> float:
    """Estimate whether near point-cloud geometry is a broad flat wall."""

    center = clearance.get("center")
    if center is None or center >= CAUTION_DISTANCE_MM or points.shape[0] < MIN_LANE_POINTS:
        return 0.0

    front = points[np.abs(points[:, 2] - center) <= 280]
    if front.shape[0] < MIN_LANE_POINTS:
        return 0.0

    front_width = (float(np.percentile(front[:, 0], 95)) - float(np.percentile(front[:, 0], 5))) / 1000
    z_iqr = float(np.percentile(front[:, 2], 75) - np.percentile(front[:, 2], 25))
    density = min(1.0, front.shape[0] / max(MIN_LANE_POINTS * 4, 1))
    width_score = min(1.0, front_width / 1.4)
    flatness_score = max(0.0, 1.0 - z_iqr / 420)
    side_block_score = 0.0
    left = clearance.get("left")
    right = clearance.get("right")
    if (left is None or left < CAUTION_DISTANCE_MM) and (right is None or right < CAUTION_DISTANCE_MM):
        side_block_score = 1.0

    confidence = (width_score * 0.4) + (flatness_score * 0.35) + (density * 0.15) + (side_block_score * 0.1)
    return round(max(0.0, min(1.0, confidence)), 2)


def clearance_text(clearance_mm: int | None) -> str:
    """Format a lane clearance value for guidance text."""

    if clearance_mm is None:
        return "uncertain"
    return f"{clearance_mm / 1000:.1f} metres"


def clearance_guidance(clearance: dict[str, int | None], space: SpaceEstimate) -> str:
    """Describe usable side clearance so turns do not drift into nearby walls."""

    left = clearance_text(clearance.get("left"))
    right = clearance_text(clearance.get("right"))
    return f"left clearance {left}; right clearance {right}"


def object_name(detection: ObjectDetection | None) -> str:
    """Return the recognized object name or a broad-surface fallback."""

    if detection is not None:
        return detection.label
    return "object"


def navigation_reason(
    base_reason: str,
    space: SpaceEstimate,
    clearance: dict[str, int | None],
    nearest_object: ObjectDetection | None,
) -> str:
    """Build user-facing guidance text from geometry and optional YOLO label."""

    name = object_name(nearest_object)
    reason = base_reason.replace("object", name).replace("obstacle", name)
    space_description = space.description
    if nearest_object is not None and space.kind in {"blocked_or_near_wall", "wall"}:
        space_description = (
            f"visible space about {space.width_m:.1f} metres wide"
            if space.width_m is not None
            else "visible space width uncertain"
        )
    if space.kind == "wall" and nearest_object is None:
        reason = "wall ahead"
        space_description = (
            f"visible span about {space.width_m:.1f} metres wide"
            if space.width_m is not None
            else "visible wall span uncertain"
        )
    elif space.kind == "blocked_or_near_wall" and nearest_object is None:
        reason = reason.replace("object", "wall or broad surface").replace("broad surface or wall", "wall or broad surface")
    return f"{reason}; {space_description}; {clearance_guidance(clearance, space)}"


def detection_label(label_index: int) -> str:
    """Return a human label for a YOLO class id."""

    if 0 <= label_index < len(COCO_LABELS):
        return COCO_LABELS[label_index]
    return f"class_{label_index}"


def normalized_bbox(detection: Any) -> list[float]:
    """Read a normalized detection bbox and clamp it to the frame."""

    values = [
        float(getattr(detection, "xmin", 0.0)),
        float(getattr(detection, "ymin", 0.0)),
        float(getattr(detection, "xmax", 1.0)),
        float(getattr(detection, "ymax", 1.0)),
    ]
    xmin, ymin, xmax, ymax = [max(0.0, min(1.0, value)) for value in values]
    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if ymax < ymin:
        ymin, ymax = ymax, ymin
    return [xmin, ymin, xmax, ymax]


def bbox_depth_mm(depth_frame: np.ndarray, bbox: list[float]) -> int | None:
    """Estimate object distance from valid aligned depth pixels inside a YOLO box."""

    height, width = depth_frame.shape[:2]
    xmin, ymin, xmax, ymax = bbox
    x1 = int(xmin * width)
    y1 = int(ymin * height)
    x2 = int(xmax * width)
    y2 = int(ymax * height)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None

    margin_x = max(1, int((x2 - x1) * 0.12))
    margin_y = max(1, int((y2 - y1) * 0.12))
    roi = depth_frame[y1 + margin_y : y2 - margin_y, x1 + margin_x : x2 - margin_x]
    if roi.size == 0:
        roi = depth_frame[y1:y2, x1:x2]

    valid = roi[(roi >= MIN_RANGE_MM) & (roi <= MAX_NAVIGATION_RANGE_MM)]
    if valid.size < 20:
        return None
    return int(np.percentile(valid, OBJECT_DEPTH_PERCENTILE))


def lateral_position(center_x: float) -> str:
    """Map a normalized object center to left, center, or right."""

    if center_x < 0.4:
        return "left"
    if center_x > 0.6:
        return "right"
    return "center"


def yolo_object_detections(yolo_message: Any | None, depth_frame: np.ndarray | None) -> list[ObjectDetection]:
    """Convert YOLO detections into labeled objects with aligned stereo distance."""

    if yolo_message is None or depth_frame is None:
        return []

    detections = []
    now = time.time()
    for detection in getattr(yolo_message, "detections", []):
        confidence = float(getattr(detection, "confidence", 0.0))
        if confidence < YOLO_CONFIDENCE:
            continue

        bbox = normalized_bbox(detection)
        center = [round((bbox[0] + bbox[2]) / 2, 3), round((bbox[1] + bbox[3]) / 2, 3)]
        object_detection = ObjectDetection(
            label=detection_label(int(getattr(detection, "label", -1))),
            confidence=round(confidence, 3),
            distance_mm=bbox_depth_mm(depth_frame, bbox),
            bbox=[round(value, 4) for value in bbox],
            center=center,
            motion="unknown",
            lateral_position=lateral_position(center[0]),
        )
        detections.append(object_motion_tracker.update(object_detection, now))

    detections.sort(key=lambda item: item.distance_mm if item.distance_mm is not None else MAX_NAVIGATION_RANGE_MM + 1)
    return detections[:8]


def analyze_point_cloud(points: np.ndarray, object_detections: list[ObjectDetection] | None = None) -> NavigationState:
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
    space = estimate_space(filtered, clearance)
    detected_objects = object_detections or []
    nearest_object = next((item for item in detected_objects if item.distance_mm is not None), None)
    label = nearest_object.label if nearest_object is not None else OBSTACLE_LABEL
    reason = navigation_reason(reason, space, clearance, nearest_object)
    confidence = decision_confidence(recommended_path, clearance, space)

    return NavigationState(
        label=label,
        detected=detected,
        recommended_path=recommended_path,
        nearest_obstacle_mm=center_clearance,
        lane_clearance_mm=clearance,
        confidence=confidence,
        reason=reason,
        sample_points=sample_points(filtered),
        object_detections=detected_objects,
        space=space,
        timestamp=time.time(),
    )


def decision_confidence(
    recommended_path: str,
    clearance: dict[str, int | None],
    space: SpaceEstimate,
) -> float:
    """Estimate how reliable the current navigation recommendation is."""

    center = clearance.get("center")
    left = clearance.get("left")
    right = clearance.get("right")
    known_lanes = sum(value is not None for value in (left, center, right))
    lane_quality = known_lanes / 3
    if center is None:
        return round(0.2 + lane_quality * 0.25, 2)

    if recommended_path == "forward":
        distance_margin = normalized_margin(center, CAUTION_DISTANCE_MM, MAX_NAVIGATION_RANGE_MM)
        return round(0.58 + distance_margin * 0.3 + lane_quality * 0.08, 2)

    if recommended_path == "forward_slow":
        distance_margin = normalized_margin(center, OBSTACLE_DISTANCE_MM, CAUTION_DISTANCE_MM)
        return round(0.48 + distance_margin * 0.27 + lane_quality * 0.08, 2)

    if recommended_path == "stop":
        if space.kind == "wall":
            return round(0.5 + space.wall_confidence * 0.43, 2)
        blocked_sides = sum(value is not None and value < CAUTION_DISTANCE_MM for value in (left, right))
        return round(0.42 + lane_quality * 0.18 + blocked_sides * 0.12, 2)

    if recommended_path in {"scan_left", "scan_right"}:
        return round(0.58 + space.wall_confidence * 0.25 + lane_quality * 0.08, 2)

    if recommended_path in {"go_left", "go_right"}:
        side = left if recommended_path == "go_left" else right
        if side is None:
            return round(0.35 + lane_quality * 0.15, 2)
        route_margin = normalized_margin(side - center, 500, MAX_NAVIGATION_RANGE_MM)
        side_distance = normalized_margin(side, CAUTION_DISTANCE_MM, MAX_NAVIGATION_RANGE_MM)
        return round(0.48 + route_margin * 0.24 + side_distance * 0.14 + lane_quality * 0.08, 2)

    return round(0.35 + lane_quality * 0.25, 2)


def normalized_margin(value: int | float, low: int | float, high: int | float) -> float:
    """Scale one confidence component to the 0..1 range."""

    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (float(value) - float(low)) / (float(high) - float(low))))


def render_depth_frame(depth_frame: np.ndarray, state: NavigationState) -> bytes | None:
    """Create a JPEG depth visualization with lane and path overlays."""

    cv2 = get_cv2()
    if cv2 is None:
        return None

    valid = np.where(depth_frame > 0, depth_frame, MAX_NAVIGATION_RANGE_MM)
    clipped = np.clip(valid, MIN_RANGE_MM, MAX_NAVIGATION_RANGE_MM)
    scaled = (MAX_NAVIGATION_RANGE_MM - clipped) / max(MAX_NAVIGATION_RANGE_MM - MIN_RANGE_MM, 1)
    normalized = np.clip((scaled**0.65) * 255, 0, 255).astype(np.uint8)
    resized = cv2.resize(normalized, FRAME_SIZE)
    frame = cv2.applyColorMap(resized, cv2.COLORMAP_JET)

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
    draw_yolo_path_boundaries(cv2, frame)
    cv2.putText(frame, state.label.upper(), (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, state.recommended_path, (18, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, state.reason, (18, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    draw_object_detections(cv2, frame, state.object_detections)

    success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return encoded.tobytes() if success else None


def render_rgb_frame(rgb_message: Any | None, state: NavigationState) -> bytes | None:
    """Create a JPEG RGB visualization with the same obstacle overlays."""

    if rgb_message is None:
        return None

    cv2 = get_cv2()
    if cv2 is None:
        return None

    try:
        frame = rgb_message.getCvFrame()
    except BaseException as exc:
        log_json({"event": "rgb_frame_decode_failed", "error": repr(exc), "message_type": type(rgb_message).__name__})
        return None

    if frame.shape[1::-1] != FRAME_SIZE:
        frame = cv2.resize(frame, FRAME_SIZE)

    height, width = frame.shape[:2]
    draw_yolo_path_boundaries(cv2, frame)
    color = (0, 0, 255) if state.detected else (0, 180, 0)
    cv2.putText(frame, state.label.upper(), (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, state.recommended_path, (18, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, state.reason, (18, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    draw_object_detections(cv2, frame, state.object_detections)

    success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return encoded.tobytes() if success else None


def render_waiting_frame(label: str, detail: str) -> bytes | None:
    """Create a small JPEG placeholder for a stream that has no frames yet."""

    cv2 = get_cv2()
    if cv2 is None:
        return None

    frame = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
    frame[:] = (18, 24, 28)
    cv2.putText(frame, label, (28, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (79, 184, 255), 2)
    cv2.putText(frame, detail, (28, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (238, 244, 245), 1)
    cv2.putText(frame, "Check /api/status viewer.rgb_ready and app logs.", (28, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (156, 173, 178), 1)
    success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return encoded.tobytes() if success else None


def draw_yolo_path_boundaries(cv2_module: Any, frame: np.ndarray) -> None:
    """Draw the horizontal center band used for live YOLO obstacle alerts."""

    height, width = frame.shape[:2]
    left_boundary = int(width * (0.5 - YOLO_CENTER_FRACTION / 2))
    right_boundary = int(width * (0.5 + YOLO_CENTER_FRACTION / 2))
    cv2_module.line(frame, (left_boundary, 0), (left_boundary, height), (0, 255, 0), 2)
    cv2_module.line(frame, (right_boundary, 0), (right_boundary, height), (0, 255, 0), 2)


def draw_object_detections(cv2_module: Any, frame: np.ndarray, detections: list[ObjectDetection]) -> None:
    """Draw YOLO object boxes and depth estimates on the debug frame."""

    height, width = frame.shape[:2]
    for detection in detections[:5]:
        xmin, ymin, xmax, ymax = detection.bbox
        x1 = int(xmin * width)
        y1 = int(ymin * height)
        x2 = int(xmax * width)
        y2 = int(ymax * height)
        box_color = (0, 220, 255) if detection.distance_mm is None else (80, 220, 80)
        if detection.distance_mm is not None and detection.distance_mm < OBSTACLE_DISTANCE_MM:
            box_color = (0, 0, 255)

        distance = f"{detection.distance_mm / 1000:.1f}m" if detection.distance_mm is not None else "--"
        label = f"{detection.label} {distance}"
        if detection.motion not in {"unknown", "stationary"}:
            label = f"{label} {detection.motion.replace('_', ' ')}"

        cv2_module.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        label_y = max(20, y1 - 8)
        cv2_module.putText(frame, label, (x1, label_y), cv2_module.FONT_HERSHEY_SIMPLEX, 0.48, box_color, 1)


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


def update_runtime_state(state: NavigationState, depth_frame: bytes | None, rgb_frame: bytes | None) -> None:
    """Store the latest navigation and visualization outputs."""

    global _last_runtime_log_at, _latest_frame, _latest_rgb_frame, _latest_state
    with _state_lock:
        _latest_state = state
        if depth_frame is not None:
            _latest_frame = depth_frame
        if rgb_frame is not None:
            _latest_rgb_frame = rgb_frame
        depth_ready = _latest_frame is not None
        rgb_ready = _latest_rgb_frame is not None
    tts_notifier.maybe_describe(state)
    now = time.time()
    if now - _last_runtime_log_at >= 2.0:
        _last_runtime_log_at = now
        log_json(navigation_log_payload(state, depth_ready, rgb_ready))


def navigation_log_payload(state: NavigationState, depth_ready: bool, rgb_ready: bool) -> dict[str, object]:
    """Build compact runtime logs without dumping point-cloud vectors."""

    return {
        "event": "navigation_state",
        "path": state.recommended_path,
        "detected": state.detected,
        "nearest_mm": state.nearest_obstacle_mm,
        "confidence": state.confidence,
        "reason": state.reason,
        "lane_clearance_mm": state.lane_clearance_mm,
        "objects": [
            {
                "label": detection.label,
                "distance_mm": detection.distance_mm,
                "confidence": detection.confidence,
                "side": detection.lateral_position,
            }
            for detection in state.object_detections[:4]
        ],
        "space": {
            "kind": state.space.kind,
            "width_m": state.space.width_m,
            "wall_confidence": state.space.wall_confidence,
        },
        "frames": {
            "depth_ready": depth_ready,
            "rgb_ready": rgb_ready,
        },
        "sample_points": len(state.sample_points),
    }


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
                depth_message = queues["depth"].tryGet()
                rgb_message = queues["rgb"].tryGet()
                yolo_message = queues["yolo"].tryGet()
                depth_frame = None
                if isinstance(depth_message, dai.ImgFrame):
                    depth_frame = depth_message.getFrame()

                scene_objects = yolo_object_detections(yolo_message, depth_frame)
                store_scene_detections(scene_objects)
                navigation_objects = center_frame_detections(scene_objects)
                state = search_route_advisor.apply(
                    analyze_point_cloud(points_to_array(point_message), navigation_objects)
                )

                rendered_depth = None
                if depth_frame is not None:
                    rendered_depth = render_depth_frame(depth_frame, state)
                rendered_rgb = render_rgb_frame(rgb_message, state)

                update_runtime_state(state, rendered_depth, rendered_rgb)

            set_pipeline_status("stopping")
            pipeline.stop()
            pipeline.wait()
    except BaseException as exc:
        set_pipeline_status("error", repr(exc))


def status_payload() -> dict[str, object]:
    """Build the latest JSON payload served to the local web viewer."""

    with _state_lock:
        state = asdict(_latest_state) if _latest_state else None
        tts_events = list(_tts_events[-12:])
        pipeline_status = _pipeline_status
        pipeline_error = _pipeline_error
        depth_ready = _latest_frame is not None
        rgb_ready = _latest_rgb_frame is not None

    return {
        "pipeline_status": pipeline_status,
        "pipeline_error": pipeline_error,
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
            "stereo_width": STEREO_SIZE[0],
            "stereo_height": STEREO_SIZE[1],
            "stream_fps": STREAM_FPS,
            "stream_modes": ["depth", "rgb"],
            "depth_ready": depth_ready,
            "rgb_ready": rgb_ready,
            "jpeg_quality": JPEG_QUALITY,
            "point_sample_limit": POINT_SAMPLE_LIMIT,
            "wall_stop_frames": WALL_STOP_FRAMES,
            "object_trigger_range_mm": OBJECT_TRIGGER_RANGE_MM,
            "object_notify_cooldown_seconds": OBJECT_NOTIFY_COOLDOWN_SECONDS,
            "yolo_center_fraction": YOLO_CENTER_FRACTION,
            "vertical_limit_mm": VERTICAL_LIMIT_MM,
        },
    }


def latest_frame(stream_name: str) -> bytes | None:
    """Return the most recent rendered JPEG frame for MJPEG streaming."""

    with _state_lock:
        if stream_name == "rgb":
            frame = _latest_rgb_frame
            fallback = "Waiting for RGB camera frames"
        else:
            frame = _latest_frame
            fallback = "Waiting for depth frames"
    if frame is not None:
        return frame
    return render_waiting_frame(f"{stream_name.upper()} stream waiting", fallback)


def should_stop() -> bool:
    """Return whether the app is shutting down."""

    return _should_stop


def webapp_context() -> WebAppContext:
    """Create the callback context used by the frontend HTTP server."""

    return WebAppContext(
        frontend_dir=FRONTEND_DIR,
        get_status_payload=status_payload,
        get_latest_frame=latest_frame,
        should_stop=should_stop,
        log_json=log_json,
        stream_fps=STREAM_FPS,
    )


def main() -> None:
    """Start the frontend and point-cloud processing worker."""

    log_json({"event": "app_boot"})
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)
    worker = threading.Thread(target=pipeline_worker, daemon=True)
    worker.start()
    serve_frontend(webapp_context())


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
