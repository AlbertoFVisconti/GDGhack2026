#!/usr/bin/env python3
"""Standalone OAK 4 D Pro point-cloud obstacle avoidance prototype."""

from __future__ import annotations

import json
import signal
import time
from dataclasses import asdict, dataclass
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
REPORT_INTERVAL_SECONDS: Final = 0.5

_should_stop = False


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
    timestamp: float


def log_event(message: str, **fields: object) -> None:
    """Write one structured diagnostic log line."""

    print(json.dumps({"event": message, **fields}, separators=(",", ":")), flush=True)


def handle_shutdown_signal(_signum: int, _frame: object) -> None:
    """Request a clean shutdown when the OAK app container is stopped."""

    global _should_stop
    _should_stop = True


def create_pipeline() -> tuple[dai.Pipeline, Any]:
    """Create a DepthAI v3 stereo depth and point-cloud pipeline."""

    log_event("creating_pipeline")
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
    point_queue = point_cloud.outputPointCloud.createOutputQueue()

    log_event("pipeline_created")
    return pipeline, point_queue


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
        timestamp=time.time(),
    )


def print_state(state: NavigationState) -> None:
    """Write the current navigation state as one JSON log line."""

    print(json.dumps(asdict(state), separators=(",", ":")), flush=True)


def run() -> None:
    """Run point-cloud obstacle avoidance until the app is stopped."""

    log_event("starting_app")
    pipeline, point_queue = create_pipeline()
    last_report = 0.0

    with pipeline:
        log_event("starting_pipeline")
        pipeline.start()
        log_event("pipeline_started")

        while pipeline.isRunning() and not _should_stop:
            point_message = point_queue.get()
            state = analyze_point_cloud(points_to_array(point_message))

            now = time.monotonic()
            if now - last_report >= REPORT_INTERVAL_SECONDS:
                print_state(state)
                last_report = now

        log_event("stopping_pipeline")
        pipeline.stop()
        pipeline.wait()


def main() -> None:
    """Configure shutdown handling and start the navigation app."""

    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)
    run()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        log_event("fatal_error", error=repr(exc), error_type=type(exc).__name__)
        raise
