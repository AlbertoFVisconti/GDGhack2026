# OAK 4 D Pro Navigation App Technical Writeup

## Overview

This repository contains a standalone OAK app for the Luxonis OAK 4 D Pro. Its goal is to help a visually impaired user understand whether the path in front of them is clear, blocked, or navigable by moving left or right. The system combines stereo depth, point-cloud analysis, object detection, and WebSocket-based speech events.

At a high level, the application has four parts:

- **DepthAI pipeline** in `main.py`: builds the OAK camera, stereo depth, point cloud, RGB, and YOLO outputs.
- **Navigation engine** in `main.py`: converts point-cloud geometry and YOLO detections into guidance such as `forward`, `go_left`, `go_right`, `stop`, or `scan_left`.
- **Local debug viewer** in `frontend/` served by `webapp.py`: shows video, radar-style point cloud, lane clearances, detected objects, and speech events.
- **External Webapp integration** in `tts.py` and `Webapp/`: sends camera status and spoken guidance through a WebSocket, and responds to user-initiated scene-description requests.

The app is packaged for standalone OAK deployment by `oakapp.toml`, which copies the Python modules and `frontend/` assets into the container and starts `python3 /app/main.py`.

## Runtime Architecture

The OAK app starts from `main.py`. Startup performs two long-running tasks:

1. A frontend HTTP server is started in a background thread using `serve_frontend()`.
2. The DepthAI pipeline worker is started with `pipeline_worker()`.

The pipeline worker owns the camera pipeline and continuously updates shared runtime state. The HTTP server reads that state to serve `/api/status` and the MJPEG streams. The TTS module reads the same navigation state and sends debounced messages to the external Webapp over WebSocket.

```text
OAK cameras
   |
   v
DepthAI pipeline
   |-- RGB camera output -----------> local MJPEG RGB stream
   |-- YOLO DetectionNetwork -------> object labels + boxes
   |-- StereoDepth -----------------> depth frame
   |-- PointCloud ------------------> navigation geometry
                                      |
                                      v
                              NavigationState
                                      |
        +-----------------------------+-----------------------------+
        |                             |                             |
 local debug viewer             TTS WebSocket              scene description
 /api/status + MJPEG            guidance events             request handler
```

## DepthAI Pipeline

The camera pipeline is created in `create_pipeline()`. It uses DepthAI v3 APIs and the OAK 4 D Pro camera sockets:

- `CAM_A`: center RGB camera.
- `CAM_B`: left stereo camera.
- `CAM_C`: right stereo camera.

The main nodes are:

- `Camera` for the RGB stream.
- `Camera` for left and right mono/stereo sensors.
- `StereoDepth` for depth estimation.
- `PointCloud` for 3D geometry.
- `DetectionNetwork` for YOLO object detection.

The stereo cameras are requested at `STEREO_SIZE`, currently `640 x 400`, because RVC4 stereo input widths need to be compatible with the device stride requirements. The debug/RGB video rendering size is separate (`FRAME_SIZE`), currently `800 x 500`.

The stereo node is configured with:

- `FAST_DENSITY` preset, favoring dense depth data.
- Depth alignment to the RGB camera socket.
- Rectification enabled.
- Extended disparity enabled.
- Left-right check enabled.
- A threshold filter from `MIN_RANGE_MM` to `MAX_NAVIGATION_RANGE_MM`.

The pipeline exports four queues:

- `points`: point-cloud messages from `PointCloud`.
- `depth`: depth frames from `StereoDepth`.
- `yolo`: detections from `DetectionNetwork`.
- `rgb`: RGB camera frames for the debug stream.

## Frame Processing Loop

`pipeline_worker()` is the main runtime loop. Each iteration:

1. Blocks for a point-cloud message.
2. Tries to read the latest depth, RGB, and YOLO messages.
3. Converts the point cloud into a NumPy array.
4. Converts YOLO detections into `ObjectDetection` records, enriched with depth.
5. Stores full-frame YOLO detections for room-description requests.
6. Filters live navigation detections to the center horizontal band.
7. Analyzes point-cloud geometry into a `NavigationState`.
8. Renders depth and RGB debug frames when available.
9. Updates shared runtime state.
10. Lets the TTS notifier decide whether a new spoken guidance message should be sent.

The worker logs compact `navigation_state` messages every few seconds rather than dumping the full point cloud. This keeps logs useful during live testing.

## Point-Cloud Navigation

The navigation algorithm treats the point cloud as the primary safety signal. YOLO is useful for labeling objects, but the geometry decides whether the path is physically blocked.

### Filtering

`navigation_points()` removes points that are outside the useful walking volume:

- Too close or too far.
- Too high or too low vertically.
- Outside the approximate forward field of view.
- Non-finite values.

The remaining points are interpreted as obstacles or surfaces in front of the user.

### Lane Clearances

The forward space is split into three angular lanes:

- Left.
- Center.
- Right.

`lane_clearance()` estimates how much free distance exists in each lane by taking a low percentile of depth values. Using a percentile rather than a minimum makes the result less sensitive to isolated noisy points.

The route decision is made by `choose_path()`:

- If the center lane is clear beyond the caution threshold, recommend `forward`.
- If the center lane is partially clear, recommend `forward_slow`.
- If the center is blocked and one side has significantly more clearance, recommend `go_left` or `go_right`.
- If neither side is safe, recommend `stop`.

Repeated `stop` states can escalate through `SearchRouteAdvisor`, which suggests slowly scanning left or right when the camera remains blocked for many frames.

## Space Understanding

The app estimates more than just “blocked” or “not blocked.” `estimate_space()` creates a `SpaceEstimate` that classifies the surrounding geometry as:

- `open_area`
- `corridor_like`
- `narrow_passage`
- `partly_open`
- `blocked_or_near_wall`
- `wall`
- `stairs`
- `unknown`

This classification is used in the debug UI, logs, reasoning text, and speech subject selection.

### Wall Detection

`estimate_wall_confidence()` looks for a broad, dense, relatively flat cluster of points at a similar depth. This helps distinguish a wall or broad surface from a smaller obstacle like a chair or bin.

The wall score considers:

- Horizontal span of the near points.
- Flatness in depth.
- Point density.
- Whether both side lanes are also blocked.

When confidence is high, the app reports a wall-like surface instead of a generic obstacle.

### Stair Detection

`estimate_stairs_confidence()` is a first-pass geometry detector for stairs. It focuses on the center walking corridor and looks for repeated height changes across forward distance bands. When the score is high enough, the app sets:

- `space.kind = "stairs"`
- `label = "stairs"`
- `recommended_path = "stop"`

This is intentionally conservative. The current behavior is to warn and stop rather than attempt to route the user onto or around stairs. The threshold should be tuned on real staircases because stereo depth can vary with lighting, texture, angle, and distance.

## YOLO Object Detection

YOLO runs on the center RGB camera using a Luxonis model description. Detections are converted into `ObjectDetection` records with:

- Class label.
- Confidence.
- Bounding box.
- Approximate lateral position.
- Distance from aligned depth pixels inside the bounding box.
- Simple motion estimate from recent distance history.

The app keeps two detection sets:

- **Full-frame detections** are cached for explicit room-description requests.
- **Center-band detections** are used for live navigation labels, so side objects do not constantly interrupt the user.

This means YOLO can say “chair ahead,” but the point cloud can still warn about an unrecognized obstacle when YOLO misses something.

## Navigation State

The central data model is `NavigationState`. It summarizes one frame of navigation:

- `label`: object or hazard label.
- `detected`: whether something is currently blocking the relevant path.
- `recommended_path`: the action recommendation.
- `nearest_obstacle_mm`: center lane distance estimate.
- `lane_clearance_mm`: left, center, and right clearances.
- `confidence`: route-confidence estimate.
- `reason`: human-readable explanation.
- `sample_points`: downsampled point cloud for the browser radar view.
- `object_detections`: YOLO detections relevant to navigation.
- `space`: coarse space classification.
- `timestamp`: state update time.

The local viewer polls this state through `/api/status`.

## Route Confidence

The displayed confidence is not a neural-network confidence. It is a route-confidence score derived from the navigation decision.

The score considers:

- Whether the center lane has usable depth.
- How far the center lane is beyond the warning thresholds.
- Whether a side route is genuinely clearer than the blocked center.
- Whether lane data is missing.
- Whether wall confidence supports a stop decision.

This is why it can be lower during uncertain geometry even when the point cloud itself is dense.

## Local Debug Viewer

The local debug viewer is served by `webapp.py` from the `frontend/` folder. It is intended for development, demos, and tuning.

It provides:

- A split main view with video on one side and a radar-style top-down point-cloud map on the other.
- A Depth/RGB stream toggle.
- Current guidance recommendation.
- Route confidence.
- Lane clearance meters.
- YOLO object list.
- Recent TTS events.

The HTTP routes are:

- `/`: frontend HTML.
- `/app.js`: frontend logic.
- `/styles.css`: frontend styles.
- `/api/status`: latest navigation state and viewer settings.
- `/stream/depth.mjpg`: rendered depth heatmap stream.
- `/stream/rgb.mjpg`: rendered RGB stream.

The depth stream overlays lane boundaries, YOLO boxes, labels, and path state. The RGB stream uses the camera frame and draws the same kind of debugging overlays. If a stream has no frames yet, the backend can render a placeholder frame to make the problem visible instead of leaving the browser blank.

## External Webapp Integration

The repository also contains `Webapp/Vision-Navigator/Vision-Navigator`, a TypeScript workspace for the user-facing Webapp. Its API specification includes camera status and scene-description operations. In the current architecture, the OAK app talks to that user-facing app through the WebSocket configured by `VISION_WS_URL`.

The Webapp is responsible for reading directions aloud and letting the user request a description of the surrounding room. The camera app supports that flow with three message types:

- Camera status:

```json
{ "type": "status", "cameraConnected": true }
```

- Normal description/guidance:

```json
{ "type": "description", "text": "Stop. Stairs ahead." }
```

- Incoming description request:

```json
{ "type": "request_description" }
```

When the OAK app receives a description request over the WebSocket, `TtsNotifier` calls `describe_surroundings()`. That function uses the cached full-frame YOLO detections plus the current navigation state to generate a short room overview. This is where a stronger scene-recognition model can be integrated later.

## TTS Debouncing

Speech guidance is deliberately debounced. The app should not repeat “stop” every frame while the user is standing still in front of a wall.

`TtsNotifier` tracks a stable guidance signature and only sends a new message when:

- The recommendation has been stable for enough frames.
- The recommendation meaningfully changed.
- The minimum speech interval has elapsed.

For example, small depth jitter while still blocked should not produce repeated warnings. A meaningful change such as `stop` to `go_left`, or `go_left` to `forward`, can produce a new message.

## Deployment Configuration

`oakapp.toml` configures the standalone OAK application:

- Entry point: `python3 /app/main.py`
- Python dependencies: installed from `requirements.txt`
- Copied app files: `main.py`, `tts.py`, `webapp.py`, and `frontend/`
- Runtime environment variables for stream size, FPS, thresholds, TTS timing, public frontend host, and WebSocket URL.

Important environment variables include:

- `DEBUG_STREAM_FPS`
- `DEBUG_FRAME_WIDTH`
- `DEBUG_FRAME_HEIGHT`
- `STEREO_FRAME_WIDTH`
- `STEREO_FRAME_HEIGHT`
- `POINT_SAMPLE_LIMIT`
- `TTS_MIN_INTERVAL_SECONDS`
- `YOLO_CONFIDENCE`
- `YOLO_CENTER_FRACTION`
- `VISION_WS_URL`

## Current Limitations

The current implementation is practical for a hackathon prototype, but it is not a certified mobility aid.

Known limitations:

- Stair detection is geometry-only and requires real-world tuning.
- YOLO labels depend on the model classes; objects outside the model vocabulary become generic geometry hazards.
- Point-cloud quality depends on lighting, surface texture, and stereo visibility.
- The RGB debug stream and the depth stream are for development; the spoken guidance should remain the primary user-facing output.
- Room descriptions are currently based on cached detections and coarse navigation state. A dedicated scene model can make this much richer.

## Extension Points

Good next engineering steps:

- Add a trained class for `stairs`, `trash bin`, and other mobility-relevant hazards.
- Fuse YOLO staircase detection with the current geometric `stairs_confidence`.
- Add IMU/gyro logic to detect whether the user has actually turned after a scan instruction.
- Tune thresholds from real hallway, sidewalk, wall, stair, and clutter recordings.
- Improve `describe_surroundings()` with a richer object-recognition or scene-description model.
- Persist short motion history for non-YOLO obstacles, not only labeled detections.

## Mental Model

The safest way to understand the system is:

- **Point cloud decides whether the path is physically safe.**
- **YOLO gives names to things when it can.**
- **The local viewer explains what the algorithm is seeing.**
- **The external Webapp turns guidance and room descriptions into user-facing speech.**

