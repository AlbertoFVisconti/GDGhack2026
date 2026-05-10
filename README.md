# GDGhack2026 OAK Pathfinder

Standalone navigation prototype for the Luxonis OAK 4 D Pro. The app uses stereo depth, point-cloud geometry, YOLO object detection, and WebSocket speech messages to warn a visually impaired user about obstacles and suggest a simple path direction.

## What It Does

- Detects blocked paths from stereo point-cloud data.
- Suggests `forward`, `forward_slow`, `go_left`, `go_right`, `stop`, or scan guidance.
- Uses YOLO to label recognized objects when available.
- Falls back to geometry-only warnings for unrecognized obstacles.
- Estimates broad walls, corridor-like spaces, narrow passages, and stair-like geometry.
- Serves a local debug viewer with video, radar, lane clearances, detections, and TTS events.
- Sends camera status and spoken guidance to the external Webapp over WebSocket.
- Responds to scene-description requests from the Webapp.

## Repo Map

- `main.py` - DepthAI pipeline, point-cloud navigation, object fusion, rendering, and app runtime.
- `tts.py` - WebSocket connection to the speech/Webapp side and debounced guidance messages.
- `webapp.py` - Local HTTP server for the OAK-hosted debug viewer.
- `frontend/` - Debug viewer UI served from the OAK app.
- `Webapp/` - External user-facing webapp workspace that consumes directions and can request room descriptions.
- `oakapp.toml` - OAK standalone app configuration.
- `requirements.txt` - Python runtime dependencies.
- `TECHNICAL_WRITEUP.md` - More detailed architecture and pipeline documentation.

## Run Locally

Install dependencies in your Python environment:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
python main.py
```

The local debug viewer is served by the app. In OAK app mode, `oakctl` will usually print the forwarded frontend URL.

## Run On OAK

Use Luxonis `oakctl` from the repo root:

```bash
oakctl app run .
```

The app is configured by `oakapp.toml`. Key environment values include stream size/FPS, YOLO confidence, TTS debounce timing, frontend host/port, and `VISION_WS_URL`.

## Debug Viewer

The OAK-hosted debug viewer shows:

- Video stream with Depth/RGB toggle.
- Top-down radar point cloud.
- Current guidance recommendation.
- Lane clearance meters.
- Route confidence.
- YOLO detections.
- Recent text-to-speech events.

## Notes

This is a hackathon prototype, not a certified mobility aid. Geometry thresholds, stair detection, and object labels should be tuned with real-world recordings before relying on the output.

For implementation details, see [TECHNICAL_WRITEUP.md](TECHNICAL_WRITEUP.md).

