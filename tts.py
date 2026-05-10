"""WebSocket notification helpers for the vision frontend."""

from __future__ import annotations

import json
import queue
import threading
import time
from typing import Any, Callable


NavigationStateLike = Any
LogFn = Callable[[dict[str, object]], None]
RecordEventFn = Callable[[str, dict[str, object]], None]
DescribeFn = Callable[[], str]
ShouldStopFn = Callable[[], bool]


class TtsNotifier:
    """Notify the vision frontend WebSocket when guidance state changes."""

    def __init__(
        self,
        vision_ws_url: str | None,
        min_interval_seconds: float,
        stable_frames: int,
        record_event: RecordEventFn,
        describe_surroundings: DescribeFn,
        should_stop: ShouldStopFn,
        log_json: LogFn,
    ) -> None:
        """Create a notifier for the frontend /api/camera WebSocket."""

        self.vision_ws_url = vision_ws_url
        self.min_interval_seconds = min_interval_seconds
        self.stable_frames = stable_frames
        self.record_event = record_event
        self.describe_surroundings = describe_surroundings
        self.should_stop = should_stop
        self.log_json = log_json
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
        self.record_event("camera_status", payload)
        self.send(payload)

    def maybe_describe(self, state: NavigationStateLike) -> None:
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
        self.record_event("description", payload)
        self.send(payload)

    def send(self, payload: dict[str, object]) -> None:
        """Queue one JSON payload for the /api/camera WebSocket."""

        if not self.vision_ws_url:
            self.log_json({"event": "vision_ws_skipped", "payload": payload, "reason": "url_not_configured"})
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

        while not self.should_stop():
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
                self.log_json({"event": "vision_ws_receive_failed", "error": repr(exc)})
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
                self.log_json({"event": "vision_ws_sent", "payload": payload})
            except BaseException as exc:
                self.log_json({"event": "vision_ws_failed", "error": repr(exc), "payload": payload})
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
            self.log_json({"event": "vision_ws_bad_message", "error": repr(exc), "raw": str(raw_message)})
            return

        self.log_json({"event": "vision_ws_received", "payload": message})
        if is_description_request(message):
            payload = {"type": "description", "text": self.describe_surroundings()}
            self.record_event("description", payload)
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
                self.log_json({"event": "vision_ws_connected", "url": self.vision_ws_url})
                return self._ws
            except BaseException as exc:
                self.log_json({"event": "vision_ws_connect_failed", "error": repr(exc), "url": self.vision_ws_url})
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
