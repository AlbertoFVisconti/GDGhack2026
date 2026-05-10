import { useEffect, useRef, useState, useCallback } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { getGetCameraStatusQueryKey } from "@workspace/api-client-react";

export type LogMessage = {
  id: string;
  timestamp: Date;
  type: "path_blocked" | "description" | "status" | "error";
  text: string;
  isUrgent?: boolean;
};

type CameraObject = { label: string; distance: string };

type WsMessage =
  | { type: "path_blocked"; objects: CameraObject[] }
  | { type: "description"; text: string }
  | { type: "status"; cameraConnected: boolean }
  | { type: "error"; message: string };

function formatDistance(raw: string): string {
  return raw.replace(/(\d+(?:\.\d+)?)\s*m\b/gi, (_, n) => `${n} metre${Number(n) === 1 ? "" : "s"}`);
}

function formatObjectSpeech(obj: CameraObject): string {
  const label = obj.label.charAt(0).toUpperCase() + obj.label.slice(1).toLowerCase();
  const distance = formatDistance(obj.distance);
  return `${label} ${distance} ahead of you.`;
}

export function useVisionWs(onSpeak: (text: string, urgent: boolean) => void) {
  const [messages, setMessages] = useState<LogMessage[]>([]);
  const [cameraConnected, setCameraConnected] = useState<boolean | null>(null);
  const [wsStatus, setWsStatus] = useState<"connecting" | "connected" | "disconnected">("connecting");
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const queryClient = useQueryClient();

  const addMessage = useCallback((msg: Omit<LogMessage, "id" | "timestamp">) => {
    setMessages((prev) => [
      {
        ...msg,
        id: Math.random().toString(36).substring(7),
        timestamp: new Date(),
      },
      ...prev,
    ]);
  }, []);

  const sendDescribeViaWs = useCallback((): boolean => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "describe" }));
      return true;
    }
    return false;
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setWsStatus("connecting");
    const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${wsProtocol}//${window.location.host}/api/live`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      setWsStatus("connected");
      reconnectAttemptsRef.current = 0;
    };

    ws.onmessage = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data as string) as WsMessage;
        if (data.type === "path_blocked") {
          const objects = data.objects ?? [];
          if (objects.length > 0) {
            const spoken = objects.map(formatObjectSpeech).join(" ");
            const displayText = objects
              .map((o) => `${o.label.charAt(0).toUpperCase() + o.label.slice(1)} ${o.distance} ahead`)
              .join(", ");
            addMessage({ type: "path_blocked", text: displayText, isUrgent: true });
            onSpeak(spoken, true);
          }
        } else if (data.type === "description") {
          addMessage({ type: "description", text: data.text });
          onSpeak(data.text, false);
        } else if (data.type === "status") {
          setCameraConnected(data.cameraConnected);
          addMessage({ type: "status", text: `Camera ${data.cameraConnected ? "connected" : "disconnected"}` });
          queryClient.setQueryData(getGetCameraStatusQueryKey(), { connected: data.cameraConnected });
        } else if (data.type === "error") {
          addMessage({ type: "error", text: data.message });
        }
      } catch (err) {
        console.error("Failed to parse WS message", err);
      }
    };

    ws.onclose = () => {
      setWsStatus("disconnected");
      const timeout = Math.min(1000 * 2 ** reconnectAttemptsRef.current, 30000);
      reconnectAttemptsRef.current++;
      reconnectTimeoutRef.current = setTimeout(connect, timeout);
    };

    ws.onerror = () => {
      // onclose will also fire
    };

    wsRef.current = ws;
  }, [addMessage, onSpeak, queryClient]);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimeoutRef.current !== null) clearTimeout(reconnectTimeoutRef.current);
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  return { messages, cameraConnected, wsStatus, setCameraConnected, sendDescribeViaWs };
}
