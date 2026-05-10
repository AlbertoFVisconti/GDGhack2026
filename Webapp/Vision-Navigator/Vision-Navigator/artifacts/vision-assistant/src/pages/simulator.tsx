import { useEffect, useRef, useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { format } from "date-fns";
import {
  Wifi,
  WifiOff,
  RefreshCw,
  AlertTriangle,
  Eye,
  Send,
  Trash2,
  ArrowLeft,
  Video,
} from "lucide-react";
import { Link } from "wouter";

type LogEntry = {
  id: string;
  timestamp: Date;
  direction: "sent" | "received";
  payload: string;
};

type PresetAlert = {
  label: string;
  objects: Array<{ label: string; distance: string }>;
};

const PRESET_ALERTS: PresetAlert[] = [
  { label: "Person 2m", objects: [{ label: "person", distance: "2m" }] },
  { label: "Chair 1m", objects: [{ label: "chair", distance: "1m" }] },
  { label: "Bicycle 3m", objects: [{ label: "bicycle", distance: "3m" }] },
  {
    label: "Multi-obstacle",
    objects: [
      { label: "person", distance: "2m" },
      { label: "door", distance: "5m" },
    ],
  },
  { label: "Car 10m", objects: [{ label: "car", distance: "10m" }] },
  { label: "Table 1.5m", objects: [{ label: "table", distance: "1.5m" }] },
];

const PRESET_DESCRIPTIONS = [
  "You are in a corridor. There is a door straight ahead about 8 metres away, and windows on your right side letting in natural light.",
  "You are standing in an open office area. Several desks are arranged around you. There are people working at computers to your left.",
  "You are at the top of a staircase. The stairs descend in front of you. There is a handrail on both sides.",
  "You are outdoors on a pavement. The road is to your right. There are trees and a bench to your left about 4 metres away.",
];

export function Simulator() {
  const [wsStatus, setWsStatus] = useState<"connecting" | "connected" | "disconnected">("connecting");
  const [log, setLog] = useState<LogEntry[]>([]);
  const [customLabel, setCustomLabel] = useState("");
  const [customDistance, setCustomDistance] = useState("");
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const addLog = useCallback((direction: "sent" | "received", payload: string) => {
    setLog((prev) => [
      {
        id: Math.random().toString(36).substring(7),
        timestamp: new Date(),
        direction,
        payload,
      },
      ...prev,
    ]);
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    setWsStatus("connecting");
    const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${wsProtocol}//${window.location.host}/api/camera`);

    ws.onopen = () => {
      setWsStatus("connected");
      addLog("sent", '{"type":"status","cameraConnected":true}');
      ws.send(JSON.stringify({ type: "status", cameraConnected: true }));
    };

    ws.onmessage = (event: MessageEvent) => {
      addLog("received", event.data as string);
    };

    ws.onclose = () => {
      setWsStatus("disconnected");
      reconnectTimeoutRef.current = setTimeout(connect, 3000);
    };

    ws.onerror = () => {};
    wsRef.current = ws;
  }, [addLog]);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimeoutRef.current !== null) clearTimeout(reconnectTimeoutRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  const send = useCallback(
    (payload: object) => {
      if (wsRef.current?.readyState !== WebSocket.OPEN) return;
      const text = JSON.stringify(payload);
      wsRef.current.send(text);
      addLog("sent", text);
    },
    [addLog]
  );

  const sendPathBlocked = (objects: Array<{ label: string; distance: string }>) => {
    send({ type: "path_blocked", objects });
  };

  const sendDescription = (text: string) => {
    send({ type: "description", text });
  };

  const sendCustom = () => {
    if (!customLabel.trim() || !customDistance.trim()) return;
    sendPathBlocked([{ label: customLabel.trim(), distance: customDistance.trim() }]);
    setCustomLabel("");
    setCustomDistance("");
  };

  return (
    <div className="flex flex-col h-screen bg-background text-foreground overflow-hidden">
      {/* Header */}
      <header className="flex items-center gap-4 px-4 py-3 border-b border-border bg-card/50 backdrop-blur-sm shrink-0">
        <Link href="/">
          <Button variant="ghost" size="icon" aria-label="Back to main app">
            <ArrowLeft className="h-4 w-4" />
          </Button>
        </Link>
        <div className="flex items-center gap-2">
          <Video className="h-4 w-4 text-primary" />
          <h1 className="font-mono font-bold text-base tracking-tight uppercase">Camera Simulator</h1>
        </div>
        <div className="flex items-center gap-1.5 ml-2">
          <Badge
            variant="outline"
            className={`font-mono text-xs uppercase px-2 py-1 flex items-center gap-1.5 ${
              wsStatus === "connected"
                ? "border-primary/50 text-primary bg-primary/10"
                : wsStatus === "connecting"
                ? "border-yellow-500/50 text-yellow-400 bg-yellow-500/10"
                : "border-destructive/50 text-destructive bg-destructive/10"
            }`}
          >
            {wsStatus === "connected" ? (
              <Wifi className="h-3 w-3" />
            ) : wsStatus === "connecting" ? (
              <RefreshCw className="h-3 w-3 animate-spin" />
            ) : (
              <WifiOff className="h-3 w-3" />
            )}
            {wsStatus === "connected" ? "Camera Connected" : wsStatus}
          </Badge>
        </div>
        <div className="ml-auto">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setLog([])}
            aria-label="Clear log"
            className="text-muted-foreground hover:text-foreground"
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>
      </header>

      <div className="flex-1 flex flex-col sm:flex-row min-h-0">
        {/* Left: Controls */}
        <div className="w-full sm:w-80 shrink-0 border-b sm:border-b-0 sm:border-r border-border overflow-y-auto p-4 flex flex-col gap-5 bg-card/20">

          {/* Path Blocked Presets */}
          <section>
            <p className="font-mono text-xs font-semibold tracking-widest text-muted-foreground uppercase mb-3 flex items-center gap-2">
              <AlertTriangle className="h-3 w-3 text-destructive" />
              Path Blocked Alerts
            </p>
            <div className="grid grid-cols-2 gap-2">
              {PRESET_ALERTS.map((preset) => (
                <Button
                  key={preset.label}
                  variant="outline"
                  size="sm"
                  className="font-mono text-xs h-auto py-2 text-left justify-start border-destructive/20 hover:bg-destructive/10 hover:border-destructive/40 hover:text-foreground"
                  onClick={() => sendPathBlocked(preset.objects)}
                  disabled={wsStatus !== "connected"}
                >
                  <AlertTriangle className="h-3 w-3 text-destructive shrink-0 mr-1.5" />
                  {preset.label}
                </Button>
              ))}
            </div>
          </section>

          {/* Custom Alert */}
          <section>
            <p className="font-mono text-xs font-semibold tracking-widest text-muted-foreground uppercase mb-3">
              Custom Obstacle
            </p>
            <div className="flex flex-col gap-2">
              <Input
                placeholder="Label (e.g. dog)"
                value={customLabel}
                onChange={(e) => setCustomLabel(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && sendCustom()}
                className="font-mono text-sm h-8"
              />
              <Input
                placeholder="Distance (e.g. 3m)"
                value={customDistance}
                onChange={(e) => setCustomDistance(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && sendCustom()}
                className="font-mono text-sm h-8"
              />
              <Button
                size="sm"
                onClick={sendCustom}
                disabled={wsStatus !== "connected" || !customLabel.trim() || !customDistance.trim()}
                className="font-mono text-xs"
              >
                <Send className="h-3 w-3 mr-1.5" />
                Send Alert
              </Button>
            </div>
          </section>

          <Separator />

          {/* Scene Descriptions */}
          <section>
            <p className="font-mono text-xs font-semibold tracking-widest text-muted-foreground uppercase mb-3 flex items-center gap-2">
              <Eye className="h-3 w-3 text-primary" />
              Scene Descriptions
            </p>
            <div className="flex flex-col gap-2">
              {PRESET_DESCRIPTIONS.map((desc, i) => (
                <Button
                  key={i}
                  variant="outline"
                  size="sm"
                  className="font-mono text-xs h-auto py-2 text-left justify-start whitespace-normal border-primary/20 hover:bg-primary/10 hover:border-primary/40 hover:text-foreground"
                  onClick={() => sendDescription(desc)}
                  disabled={wsStatus !== "connected"}
                >
                  <Eye className="h-3 w-3 text-primary shrink-0 mr-1.5 mt-0.5 self-start" />
                  <span className="line-clamp-2">{desc}</span>
                </Button>
              ))}
            </div>
          </section>
        </div>

        {/* Right: Log */}
        <div className="flex-1 flex flex-col min-h-0">
          <div className="px-4 py-3 border-b border-border flex items-center justify-between shrink-0 bg-card/50">
            <h2 className="font-mono text-sm font-semibold tracking-widest text-muted-foreground uppercase">Message Log</h2>
            <span className="font-mono text-xs text-muted-foreground">{log.length} messages</span>
          </div>
          <ScrollArea className="flex-1 p-4">
            <div className="flex flex-col gap-2 pb-8">
              {log.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-48 text-muted-foreground font-mono text-sm">
                  <RefreshCw className="h-6 w-6 mb-3 animate-spin opacity-20" />
                  No messages yet — trigger an event on the left.
                </div>
              ) : (
                log.map((entry) => (
                  <div
                    key={entry.id}
                    className={`p-3 rounded-md border font-mono text-xs ${
                      entry.direction === "sent"
                        ? "bg-card border-border text-foreground"
                        : "bg-primary/5 border-primary/20 text-primary"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1.5">
                      <span
                        className={`text-[10px] font-bold tracking-wider uppercase ${
                          entry.direction === "sent" ? "text-muted-foreground" : "text-primary"
                        }`}
                      >
                        {entry.direction === "sent" ? "→ Camera sent" : "← Server replied"}
                      </span>
                      <span className="text-[10px] opacity-50 tabular-nums">
                        {format(entry.timestamp, "HH:mm:ss.SSS")}
                      </span>
                    </div>
                    <pre className="whitespace-pre-wrap break-all leading-relaxed">
                      {(() => {
                        try {
                          return JSON.stringify(JSON.parse(entry.payload), null, 2);
                        } catch {
                          return entry.payload;
                        }
                      })()}
                    </pre>
                  </div>
                ))
              )}
            </div>
          </ScrollArea>
        </div>
      </div>
    </div>
  );
}
