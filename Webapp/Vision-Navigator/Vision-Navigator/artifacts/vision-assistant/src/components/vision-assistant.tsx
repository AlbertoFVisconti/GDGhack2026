import { useState, useEffect } from "react";
import { type LucideIcon } from "lucide-react";
import { useSpeech } from "@/hooks/use-speech";
import { useVisionWs } from "@/hooks/use-vision-ws";
import {
  useGetCameraStatus,
  getGetCameraStatusQueryKey,
  useRequestDescription,
} from "@workspace/api-client-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Volume2,
  VolumeX,
  Camera,
  CameraOff,
  Wifi,
  WifiOff,
  RefreshCw,
  AlertTriangle,
  Eye,
  AlertCircle,
} from "lucide-react";
import { format } from "date-fns";
import { Badge } from "@/components/ui/badge";

export function VisionAssistant() {
  const { speak, isMuted, toggleMute } = useSpeech();
  const {
    messages,
    cameraConnected: wsCameraConnected,
    wsStatus,
    setCameraConnected,
    sendDescribeViaWs,
  } = useVisionWs(speak);

  const { data: initialCameraStatus } = useGetCameraStatus({
    query: {
      refetchInterval: 5000,
      queryKey: getGetCameraStatusQueryKey(),
    },
  });

  useEffect(() => {
    if (initialCameraStatus?.connected !== undefined) {
      setCameraConnected(initialCameraStatus.connected);
    }
  }, [initialCameraStatus?.connected, setCameraConnected]);

  const isCameraConnected = wsCameraConnected ?? initialCameraStatus?.connected ?? false;

  const describeMutation = useRequestDescription();

  const handleDescribe = () => {
    const sentViaWs = sendDescribeViaWs();
    if (!sentViaWs) {
      describeMutation.mutate({ data: {} });
    }
  };

  return (
    <div className="flex flex-col h-screen bg-background text-foreground overflow-hidden selection:bg-primary/30">
      {/* Top Status Bar */}
      <header className="flex items-center justify-between px-4 py-3 border-b border-border bg-card/50 backdrop-blur-sm z-10 shrink-0">
        <div className="flex items-center gap-3">
          <h1 className="font-mono font-bold text-lg tracking-tight uppercase">Vision Assist</h1>
          <div className="hidden sm:flex items-center gap-2 border-l border-border pl-3">
            <StatusBadge
              active={wsStatus === "connected"}
              icon={wsStatus === "connected" ? Wifi : wsStatus === "connecting" ? RefreshCw : WifiOff}
              label={wsStatus}
              spin={wsStatus === "connecting"}
              dataTestId="status-ws"
            />
            <StatusBadge
              active={isCameraConnected}
              icon={isCameraConnected ? Camera : CameraOff}
              label={isCameraConnected ? "Cam Active" : "Cam Offline"}
              variant={isCameraConnected ? "default" : "destructive"}
              dataTestId="status-camera"
            />
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="icon"
            onClick={toggleMute}
            className={`transition-colors ${isMuted ? "bg-destructive/10 text-destructive border-destructive/20 hover:bg-destructive/20 hover:text-destructive" : "text-muted-foreground"}`}
            data-testid="button-mute"
            aria-label={isMuted ? "Unmute" : "Mute"}
          >
            {isMuted ? <VolumeX className="h-5 w-5" /> : <Volume2 className="h-5 w-5" />}
          </Button>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col sm:flex-row min-h-0 relative">
        {/* Left Side: Big Actions */}
        <div className="w-full sm:w-1/3 p-4 sm:p-6 flex flex-col gap-4 border-b sm:border-b-0 sm:border-r border-border bg-card/30 shrink-0">
          <div className="flex-1 flex flex-col justify-center">
            <Button
              size="lg"
              className="w-full h-32 sm:h-48 text-xl sm:text-2xl font-mono uppercase tracking-wider bg-primary hover:bg-primary/90 text-primary-foreground shadow-[0_0_40px_rgba(255,158,0,0.15)] hover:shadow-[0_0_60px_rgba(255,158,0,0.3)] transition-all active:scale-[0.98]"
              onClick={handleDescribe}
              disabled={describeMutation.isPending || !isCameraConnected}
              data-testid="button-describe"
            >
              {describeMutation.isPending ? (
                <span className="flex flex-col items-center gap-3">
                  <RefreshCw className="h-8 w-8 animate-spin" />
                  Requesting...
                </span>
              ) : (
                <span className="flex flex-col items-center gap-3">
                  <Eye className="h-8 w-8" />
                  Describe Surroundings
                </span>
              )}
            </Button>
          </div>

          <div className="sm:hidden flex items-center justify-center gap-2 pt-2">
            <StatusBadge
              active={wsStatus === "connected"}
              icon={wsStatus === "connected" ? Wifi : wsStatus === "connecting" ? RefreshCw : WifiOff}
              label={wsStatus}
              spin={wsStatus === "connecting"}
            />
            <StatusBadge
              active={isCameraConnected}
              icon={isCameraConnected ? Camera : CameraOff}
              label={isCameraConnected ? "Cam Active" : "Cam Offline"}
              variant={isCameraConnected ? "default" : "destructive"}
            />
          </div>
        </div>

        {/* Right Side: Log */}
        <div className="flex-1 flex flex-col min-h-0 bg-background/50 relative">
          <div className="px-4 py-3 border-b border-border flex items-center justify-between shrink-0 bg-card/50">
            <h2 className="font-mono text-sm font-semibold tracking-widest text-muted-foreground uppercase">Activity Log</h2>
            <span className="font-mono text-xs text-muted-foreground">{messages.length} Events</span>
          </div>

          <ScrollArea className="flex-1 p-4" data-testid="log-container">
            <div className="flex flex-col gap-3 pb-8">
              {messages.length === 0 ? (
                <div
                  className="flex flex-col items-center justify-center h-48 text-muted-foreground font-mono text-sm"
                  data-testid="empty-log"
                >
                  <RefreshCw className="h-6 w-6 mb-3 animate-spin opacity-20" />
                  Waiting for events...
                </div>
              ) : (
                messages.map((msg) => (
                  <div
                    key={msg.id}
                    className={`p-3 sm:p-4 rounded-md border font-mono text-sm transition-all duration-300 animate-in fade-in slide-in-from-top-2 ${
                      msg.type === "path_blocked"
                        ? "bg-destructive/10 border-destructive/30 text-foreground shadow-[0_0_15px_rgba(255,0,0,0.05)]"
                        : msg.type === "error"
                        ? "bg-destructive/5 border-destructive/20 text-destructive/90"
                        : msg.type === "description"
                        ? "bg-primary/5 border-primary/20 text-foreground"
                        : "bg-card border-border text-muted-foreground"
                    }`}
                    data-testid={`log-item-${msg.type}`}
                  >
                    <div className="flex items-start justify-between gap-4 mb-1">
                      <div className="flex items-center gap-2">
                        {msg.type === "path_blocked" && <AlertTriangle className="h-4 w-4 text-destructive" />}
                        {msg.type === "error" && <AlertCircle className="h-4 w-4 text-destructive" />}
                        {msg.type === "description" && <Eye className="h-4 w-4 text-primary" />}
                        <span
                          className={`text-xs font-bold tracking-wider uppercase ${
                            msg.type === "path_blocked"
                              ? "text-destructive"
                              : msg.type === "description"
                              ? "text-primary"
                              : ""
                          }`}
                        >
                          {msg.type.replace("_", " ")}
                        </span>
                      </div>
                      <span className="text-[10px] opacity-60 tabular-nums shrink-0 mt-0.5">
                        {format(msg.timestamp, "HH:mm:ss.SSS")}
                      </span>
                    </div>
                    <div className="pl-6 text-sm leading-relaxed whitespace-pre-wrap">{msg.text}</div>
                  </div>
                ))
              )}
            </div>
          </ScrollArea>
        </div>
      </main>
    </div>
  );
}

function StatusBadge({
  active,
  icon: Icon,
  label,
  spin = false,
  variant = "default",
  dataTestId,
}: {
  active: boolean;
  icon: LucideIcon;
  label: string;
  spin?: boolean;
  variant?: "default" | "destructive";
  dataTestId?: string;
}) {
  return (
    <Badge
      variant="outline"
      className={`font-mono text-xs uppercase px-2 py-1 flex items-center gap-1.5 transition-colors ${
        active
          ? variant === "default"
            ? "border-primary/50 text-primary bg-primary/10"
            : "border-destructive/50 text-destructive bg-destructive/10"
          : "border-border text-muted-foreground bg-card/50"
      }`}
      data-testid={dataTestId}
    >
      <Icon className={`h-3 w-3 ${spin ? "animate-spin" : ""}`} />
      {label}
    </Badge>
  );
}
