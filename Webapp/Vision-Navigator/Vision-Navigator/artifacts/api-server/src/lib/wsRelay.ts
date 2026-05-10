import { WebSocketServer, WebSocket } from "ws";
import type { IncomingMessage } from "node:http";
import type { Server } from "node:http";
import { logger } from "./logger";

export type CameraMessage =
  | { type: "path_blocked"; objects: Array<{ label: string; distance: string }> }
  | { type: "description"; text: string }
  | { type: "status"; cameraConnected: boolean };

let cameraSocket: WebSocket | null = null;
const browserClients = new Set<WebSocket>();

function broadcast(msg: CameraMessage) {
  const payload = JSON.stringify(msg);
  for (const client of browserClients) {
    if (client.readyState === WebSocket.OPEN) {
      client.send(payload);
    }
  }
}

function broadcastStatus() {
  broadcast({ type: "status", cameraConnected: cameraSocket !== null });
}

export function isCameraConnected(): boolean {
  return cameraSocket !== null && cameraSocket.readyState === WebSocket.OPEN;
}

export function sendDescribeRequest() {
  if (!isCameraConnected()) return false;
  cameraSocket!.send(JSON.stringify({ type: "describe" }));
  return true;
}

export function setupWebSockets(server: Server) {
  const cameraWss = new WebSocketServer({ noServer: true });
  const browserWss = new WebSocketServer({ noServer: true });

  server.on("upgrade", (req: IncomingMessage, socket, head) => {
    const url = req.url ?? "";

    if (url === "/api/camera" || url.startsWith("/api/camera?")) {
      cameraWss.handleUpgrade(req, socket, head, (ws) => {
        cameraWss.emit("connection", ws, req);
      });
    } else if (url === "/api/live" || url.startsWith("/api/live?")) {
      browserWss.handleUpgrade(req, socket, head, (ws) => {
        browserWss.emit("connection", ws, req);
      });
    } else {
      socket.destroy();
    }
  });

  cameraWss.on("connection", (ws: WebSocket) => {
    logger.info("Camera connected");
    cameraSocket = ws;
    broadcastStatus();

    ws.on("message", (data) => {
      try {
        const msg = JSON.parse(data.toString()) as CameraMessage;
        logger.info({ msg }, "Camera message received");
        broadcast(msg);
      } catch (err) {
        logger.error({ err }, "Failed to parse camera message");
      }
    });

    ws.on("close", () => {
      logger.info("Camera disconnected");
      if (cameraSocket === ws) {
        cameraSocket = null;
      }
      broadcastStatus();
    });

    ws.on("error", (err) => {
      logger.error({ err }, "Camera WebSocket error");
    });
  });

  browserWss.on("connection", (ws: WebSocket) => {
    logger.info("Browser client connected");
    browserClients.add(ws);

    ws.send(JSON.stringify({ type: "status", cameraConnected: isCameraConnected() }));

    ws.on("message", (data) => {
      try {
        const msg = JSON.parse(data.toString());
        if (msg.type === "describe") {
          const forwarded = sendDescribeRequest();
          if (!forwarded) {
            ws.send(JSON.stringify({ type: "error", message: "Camera not connected" }));
          }
        }
      } catch (err) {
        logger.error({ err }, "Failed to parse browser message");
      }
    });

    ws.on("close", () => {
      browserClients.delete(ws);
      logger.info("Browser client disconnected");
    });

    ws.on("error", (err) => {
      logger.error({ err }, "Browser WebSocket error");
    });
  });

  logger.info("WebSocket relay initialized");
}
