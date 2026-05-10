import { Router } from "express";
import { isCameraConnected, sendDescribeRequest } from "../lib/wsRelay";

const router = Router();

router.get("/status", (_req, res) => {
  res.json({ connected: isCameraConnected() });
});

router.post("/describe", (_req, res) => {
  if (!isCameraConnected()) {
    res.status(503).json({ error: "Camera not connected" });
    return;
  }
  sendDescribeRequest();
  res.json({ queued: true });
});

export default router;
