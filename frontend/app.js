const els = {
  statusDot: document.getElementById("statusDot"),
  pipelineStatus: document.getElementById("pipelineStatus"),
  path: document.getElementById("path"),
  reason: document.getElementById("reason"),
  detected: document.getElementById("detected"),
  nearest: document.getElementById("nearest"),
  confidence: document.getElementById("confidence"),
  leftMeter: document.getElementById("leftMeter"),
  centerMeter: document.getElementById("centerMeter"),
  rightMeter: document.getElementById("rightMeter"),
  leftValue: document.getElementById("leftValue"),
  centerValue: document.getElementById("centerValue"),
  rightValue: document.getElementById("rightValue"),
  cloud: document.getElementById("cloud"),
  rangeLabel: document.getElementById("rangeLabel"),
  viewerFps: document.getElementById("viewerFps"),
  viewerResolution: document.getElementById("viewerResolution"),
  viewerPoints: document.getElementById("viewerPoints"),
  objectCount: document.getElementById("objectCount"),
  objects: document.getElementById("objects"),
  ttsCount: document.getElementById("ttsCount"),
  ttsEvents: document.getElementById("ttsEvents"),
};

const ctx = els.cloud.getContext("2d");
let latestData = null;

function formatMm(value) {
  if (value === null || value === undefined) return "--";
  return `${(value / 1000).toFixed(2)} m`;
}

function setLane(name, value) {
  const meter = els[`${name}Meter`];
  const label = els[`${name}Value`];
  meter.value = value || 0;
  label.textContent = formatMm(value);
}

function updateStatus(data) {
  const status = data.pipeline_status || "unknown";
  els.pipelineStatus.textContent = status;
  els.statusDot.className = `dot ${status}`;

  if (!data.state) {
    drawCloud([], data.thresholds);
    updateViewerStats(data.viewer, 0);
    updateObjects([]);
    updateTtsEvents(data.tts_events || []);
    return;
  }

  const state = data.state;
  els.path.textContent = state.recommended_path.replaceAll("_", " ");
  els.reason.textContent = state.reason;
  els.detected.textContent = state.detected ? "Yes" : "No";
  els.nearest.textContent = formatMm(state.nearest_obstacle_mm);
  els.confidence.textContent = `${Math.round(state.confidence * 100)}%`;

  setLane("left", state.lane_clearance_mm.left);
  setLane("center", state.lane_clearance_mm.center);
  setLane("right", state.lane_clearance_mm.right);
  updateViewerStats(data.viewer, state.sample_points?.length || 0);
  updateObjects(state.object_detections || []);
  updateTtsEvents(data.tts_events || []);
  drawCloud(state.sample_points || [], data.thresholds, state.recommended_path);
}

function updateObjects(objects) {
  els.objectCount.textContent = String(objects.length);
  if (!objects.length) {
    els.objects.innerHTML = "<p>No YOLO objects yet.</p>";
    return;
  }

  els.objects.innerHTML = objects
    .map((object) => {
      const distance = formatMm(object.distance_mm);
      const motion = (object.motion || "unknown").replaceAll("_", " ");
      const side = object.lateral_position || "unknown";
      const confidence = Math.round((object.confidence || 0) * 100);
      return `<div class="object-row"><strong>${object.label}</strong><span>${distance}</span><span>${side}</span><span>${motion}</span><small>${confidence}%</small></div>`;
    })
    .join("");
}

function updateTtsEvents(events) {
  els.ttsCount.textContent = String(events.length);
  if (!events.length) {
    els.ttsEvents.innerHTML = "<p>No speech events yet.</p>";
    return;
  }

  els.ttsEvents.innerHTML = events
    .slice()
    .reverse()
    .map((event) => {
      const label = event.type === "camera_status" ? "Camera status" : "Description message";
      const payload = event.payload || {};
      const message = payload.text || `connected: ${payload.cameraConnected}`;
      const time = new Date((event.timestamp || 0) * 1000).toLocaleTimeString();
      return `<div class="tts-event"><strong>${label}</strong><span>${message}</span><span>${time}</span></div>`;
    })
    .join("");
}

function updateViewerStats(viewer, pointCount) {
  if (!viewer) return;
  els.viewerFps.textContent = `${viewer.stream_fps} FPS`;
  els.viewerResolution.textContent = `${viewer.frame_width} x ${viewer.frame_height}`;
  els.viewerPoints.textContent = `${pointCount} / ${viewer.point_sample_limit} pts`;
}

function drawCloud(points, thresholds, path = "") {
  const w = els.cloud.width;
  const h = els.cloud.height;
  const maxRange = (thresholds?.max_range_mm || 3500) / 1000;
  const obstacle = (thresholds?.obstacle_mm || 1500) / 1000;
  const caution = (thresholds?.caution_mm || 2200) / 1000;
  const laneTan = thresholds?.lane_angle_tan || 0.1417;
  const halfTan = thresholds?.fov_half_tan || 0.425;
  els.rangeLabel.textContent = `${maxRange.toFixed(1)} m`;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0a0f12";
  ctx.fillRect(0, 0, w, h);

  const originX = w / 2;
  const originY = h - 24;
  const scale = (h - 54) / maxRange;

  ctx.strokeStyle = "#2b363d";
  ctx.lineWidth = 1;
  for (let m = 1; m <= Math.floor(maxRange); m += 1) {
    const y = originY - m * scale;
    ctx.beginPath();
    ctx.moveTo(24, y);
    ctx.lineTo(w - 24, y);
    ctx.stroke();
    ctx.fillStyle = "#6f8086";
    ctx.fillText(`${m}m`, 28, y - 4);
  }

  drawRangeArc(originX, originY, obstacle * scale, "#ff5d5d");
  drawRangeArc(originX, originY, caution * scale, "#ffd166");

  ctx.strokeStyle = "#ffffff66";
  ctx.beginPath();
  ctx.moveTo(originX, originY);
  ctx.lineTo(originX - maxRange * laneTan * scale, originY - maxRange * scale);
  ctx.moveTo(originX, originY);
  ctx.lineTo(originX + maxRange * laneTan * scale, originY - maxRange * scale);
  ctx.moveTo(originX, originY);
  ctx.lineTo(originX - maxRange * halfTan * scale, originY - maxRange * scale);
  ctx.moveTo(originX, originY);
  ctx.lineTo(originX + maxRange * halfTan * scale, originY - maxRange * scale);
  ctx.stroke();

  drawSuggestedPath(originX, originY, scale, maxRange, laneTan, halfTan, path);

  for (const [x, z] of points) {
    const px = originX + x * scale;
    const py = originY - z * scale;
    const close = z < obstacle;
    ctx.fillStyle = close ? "#ff5d5d" : "#4fb8ff";
    ctx.fillRect(px - 1.5, py - 1.5, 3, 3);
  }

  ctx.fillStyle = path.includes("left") ? "#49d17d" : "#eef4f5";
  ctx.fillText("LEFT", 46, 24);
  ctx.fillStyle = path.includes("right") ? "#49d17d" : "#eef4f5";
  ctx.fillText("RIGHT", w - 88, 24);
  ctx.fillStyle = path.includes("forward") ? "#49d17d" : "#eef4f5";
  ctx.fillText("FORWARD", originX - 28, 24);
}

function drawSuggestedPath(originX, originY, scale, maxRange, laneTan, halfTan, path) {
  if (!path) return;
  ctx.save();
  const pathColor = path.includes("slow") ? "#ffd166" : "#49d17d";
  ctx.lineWidth = 5;
  ctx.strokeStyle = pathColor;
  ctx.fillStyle = pathColor;

  if (path.includes("stop")) {
    ctx.strokeStyle = "#ff5d5d";
    ctx.beginPath();
    ctx.moveTo(originX - 24, originY - 42);
    ctx.lineTo(originX + 24, originY - 78);
    ctx.moveTo(originX + 24, originY - 42);
    ctx.lineTo(originX - 24, originY - 78);
    ctx.stroke();
    ctx.restore();
    return;
  }

  const targetTan =
    path.includes("scan_left") ? -halfTan :
    path.includes("scan_right") ? halfTan :
    path.includes("left") ? -(halfTan + laneTan) / 2 :
    path.includes("right") ? (halfTan + laneTan) / 2 :
    0;
  const points = [];
  const leftEdge = [];
  const rightEdge = [];

  for (let i = 0; i < 18; i += 1) {
    const t = i / 17;
    const z = maxRange * (0.12 + t * 0.62);
    const blend = 1 - Math.pow(1 - t, 2);
    const x = targetTan * z * blend;
    const px = originX + x * scale;
    const py = originY - z * scale;
    const halfWidth = (0.28 * (1 - t) + 0.08 * t) * scale;
    points.push([px, py]);
    leftEdge.push([px - halfWidth, py]);
    rightEdge.push([px + halfWidth, py]);
  }

  ctx.globalAlpha = 0.2;
  ctx.beginPath();
  for (const [i, point] of leftEdge.entries()) {
    if (i === 0) ctx.moveTo(point[0], point[1]);
    else ctx.lineTo(point[0], point[1]);
  }
  for (const point of rightEdge.reverse()) ctx.lineTo(point[0], point[1]);
  ctx.closePath();
  ctx.fill();
  ctx.globalAlpha = 1;

  ctx.beginPath();
  for (const [i, point] of points.entries()) {
    if (i === 0) ctx.moveTo(point[0], point[1]);
    else ctx.lineTo(point[0], point[1]);
  }
  ctx.stroke();
  ctx.restore();
}

function drawRangeArc(originX, originY, radius, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(originX, originY, radius, Math.PI, Math.PI * 2);
  ctx.stroke();
}

async function poll() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    latestData = await response.json();
    updateStatus(latestData);
  } catch {
    els.pipelineStatus.textContent = "offline";
    els.statusDot.className = "dot error";
  }
}

function animateRadar() {
  if (latestData?.state) {
    drawCloud(latestData.state.sample_points || [], latestData.thresholds, latestData.state.recommended_path);
  }
  requestAnimationFrame(animateRadar);
}

poll();
animateRadar();
setInterval(poll, 250);
