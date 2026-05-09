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
};

const ctx = els.cloud.getContext("2d");

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
  drawCloud(state.sample_points || [], data.thresholds, state.recommended_path);
}

function drawCloud(points, thresholds, path = "") {
  const w = els.cloud.width;
  const h = els.cloud.height;
  const maxRange = (thresholds?.max_range_mm || 3500) / 1000;
  const obstacle = (thresholds?.obstacle_mm || 1500) / 1000;
  const caution = (thresholds?.caution_mm || 2200) / 1000;
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
  ctx.lineTo(originX - maxRange * 0.18 * scale, originY - maxRange * scale);
  ctx.moveTo(originX, originY);
  ctx.lineTo(originX + maxRange * 0.18 * scale, originY - maxRange * scale);
  ctx.stroke();

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
    updateStatus(await response.json());
  } catch {
    els.pipelineStatus.textContent = "offline";
    els.statusDot.className = "dot error";
  }
}

poll();
setInterval(poll, 500);
