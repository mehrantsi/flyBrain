import { createSceneRenderer, drawRetinaPixels } from "./scene.js";

const canvas = document.querySelector("#scene-canvas");
const retinaCanvas = document.querySelector("#retina-canvas");
const fieldCanvas = document.querySelector("#field-canvas");
const worker = new Worker("./simulation-worker.js", { type: "module" });

const ui = {
  runtimeBadge: document.querySelector("#runtime-badge"),
  status: document.querySelector("#hud-status"),
  statusDot: document.querySelector("#status-dot"),
  error: document.querySelector("#error-toast"),
  startBrain: document.querySelector("#start-brain"),
  startWorld: document.querySelector("#start-world"),
  pause: document.querySelector("#pause-button"),
  retinaToggle: document.querySelector("#retina-toggle"),
  brainToggle: document.querySelector("#brain-toggle"),
  retinaPanel: document.querySelector("#retina-panel"),
  fieldPanel: document.querySelector("#field-panel"),
  retinaState: document.querySelector("#retina-state"),
  launchSheet: document.querySelector("#launch-sheet"),
  panelsToggle: document.querySelector("#panels-toggle"),
  speed: document.querySelector("#stat-speed"),
  brain: document.querySelector("#stat-brain"),
  brainSub: document.querySelector("#stat-brain-sub"),
  mode: document.querySelector("#stat-mode"),
  flight: document.querySelector("#stat-flight"),
  flightMetric: document.querySelector("#stat-flight-metric"),
  altitude: document.querySelector("#stat-altitude"),
  realtime: document.querySelector("#stat-realtime"),
  food: document.querySelector("#stat-food"),
  foodSub: document.querySelector("#stat-food-sub"),
  motor: document.querySelector("#stat-motor"),
  motorSub: document.querySelector("#stat-motor-sub"),
  clock: document.querySelector("#sim-clock"),
  progress: document.querySelector("#sim-progress"),
  diagSimTime: document.querySelector("#diag-sim-time"),
  diagWallElapsed: document.querySelector("#diag-wall-elapsed"),
  diagRealtime: document.querySelector("#diag-realtime"),
  diagBodyNeurons: document.querySelector("#diag-body-neurons"),
  diagTotalSpikes: document.querySelector("#diag-total-spikes"),
  diagPhysicsWarnings: document.querySelector("#diag-physics-warnings"),
  fieldValue: document.querySelector("#field-value"),
  fieldFrequency: document.querySelector("#field-frequency"),
  cameraDistance: document.querySelector("#camera-distance"),
};

const state = {
  running: false,
  paused: false,
  brain: true,
  retinaVisible: false,
  fieldVisible: false,
  cameraMode: "chase",
  lastSnapshot: null,
  sceneDescriptor: null,
  pendingFrame: null,
  fieldHistory: [],
  lastFieldSampleSequence: null,
  lastFieldSampleTime: null,
};

let renderer;
try {
  renderer = createSceneRenderer(canvas, {
    onVision: (pixels) => {
      worker.postMessage({ type: "vision", pixels }, [pixels]);
    },
    onError: (error) => showError(`Retina capture failed: ${error.message ?? error}`),
  });
} catch (error) {
  showError(`3D renderer unavailable: ${error instanceof Error ? error.message : String(error)}`);
}

function finite(value, fallback = 0) {
  if (value === null || value === undefined || value === "") return fallback;
  return Number.isFinite(Number(value)) ? Number(value) : fallback;
}

function text(value, fallback = "—") {
  return value === undefined || value === null || value === "" ? fallback : String(value);
}

function formatNumber(value, digits = 1) {
  const number = finite(value, NaN);
  if (!Number.isFinite(number)) return "—";
  return number.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

function formatInteger(value) {
  const number = finite(value, NaN);
  if (!Number.isFinite(number)) return "—";
  return Math.round(number).toLocaleString();
}

function modeLabel(value) {
  return text(value, "—").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function setStatus(message, kind = "working") {
  ui.status.textContent = text(message, "Worker status unavailable");
  ui.runtimeBadge.textContent = kind === "ready" ? "Live" : kind === "error" ? "Error" : kind === "idle" ? "Ready" : "Loading";
  ui.statusDot.className = `status-dot ${kind === "ready" ? "ready" : kind === "error" ? "error" : ""}`;
}

function showError(message) {
  ui.error.textContent = text(message, "Unknown browser runtime error");
  ui.error.hidden = false;
  setStatus(ui.error.textContent, "error");
  ui.progress.textContent = "runtime error";
}

function clearError() {
  ui.error.hidden = true;
}

function returnToStartup() {
  document.body.classList.remove("running", "inspector-open", "panels-hidden");
  document.body.classList.add("startup");
  ui.launchSheet.hidden = false;
  updatePanelsToggle();
}

function setStartButtonsEnabled(enabled) {
  ui.startBrain.disabled = !enabled;
  ui.startWorld.disabled = !enabled;
}

function resetTelemetry(clearScene = false) {
  state.lastSnapshot = null;
  if (clearScene) state.sceneDescriptor = null;
  state.fieldHistory = [];
  state.lastFieldSampleSequence = null;
  state.lastFieldSampleTime = null;
  ui.speed.textContent = "—";
  ui.brain.textContent = "—";
  ui.brainSub.textContent = "network idle";
  ui.mode.textContent = "—";
  ui.flight.textContent = "—";
  ui.flightMetric.textContent = "—";
  ui.altitude.textContent = "—";
  ui.realtime.textContent = "—";
  ui.food.textContent = "—";
  ui.foodSub.textContent = "resource not sampled";
  ui.motor.textContent = "—";
  ui.motorSub.textContent = "awaiting CNS telemetry";
  ui.clock.textContent = "t —";
  ui.progress.textContent = "idle";
  ui.diagSimTime.textContent = "sim —";
  ui.diagWallElapsed.textContent = "wall —";
  ui.diagRealtime.textContent = "rt —";
  ui.diagBodyNeurons.textContent = "body/neuron —";
  ui.diagTotalSpikes.textContent = "spikes —";
  ui.diagPhysicsWarnings.textContent = "physics —";
  ui.fieldValue.textContent = "—";
  ui.fieldFrequency.textContent = "—";
  drawField();
}

function startSimulation(brain) {
  if (state.running) return;
  state.brain = Boolean(brain);
  state.running = true;
  state.paused = false;
  const detailedPanels = !window.matchMedia("(max-width: 760px), (hover: none) and (pointer: coarse)").matches;
  document.body.classList.remove("startup", "panels-hidden");
  document.body.classList.add("running");
  document.body.classList.toggle("inspector-open", detailedPanels);
  if (!state.retinaVisible) toggleRetina();
  if (state.fieldVisible !== detailedPanels) toggleField();
  updatePanelsToggle();
  ui.launchSheet.hidden = true;
  resetTelemetry(true);
  ui.pause.textContent = "Pause";
  ui.pause.setAttribute("aria-pressed", "false");
  clearError();
  ui.progress.textContent = state.brain ? "full CNS requested" : "world-only diagnostic requested";
  worker.postMessage({ type: "start", brain: state.brain });
  ui.startBrain.classList.toggle("active", state.brain);
  ui.startWorld.classList.toggle("active", !state.brain);
  setStartButtonsEnabled(false);
}

function togglePause() {
  if (!state.running) return;
  state.paused = !state.paused;
  worker.postMessage({ type: "pause", paused: state.paused });
  ui.pause.textContent = state.paused ? "Resume" : "Pause";
  ui.pause.setAttribute("aria-pressed", String(state.paused));
  ui.progress.textContent = state.paused ? "paused" : "live";
}

function sendCommand(command) {
  if (!state.running) return;
  if (command === "reset") resetTelemetry();
  worker.postMessage({ type: "command", command });
  ui.progress.textContent = `${command} command sent`;
}

function toggleRetina() {
  state.retinaVisible = !state.retinaVisible;
  ui.retinaPanel.hidden = !state.retinaVisible;
  worker.postMessage({ type: "retina-display", visible: state.retinaVisible });
  ui.retinaToggle.setAttribute("aria-pressed", String(state.retinaVisible));
  ui.retinaState.textContent = state.retinaVisible ? "sampling · 15 Hz max" : "sampling in background";
}

function toggleField() {
  state.fieldVisible = !state.fieldVisible;
  ui.fieldPanel.hidden = !state.fieldVisible;
  ui.brainToggle.setAttribute("aria-pressed", String(state.fieldVisible));
}

function updateCameraButtons() {
  document.querySelectorAll("[data-camera]").forEach((button) => {
    button.classList.toggle("active", button.dataset.camera === state.cameraMode);
  });
}

function setCamera(mode) {
  state.cameraMode = mode === "room" || mode === "orbit" ? mode : "chase";
  renderer?.setCameraMode(state.cameraMode);
  updateCameraButtons();
}

function updateHud(snapshot) {
  if (!snapshot || typeof snapshot !== "object") return;
  state.lastSnapshot = snapshot;
  ui.speed.textContent = formatNumber(snapshot.horizontal_speed_mm_s);
  if (state.brain) {
    ui.brain.textContent = `${formatNumber(snapshot.filtered_population_rate_hz / 1e6, 2)} M spikes/s`;
    ui.brainSub.textContent = `Δ ${formatInteger(snapshot.population_spike_delta)} · ${formatInteger(snapshot.cumulative_spiking_neuron_count)} ever-spiking neurons`;
  } else {
    ui.brain.textContent = "OFF";
    ui.brainSub.textContent = "world-only diagnostic";
  }
  ui.mode.textContent = modeLabel(snapshot.behavior_mode);
  const flightPermission = snapshot.flight_allowed ? "allowed" : "disabled";
  const flightText = `${modeLabel(snapshot.flight_mode)} · ${flightPermission}`;
  ui.flight.textContent = flightText;
  ui.flightMetric.textContent = flightText;
  ui.altitude.textContent = formatNumber(snapshot.root_position[2]);
  ui.food.textContent = snapshot.food_enabled ? `${formatNumber(snapshot.food_distance)} mm` : "hidden";
  const odorText = `L ${formatNumber(snapshot.odor_left_ppm, 3)} · R ${formatNumber(snapshot.odor_right_ppm, 3)} ppm`;
  const visualText = `vision ${formatNumber(snapshot.visual_left, 2)}/${formatNumber(snapshot.visual_right, 2)} · Δ${formatInteger(snapshot.visual_event_delta)}`;
  ui.foodSub.textContent = snapshot.taste_active
    ? `physical taste active · ${odorText}`
    : `${odorText} · ${visualText} · seeking`;
  if (!state.brain) {
    ui.motor.textContent = "OFF";
  } else {
    ui.motor.textContent = `walk ${formatNumber(snapshot.brain_walking_drive, 2)} · turn ${formatNumber(snapshot.brain_walking_steering, 2)}`;
  }
  ui.motorSub.textContent = state.brain
    ? `MN9 ${formatNumber(snapshot.filtered_mn9_rate_hz)} Hz Δ${formatInteger(snapshot.mn9_spike_delta)} · extension ${formatNumber(snapshot.feeding_extension, 2)} · flight ${formatNumber(snapshot.brain_flight_drive, 2)}/${formatNumber(snapshot.brain_flight_steering, 2)}`
    : "neural motor outputs disabled";
  ui.clock.textContent = `t ${formatNumber(snapshot.time_seconds, 2)} s`;
  ui.realtime.textContent = `${formatNumber(snapshot.realtime_factor, 2)}×`;
  ui.progress.textContent = snapshot.grooming_active ? "grooming" : snapshot.taste_active ? "feeding contact" : state.paused ? "paused" : "live frame";
  updateField(snapshot);
  updateDiagnostics(snapshot);
}

function updateDiagnostics(snapshot) {
  const simulationTime = Number(snapshot.time_seconds);
  const wallSeconds = Number(snapshot.wall_elapsed_seconds);
  const realtimeRatio = Number(snapshot.realtime_factor);
  ui.diagSimTime.textContent = `sim ${Number.isFinite(simulationTime) ? `${formatNumber(simulationTime, 2)}s` : "—"}`;
  ui.diagWallElapsed.textContent = `wall ${Number.isFinite(wallSeconds) ? `${formatNumber(wallSeconds, 2)}s` : "—"}`;
  ui.diagRealtime.textContent = `rt ${Number.isFinite(realtimeRatio) ? `${formatNumber(realtimeRatio, 2)}×` : "—"}`;

  const bodyCount = Number(state.sceneDescriptor?.bodyCount);
  const neuronCount = Number(state.sceneDescriptor?.brain?.neurons);
  ui.diagBodyNeurons.textContent = `body ${Number.isFinite(bodyCount) ? formatInteger(bodyCount) : "—"} · neuron ${Number.isFinite(neuronCount) ? formatInteger(neuronCount) : "—"}`;

  const totalSpikes = state.brain ? formatInteger(snapshot.total_spikes) : "OFF";
  ui.diagTotalSpikes.textContent = `spikes ${totalSpikes}`;

  if (!Object.prototype.hasOwnProperty.call(snapshot, "physics_warnings")) {
    ui.diagPhysicsWarnings.textContent = "physics —";
  } else if (!Array.isArray(snapshot.physics_warnings) || snapshot.physics_warnings.length === 0) {
    ui.diagPhysicsWarnings.textContent = "physics ok";
  } else {
    ui.diagPhysicsWarnings.textContent = `physics ${snapshot.physics_warnings.map((warning) => text(warning)).join(",")}`;
  }
}

function updateField(snapshot) {
  const potential = Number(snapshot.brain_field_potential_mv);
  const frequency = Number(snapshot.brain_field_dominant_frequency_hz);
  const sampleSequence = Number(snapshot.brain_field_sample_sequence);
  const sampleTime = Number(snapshot.time_seconds);
  const duplicate = sampleSequence === state.lastFieldSampleSequence;
  if (Number.isFinite(potential) && Number.isFinite(frequency) && Number.isFinite(sampleSequence) && Number.isFinite(sampleTime) && !duplicate) {
    state.fieldHistory.push({ potential, frequency, time: sampleTime });
    if (state.fieldHistory.length > 180) state.fieldHistory.splice(0, state.fieldHistory.length - 180);
    ui.fieldValue.textContent = `${formatNumber(potential, 3)} mV`;
    ui.fieldFrequency.textContent = `${formatNumber(frequency, 2)} Hz`;
    state.lastFieldSampleSequence = sampleSequence;
    state.lastFieldSampleTime = sampleTime;
  }
  drawField();
}

function drawField() {
  const context = fieldCanvas?.getContext("2d");
  if (!context) return;
  const width = fieldCanvas.width;
  const height = fieldCanvas.height;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "rgba(5, 9, 13, 0.9)";
  context.fillRect(0, 0, width, height);
  context.strokeStyle = "rgba(120, 214, 208, 0.11)";
  context.lineWidth = 1;
  for (let row = 1; row < 4; row += 1) {
    const y = Math.round((height * row) / 4) + 0.5;
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(width, y);
    context.stroke();
  }
  if (state.fieldHistory.length < 2) return;
  const values = state.fieldHistory.map(({ potential }) => potential);
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const span = Math.max(maximum - minimum, 1e-6);
  context.strokeStyle = "#78d6d0";
  context.lineWidth = 2;
  context.beginPath();
  const times = state.fieldHistory.map(({ time }) => time);
  const firstTime = times[0];
  const lastTime = times[times.length - 1];
  const timeSpan = lastTime - firstTime;
  values.forEach((value, index) => {
    const x = Number.isFinite(timeSpan) && timeSpan > 0
      ? ((times[index] - firstTime) / timeSpan) * width
      : (index / (values.length - 1)) * width;
    const y = height - 8 - ((value - minimum) / span) * (height - 16);
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.stroke();
}

worker.onmessage = (event) => {
  const message = event.data ?? {};
  if (message.type === "status") {
    const live = message.message.startsWith("Running:") || message.message.startsWith("World-only diagnostic");
    setStatus(message.message, live ? "ready" : "working");
    return;
  }
  if (message.type === "error") {
    state.running = false;
    state.paused = false;
    setStartButtonsEnabled(true);
    returnToStartup();
    showError(message.message);
    return;
  }
  if (message.type === "scene") {
    state.sceneDescriptor = message.scene ?? null;
    if (!state.lastSnapshot) updateDiagnostics({});
    try {
      renderer?.setScene(message.scene);
      if (state.pendingFrame) {
        renderer?.updateFrame(state.pendingFrame.poses, state.pendingFrame.snapshot);
        state.pendingFrame = null;
      }
      setStatus("Live scene ready", "ready");
    } catch (error) {
      showError(`Scene descriptor rejected: ${error instanceof Error ? error.message : String(error)}`);
    }
    return;
  }
  if (message.type === "frame") {
    if (renderer) {
      if (!renderer.sceneReady) {
        state.pendingFrame = message;
      } else {
        renderer.updateFrame(message.poses, message.snapshot);
      }
    }
    updateHud(message.snapshot);
    return;
  }
  if (message.type === "retina") {
    drawRetinaPixels(retinaCanvas, message.pixels);
    return;
  }
};

worker.onerror = (event) => {
  state.running = false;
  state.paused = false;
  setStartButtonsEnabled(true);
  returnToStartup();
  showError(`Simulation worker failed: ${event.message || "unknown worker error"}`);
};

ui.startBrain.addEventListener("click", () => startSimulation(true));
ui.startWorld.addEventListener("click", () => startSimulation(false));
ui.pause.addEventListener("click", togglePause);
ui.retinaToggle.addEventListener("click", toggleRetina);
ui.brainToggle.addEventListener("click", toggleField);

function updatePanelsToggle() {
  const inspectorOpen = document.body.classList.contains("inspector-open");
  ui.panelsToggle.textContent = inspectorOpen ? "Close inspector" : "Inspector";
  ui.panelsToggle.setAttribute("aria-expanded", String(inspectorOpen));
}

ui.panelsToggle.addEventListener("click", () => {
  if (document.body.classList.contains("startup")) return;
  document.body.classList.remove("panels-hidden");
  document.getElementById("visibility-toggle").textContent = "Hide UI";
  document.getElementById("visibility-toggle").setAttribute("aria-pressed", "false");
  document.body.classList.toggle("inspector-open");
  updatePanelsToggle();
});
document.getElementById("visibility-toggle").addEventListener("click", (event) => {
  const hidden = document.body.classList.toggle("panels-hidden");
  event.currentTarget.textContent = hidden ? "Show UI" : "Hide UI";
  event.currentTarget.setAttribute("aria-pressed", String(hidden));
});
document.querySelectorAll("[data-camera]").forEach((button) => {
  button.addEventListener("click", () => setCamera(button.dataset.camera));
});
document.querySelectorAll("[data-command]").forEach((button) => {
  button.addEventListener("click", () => sendCommand(button.dataset.command));
});

window.addEventListener("resize", () => renderer?.resize());
window.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) return;
  const key = event.key.toLowerCase();
  if (event.code === "Space") {
    event.preventDefault();
    togglePause();
  } else if (key === "1") {
    setCamera("chase");
  } else if (key === "5") {
    setCamera("room");
  } else if (key === "h") {
    sendCommand("groom");
  } else if (key === "r") {
    sendCommand("reset");
  } else if (key === "t") {
    sendCommand("sugar");
  } else if (key === "g") {
    sendCommand("flight");
  } else if (key === "v") {
    toggleRetina();
  } else if (key === "b") {
    toggleField();
  } else if (key === "f") {
    sendCommand("food");
  }
});

window.addEventListener("beforeunload", () => {
  renderer?.dispose();
  worker.terminate();
});

setCamera("chase");
updatePanelsToggle();
setStatus("Choose full CNS or world-only diagnostic", "idle");

function animate() {
  renderer?.render();
  if (renderer?.frameReady) {
    const distance = renderer.camera.position.distanceTo(renderer.controls.target).toFixed(1);
    if (ui.cameraDistance.textContent !== distance) ui.cameraDistance.textContent = distance;
  }
  window.requestAnimationFrame(animate);
}

animate();
