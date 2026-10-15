import { createArmScene } from "./scene.js";
import { createRecordingCountdown, recordingContext } from "./recording_countdown.js";
import { createRecordingTimer } from "./recording_timer.js";

const $ = (id) => document.getElementById(id);
const el = Object.fromEntries(
  [...document.querySelectorAll("[id]")].map((node) => [node.id, node]),
);
let state = null,
  arrival = 0,
  transportOnline = false,
  selectedSource = null;
let configDirty = false,
  initializedConfig = false,
  calibrationBusy = false;
let lastSource = null,
  lastRecording = false,
  socket = null,
  reconnectTimer = null;
let toastTimer = null,
  sessionsSignature = "",
  sensorSignature = "",
  scene = null;
const sensorNodes = new Map(),
  motionHistory = new Map();
let recordRequestPending = false, pageExitCount = 0;
let savingRecording = false, selectedDuration = 15, automaticSaveErrorShown = false;
let savedSessions = [], deleteTargets = [], deletingSessions = false;
let timerStorage = null;
try { timerStorage = window.sessionStorage; } catch { /* Storage is optional. */ }
const recordingTimer = createRecordingTimer({
  storage: timerStorage,
  onExpire: (id) => stopRecording(id, true),
});
const recordContext = () => recordingContext(
  state, transportOnline, Math.max(0, performance.now() - arrival),
);
const recordingCountdown = createRecordingCountdown({
  getContext: recordContext,
  onChange: render,
  onCancel: (message) => notify(message, true),
  onStart: startAfterCountdown,
});

async function startAfterCountdown(expectedContext) {
  const exitCount = pageExitCount;
  recordRequestPending = true;
  render();
  try {
    // Check current server state again after the preparation delay.
    const current = await request("/api/state");
    if (exitCount !== pageExitCount) return;
    receive(current);
    if (recordContext() !== expectedContext)
      throw new Error("Recording did not start: sensor signal or setup changed.");
    if (await action("/api/record", { action: "start" })) {
      recordingTimer.start(state.recording, selectedDuration);
      automaticSaveErrorShown = false;
      notify(`Recording started. Automatic save in about ${selectedDuration} seconds.`);
    }
  } catch (error) {
    notify(error.message, true);
  } finally {
    recordRequestPending = false;
    render();
  }
}

async function stopRecording(expectedId, automatic = false) {
  if (recordRequestPending) return false;
  recordRequestPending = savingRecording = true;
  render();
  try {
    // A delayed timer must never deliberately stop a different take.
    const current = await request("/api/state");
    receive(current);
    if (!current.recording?.active || current.recording.session_id !== expectedId)
      return true;
    const stopped = await request("/api/record", { action: "stop" });
    receive(stopped);
    if (stopped.recording?.active)
      throw new Error("Recording is still active. Use Stop & save.");
    automaticSaveErrorShown = false;
    notify(automatic ? "Take saved automatically. Stop your phone video too."
      : "Session saved. Replay or download it on the right.");
    refreshSessions();
    return true;
  } catch (error) {
    if (!automatic || !automaticSaveErrorShown)
      notify(`Could not confirm saving: ${error.message}${automatic ? " Keeping the timer active to retry." : ""}`, true);
    automaticSaveErrorShown = automatic;
    return false;
  } finally {
    recordRequestPending = savingRecording = false;
    render();
  }
}

try {
  scene = createArmScene({
    canvas: el.armCanvas,
    container: el.viewport,
    labels: {
      shoulder: el.pointShoulder,
      elbow: el.pointElbow,
      wrist: el.pointWrist,
    },
  });
} catch (error) {
  console.error("3D view unavailable:", error);
  el.sceneError.hidden = false;
  document.querySelector(".scene-labels").hidden = true;
}

function notify(message, error = false) {
  clearTimeout(toastTimer);
  el.toast.textContent = message;
  el.toast.className = `toast${error ? " error" : ""}`;
  el.toast.hidden = false;
  toastTimer = setTimeout(
    () => {
      el.toast.hidden = true;
    },
    error ? 7000 : 3500,
  );
}

async function request(path, body) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 5000);
  try {
    const response = await fetch(
      path,
      body === undefined
        ? { cache: "no-store", signal: controller.signal }
        : {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
            signal: controller.signal,
          },
    );
    const data = await response.json();
    if (!response.ok)
      throw new Error(
        typeof data.detail === "string"
          ? data.detail
          : data.message || `Request failed (${response.status})`,
      );
    return data;
  } catch (error) {
    if (controller.signal.aborted)
      throw new Error("The local workspace did not respond within five seconds.");
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

async function action(path, body, successMessage) {
  try {
    const data = await request(path, body);
    if (data && data.config && data.source) receive(data);
    else receive(await request("/api/state"));
    if (successMessage) notify(successMessage);
    return true;
  } catch (error) {
    notify(error.message, true);
    return false;
  }
}

function sourceName(source) {
  return source === "serial" || source === "live"
    ? "LIVE"
    : source === "replay"
      ? "REPLAY"
      : source === "demo"
        ? "DEMO"
        : "OFFLINE";
}
function selectedSourceControls(source) {
  selectedSource = source;
  el.sourceDemo.classList.toggle("active", source === "demo");
  el.sourceSerial.classList.toggle("active", source === "serial");
  el.sourceDemo.setAttribute("aria-pressed", String(source === "demo"));
  el.sourceSerial.setAttribute("aria-pressed", String(source === "serial"));
  el.serialControls.hidden = source !== "serial";
  el.sourceDescription.textContent =
    source === "demo"
      ? "Explore with simulated motion. Demo readings do not come from your hardware."
      : "Connect your receiver, then check both sensor IDs before putting them on.";
}

function receive(data) {
  state = data;
  arrival = performance.now();
  transportOnline = true;
  if (lastSource !== state.source) {
    scene?.clearTrail();
    motionHistory.clear();
    if (state.source !== "replay")
      selectedSourceControls(state.source === "demo" ? "demo" : "serial");
    lastSource = state.source;
  }
  if (selectedSource === null)
    selectedSourceControls(state.source === "demo" ? "demo" : "serial");
  if (!configDirty && state.config) syncConfig(state.config);
  if (Array.isArray(state.sessions)) renderSessions(state.sessions);
  if (lastRecording && !state.recording?.active) refreshSessions();
  lastRecording = Boolean(state.recording?.active);
  render();
}

function syncConfig(config) {
  if (
    document.activeElement &&
    el.configForm.contains(document.activeElement) &&
    initializedConfig
  )
    return;
  el.upperId.value = String(config.upper_id ?? "3");
  el.forearmId.value = String(config.forearm_id ?? "4");
  el.upperLength.value = String(
    Math.round((Number(config.upper_length) || 0.3) * 1000) / 10,
  );
  el.forearmLength.value = String(
    Math.round((Number(config.forearm_length) || 0.26) * 1000) / 10,
  );
  initializedConfig = true;
}

function calibrationStatus(calibration, sensorsReady) {
  const stage = String(calibration.stage || "").toLowerCase();
  const ready = Boolean(calibration.neutral_ready);
  const tableReady = Boolean(calibration.table_ready);
  const progress = Number(calibration.progress) || 0;
  const active =
    ![
      "",
      "idle",
      "ready",
      "uncalibrated",
      "table_ready",
      "neutral_ready",
      "complete",
      "error",
    ].includes(stage) &&
    (stage.includes("captur") ||
      stage.includes("calibrat") ||
      stage.includes("collect") ||
      stage.includes("hold") ||
      stage === "table" ||
      stage === "neutral" ||
      stage.includes("countdown"));
  calibrationBusy = active;
  el.tableButton.classList.toggle("complete", tableReady);
  el.neutralButton.classList.toggle("complete", ready);
  el.tableCheck.textContent = tableReady ? "✓" : "1";
  el.neutralCheck.textContent = ready ? "✓" : "2";
  const disabled =
    !state.connected ||
    !sensorsReady ||
    !transportOnline ||
    state.source === "replay" ||
    active ||
    recordingCountdown.pending || recordRequestPending ||
    state.recording?.active;
  el.tableButton.disabled = disabled;
  el.neutralButton.disabled = disabled || !tableReady;
  el.resetCalibration.disabled =
    state.source === "replay" ||
    active ||
    recordingCountdown.pending || recordRequestPending ||
    state.recording?.active ||
    (!tableReady && !ready);
  el.calibrationProgress.hidden = !active;
  el.progressBar.style.width = `${Math.max(0, Math.min(1, progress)) * 100}%`;
  el.countdown.textContent = Number.isFinite(calibration.countdown_s)
    ? `${Math.max(0, calibration.countdown_s).toFixed(1)}s`
    : `${Math.round(progress * 100)}%`;
  el.calibrationPrompt.textContent =
    calibration.stationarity_ok === false
      ? "Keep both sensors still"
      : stage.includes("table")
        ? "Capturing table alignment"
        : "Hold your neutral pose";
  el.calibrationMessage.textContent =
    calibration.message ||
    (ready
      ? "Aligned. Move slowly and check that each segment follows your arm."
      : tableReady
        ? "Table alignment saved. Wear both sensors, then capture a neutral pose."
        : "Start with table alignment. Keep both sensors still while each step is captured.");
  el.alignmentStatus.textContent = ready
    ? "Aligned"
    : tableReady
      ? "1 of 2"
      : "Not set";
  el.alignmentDot.className = `status-dot ${ready ? "good" : tableReady ? "warning" : "neutral"}`;
  el.alignmentDetail.textContent = ready
    ? "Table + neutral pose captured"
    : tableReady
      ? "Neutral pose is next"
      : "Two quick steps to get ready";
}

function render() {
  if (!state) return;
  recordingCountdown.check();
  const elapsed = Math.max(0, performance.now() - arrival);
  const available = transportOnline && elapsed < 2500;
  const connected =
    available && (state.source === "replay" || Boolean(state.connected));
  const source = sourceName(state.source);
  el.sourceBadge.textContent = source;
  el.sourceBadge.className = `badge source-badge ${source === "LIVE" ? "live" : source === "REPLAY" ? "replay" : source === "OFFLINE" ? "offline" : ""}`;
  el.connectionDot.className = `status-dot ${connected ? "good" : "neutral"}`;
  el.connectionLabel.textContent = !available
    ? "Workspace disconnected"
    : source === "DEMO"
      ? "Simulated sensor stream"
      : source === "REPLAY"
        ? "Saved session playback"
        : connected
          ? "Receiver connected"
          : "Receiver disconnected";
  el.statusMessage.textContent = !available
    ? "Connection interrupted. Waiting for the local workspace…"
    : state.message ||
      (connected ? "Sensor stream ready." : "Choose a source to begin.");
  const sensors = Array.isArray(state.sensors) ? state.sensors : [];
  renderSensors(sensors, available ? elapsed : Infinity);
  el.streamQuality.hidden = !(
    state.source === "serial" && Number(state.recovered_frames) > 0
  );
  const pose = state.pose || {};
  const assigned = [
    String(state.config?.upper_id ?? "3"),
    String(state.config?.forearm_id ?? "4"),
  ];
  const currentSensors = assigned.map((id) =>
    sensors.find((sensor) => String(sensor.id) === id),
  );
  const fresh =
    state.source === "replay" ||
    currentSensors.every(
      (sensor) =>
        sensor &&
        sensor.ok !== false &&
        Number.isFinite(sensor.age_ms) &&
        sensor.age_ms >= 0 &&
        sensor.age_ms + elapsed < 250,
    );
  calibrationStatus(state.calibration || {}, fresh && available);
  const valid = Boolean(pose.valid && connected && fresh);
  const normalized = { ...state, pose: { ...pose, valid } };
  scene?.update(normalized);
  el.trackingBadge.className = `badge ${valid ? "ready" : "muted"}`;
  el.trackingBadge.textContent = valid
    ? state.source === "replay"
      ? state.replay?.finished
        ? "Replay complete"
        : "●  Replaying motion"
      : "●  Tracking arm"
    : !connected
      ? "Waiting for signal"
      : state.calibration?.neutral_ready
        ? "Sensor signal delayed"
        : "Calibration needed";
  const wrist = Array.isArray(pose.wrist) ? pose.wrist : [],
    shoulder = Array.isArray(pose.shoulder) ? pose.shoulder : [0, 0, 0];
  const angle = Number(pose.elbow_angle_deg ?? pose.elbow_flexion_deg);
  el.elbowAngle.textContent =
    valid && Number.isFinite(angle) ? angle.toFixed(0) : "—";
  [el.wristX, el.wristY, el.wristZ].forEach((node, index) => {
    const value = (Number(wrist[index]) - Number(shoulder[index])) * 100;
    node.textContent =
      valid && Number.isFinite(value)
        ? `${value > 0 ? "+" : ""}${value.toFixed(1)}`
        : "—";
  });
  el.sceneNotice.classList.toggle("live", valid);
  el.sceneNoticeText.textContent = valid
    ? source === "DEMO"
      ? "Simulated motion · fixed shoulder"
      : source === "REPLAY"
        ? "Recorded motion · fixed shoulder"
        : "Estimated arm motion · fixed shoulder"
    : state.calibration?.neutral_ready
      ? "Signal unavailable · pose is not current"
      : "Preview pose · align sensors to begin";
  el.returnLive.hidden = state.source !== "replay";
  const recording = state.recording || {},
    active = Boolean(recording.active),
    counting = recordingCountdown.pending,
    recordingBusy = active || counting || recordRequestPending || deletingSessions;
  const timed = recordingTimer.update(recording);
  el.recordDuration.disabled = recordingBusy;
  if (timed) el.recordDuration.value = String(timed.duration);
  el.recordIndicator.classList.toggle("active", active);
  el.recordButton.classList.toggle("active", active);
  el.recordButton.textContent = savingRecording ? "Saving…" : active
    ? "■  Stop & save"
    : counting ? "Cancel countdown"
      : recordRequestPending ? "Starting…" : "●  Start recording · 5s";
  el.recordButton.disabled =
    deletingSessions || recordRequestPending || (!counting && (!available || (!active && (!valid || state.source === "replay"))));
  const recordTitle = savingRecording ? "Saving your session…" : active
    ? source === "DEMO"
      ? "Recording demo session"
      : "Recording your session"
    : counting ? `Recording starts in ${recordingCountdown.seconds}…`
      : recordRequestPending ? "Starting recording…" : "Capture a session";
  if (el.recordTitle.textContent !== recordTitle)
    el.recordTitle.textContent = recordTitle;
  el.recordDetail.textContent = active
    ? timed ? "Automatic save is enabled. Keep this page open; stop your phone separately."
      : "Sensor readings and alignment are being saved locally."
    : counting ? `Start your phone now. Then ${selectedDuration} seconds of sensor recording.`
    : recordRequestPending ? "Checking the live signal before saving sensor data."
    : state.source === "replay"
      ? "Return to a source to record a new session."
      : !valid
        ? "Align both sensors to enable recording."
        : `5-second countdown, then ${el.recordDuration.value} seconds. Keep this page open for automatic save.`;
  el.recordFrames.classList.toggle("counting", counting);
  el.recordFrames.textContent = active
    ? `${timed ? `${timed.remaining}s left · ` : ""}${Number(recording.frames || 0).toLocaleString()} frames`
    : counting ? `${recordingCountdown.seconds}` : "";
  if (available && connected) {
    if (state.source === "replay") {
      const replay = state.replay || {};
      el.statusMessage.textContent = `${replay.finished ? "Replay complete" : "Playing saved motion"} · ${replay.source === "demo" ? "simulated source" : "recorded sensor source"} · frame ${replay.frame || 0} of ${replay.frames || 0}.`;
    } else if (active) {
      el.statusMessage.textContent = `Recording ${source === "DEMO" ? "simulated readings" : "sensor readings"} locally · ${Number(recording.frames || 0).toLocaleString()} frames captured.`;
    } else if (calibrationBusy) {
      el.statusMessage.textContent =
        state.calibration?.message ||
        "Keep both sensors still while alignment is captured.";
    } else if (valid) {
      el.statusMessage.textContent =
        source === "DEMO"
          ? "Aligned · simulated motion. Try the camera views, then record and replay the simulated motion."
          : "Aligned · live sensor motion. Lift your arm or bend your elbow to check tracking.";
    } else if (state.calibration?.neutral_ready) {
      el.statusMessage.textContent =
        pose.reason ||
        "Alignment is saved. Waiting for fresh readings from both sensors.";
    } else if (state.calibration?.table_ready) {
      el.statusMessage.textContent =
        "Table alignment saved. Mount both sensors, relax your arm down, then capture neutral pose.";
    }
  }
  el.connectButton.disabled = !available || recordingBusy;
  el.sourceDemo.disabled = recordingBusy;
  el.sourceSerial.disabled = recordingBusy;
  el.saveConfig.disabled =
    recordingBusy || state.source === "replay" || calibrationBusy;
  el.sessionList.querySelectorAll("button").forEach((button) => {
    button.disabled = recordingBusy;
  });
  el.deleteAllSessions.disabled = !available || recordingBusy || !savedSessions.length;
  el.confirmDeleteSessions.disabled = !available || recordingBusy;
  el.cancelDeleteSessions.disabled = deletingSessions;
}

function quaternionAngles(q) {
  if (!Array.isArray(q) || q.length !== 4 || !q.every(Number.isFinite))
    return [NaN, NaN, NaN];
  const magnitude = Math.hypot(...q);
  if (magnitude < 1e-9) return [NaN, NaN, NaN];
  const [w, x, y, z] = q.map((v) => v / magnitude),
    deg = 180 / Math.PI;
  return [
    Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)) * deg,
    Math.asin(Math.max(-1, Math.min(1, 2 * (w * y - z * x)))) * deg,
    Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)) * deg,
  ];
}

function rotationSpeed(sensor) {
  const id = String(sensor.id),
    now = performance.now(),
    q = sensor.q;
  let speed = null;
  if (Number.isFinite(sensor.gyro_dps)) speed = Math.abs(sensor.gyro_dps);
  else if (
    Array.isArray(sensor.gyro_dps) &&
    sensor.gyro_dps.every(Number.isFinite)
  )
    speed = Math.hypot(...sensor.gyro_dps);
  if (Array.isArray(q) && q.length === 4 && q.every(Number.isFinite)) {
    const length = Math.hypot(...q);
    if (length > 0) {
      const normal = q.map((v) => v / length),
        previous = motionHistory.get(id);
      if (speed === null && previous && arrival !== previous.arrival) {
        const dot = Math.abs(
          normal.reduce((sum, v, index) => sum + v * previous.q[index], 0),
        );
        const dt = Math.max(0.005, (arrival - previous.arrival) / 1000);
        speed =
          (Math.acos(Math.max(-1, Math.min(1, dot))) * 360) / Math.PI / dt;
      }
      if (!previous || arrival !== previous.arrival)
        motionHistory.set(id, { q: normal, arrival, speed: speed || 0, now });
      else if (speed === null)
        speed = now - previous.now < 300 ? previous.speed : 0;
    }
  }
  return Number.isFinite(speed) ? Math.min(9999, speed) : 0;
}

function createSensorNode(sensor, role) {
  const card = document.createElement("article");
  card.className = `sensor-card ${role}`;
  // All interpolated server values are written with textContent below.
  card.innerHTML =
    '<div class="sensor-card-top"><div><div class="sensor-name"><i class="swatch"></i><span></span></div><span class="sensor-id"></span></div><span class="sensor-health"><i class="status-dot"></i><span></span></span></div><div class="sensor-rotation"><div class="rotation-glyph"><i class="rotation-pointer"></i></div><div class="rotation-readout"><div class="activity-label"><span>ROTATION</span><strong class="activity-value"></strong></div><div class="activity-bar"><i></i></div><div class="sensor-stats"><span class="rate"></span><span class="age"></span></div></div></div><div class="angles-heading">Raw orientation · degrees</div><div class="sensor-angles"><div><span>ROLL</span><strong></strong></div><div><span>PITCH</span><strong></strong></div><div><span>YAW</span><strong></strong></div></div>';
  card.querySelector(".sensor-name span").textContent =
    role === "upper"
      ? "Upper arm"
      : role === "forearm"
        ? "Forearm"
        : String(sensor.label || "Unassigned").replaceAll("_", " ");
  card
    .querySelector(".swatch")
    .classList.add(role === "forearm" ? "coral" : "mint");
  card.querySelector(".sensor-id").textContent = `SENSOR ${sensor.id}`;
  card.title =
    "Raw roll, pitch and yaw identify movement; they are not calibrated anatomical joint angles.";
  return {
    card,
    health: card.querySelector(".sensor-health span"),
    dot: card.querySelector(".sensor-health .status-dot"),
    pointer: card.querySelector(".rotation-pointer"),
    activity: card.querySelector(".activity-value"),
    bar: card.querySelector(".activity-bar i"),
    rate: card.querySelector(".rate"),
    age: card.querySelector(".age"),
    angles: [...card.querySelectorAll(".sensor-angles strong")],
  };
}

function renderSensors(sensors, elapsed) {
  const upperId = String(state.config?.upper_id ?? "3"),
    forearmId = String(state.config?.forearm_id ?? "4");
  // Keep the health panel focused on this arm. A missing assigned sensor still
  // gets a card, so a dropped sensor cannot silently disappear from the setup.
  const ordered =
    state.source === "replay"
      ? []
      : [upperId, forearmId].map(
          (id) =>
            sensors.find((sensor) => String(sensor.id) === id) || {
              id,
              missing: true,
              ok: false,
              age_ms: Infinity,
              reason: `Assigned sensor ${id} is not present in the current stream. Check its connection and sensor ID.`,
            },
        );
  const signature = `${state.source}|${upperId}|${forearmId}|${ordered.map((s) => String(s.id)).join(",")}`;
  if (signature !== sensorSignature) {
    sensorSignature = signature;
    sensorNodes.clear();
    if (ordered.length) {
      el.sensorCards.replaceChildren();
      for (const sensor of ordered) {
        const role =
          String(sensor.id) === upperId
            ? "upper"
            : String(sensor.id) === forearmId
              ? "forearm"
              : "unassigned";
        const nodes = createSensorNode(sensor, role);
        sensorNodes.set(String(sensor.id), nodes);
        el.sensorCards.append(nodes.card);
      }
    } else
      el.sensorCards.innerHTML =
        state.source === "replay"
          ? '<div class="empty-sensors"><span class="empty-orbit" aria-hidden="true">▷</span><strong>Playing saved motion</strong><p>Live sensor readings are hidden during replay. Return to the source to check them.</p></div>'
          : '<div class="empty-sensors"><span class="empty-orbit" aria-hidden="true">◎</span><strong>Waiting for a signal</strong><p>Choose demo to explore, or connect your USB receiver.</p></div>';
  }
  const present = ordered.filter((sensor) => !sensor.missing).length;
  el.sensorCount.textContent = state.source === "replay" ? "—" : `${present}/2`;
  el.sensorCount.title =
    state.source === "replay"
      ? "Live sensors are hidden during replay"
      : `${present} of 2 assigned sensors present`;
  for (const sensor of ordered) {
    const nodes = sensorNodes.get(String(sensor.id));
    if (!nodes) continue;
    const hasAge = Number.isFinite(sensor.age_ms) && sensor.age_ms >= 0;
    const age = hasAge ? sensor.age_ms + elapsed : Infinity;
    const fresh = sensor.ok !== false && age < 250;
    const delayed = age < 1000;
    nodes.card.classList.toggle("stale", !delayed || sensor.ok === false);
    nodes.health.textContent =
      sensor.missing
        ? "Missing"
        : !hasAge
          ? "Unknown"
          : sensor.ok === false && delayed
            ? "Not ready"
            : fresh
              ? "Fresh"
              : delayed
                ? "Delayed"
                : "No signal";
    nodes.dot.className = `status-dot ${fresh ? "good" : delayed ? "warning" : "neutral"}`;
    nodes.card.title =
      sensor.reason ||
      "Raw roll, pitch and yaw identify movement; they are not calibrated anatomical joint angles.";
    nodes.rate.textContent = Number.isFinite(sensor.rate_hz)
      ? `${sensor.rate_hz.toFixed(0)} Hz`
      : "Rate —";
    nodes.age.textContent =
      sensor.missing
        ? "Not in stream"
        : !hasAge
          ? "Age unknown"
          : age < 1000
            ? `${Math.round(age)} ms`
            : Number.isFinite(age)
              ? `${(age / 1000).toFixed(1)} s old`
              : "Disconnected";
    const angles = quaternionAngles(sensor.q),
      speed = fresh ? rotationSpeed(sensor) : 0;
    nodes.angles.forEach((node, index) => {
      node.textContent = Number.isFinite(angles[index])
        ? `${angles[index].toFixed(0)}°`
        : "—";
    });
    nodes.pointer.style.transform = `rotate(${Number.isFinite(angles[2]) ? angles[2] : 0}deg) rotateX(${Number.isFinite(angles[0]) ? angles[0] * 0.35 : 0}deg) rotateY(${Number.isFinite(angles[1]) ? angles[1] * 0.35 : 0}deg)`;
    nodes.activity.textContent = !delayed ? "—" : `${speed.toFixed(0)}°/s`;
    nodes.bar.style.width = `${Math.min(100, Math.sqrt(speed / 180) * 100)}%`;
  }
}

function renderSessions(sessions) {
  savedSessions = sessions;
  const signature = JSON.stringify(sessions);
  if (signature === sessionsSignature) return;
  sessionsSignature = signature;
  el.sessionList.replaceChildren();
  if (!sessions.length) {
    const empty = document.createElement("p");
    empty.className = "empty-copy";
    empty.textContent = "Your recordings will appear here.";
    el.sessionList.append(empty);
    return;
  }
  // The server supplies newest-first order.
  for (const session of sessions.slice(0, 15)) {
    const item = document.createElement("div");
    item.className = "session-item";
    const name = document.createElement("span");
    name.className = "session-name";
    const created = new Date(session.created_utc);
    name.textContent =
      session.name ||
      (Number.isFinite(created.getTime())
        ? created.toLocaleString(undefined, {
            month: "short",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
          })
        : session.id);
    name.title = session.id;
    const meta = document.createElement("p");
    meta.className = "session-meta";
    const seconds = Number(session.duration ?? session.duration_s);
    meta.textContent = `${Number(session.frames || 0).toLocaleString()} frames${Number.isFinite(seconds) ? ` · ${seconds.toFixed(1)} s` : ""}${session.source ? ` · ${sourceName(session.source)}` : ""}`;
    const actions = document.createElement("div");
    actions.className = "session-actions";
    const replay = document.createElement("button");
    replay.type = "button";
    replay.textContent = "▷ Replay";
    replay.addEventListener("click", () =>
      action("/api/replay", { id: session.id }, "Playing saved session."),
    );
    const download = document.createElement("a");
    download.href = `/api/sessions/${encodeURIComponent(session.id)}/download`;
    download.textContent = "↓ Download";
    download.setAttribute("download", "");
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "delete-session";
    remove.textContent = "Delete";
    remove.setAttribute("aria-label", `Delete session ${name.textContent}`);
    remove.addEventListener("click", () => askDeleteSessions([session], name.textContent));
    actions.append(replay, download, remove);
    item.append(name, meta, actions);
    el.sessionList.append(item);
  }
}

function askDeleteSessions(sessions, label) {
  if (state?.recording?.active || recordingCountdown.pending || deletingSessions) return;
  deleteTargets = sessions.map((session) => session.id);
  if (!deleteTargets.length) return;
  el.deleteSessionTitle.textContent = deleteTargets.length === 1 ? "Delete this session?" : `Delete ${deleteTargets.length} sessions?`;
  el.deleteSessionDescription.textContent = deleteTargets.length === 1
    ? `${label || "This recording"} will be removed from Saved sessions.`
    : `All ${deleteTargets.length} listed recordings will be removed from Saved sessions.`;
  el.confirmDeleteSessions.textContent = deleteTargets.length === 1 ? "Delete session" : "Delete sessions";
  el.deleteSessionDialog.showModal();
  el.cancelDeleteSessions.focus();
  render();
}

el.deleteAllSessions.addEventListener("click", () => askDeleteSessions(savedSessions));
el.cancelDeleteSessions.addEventListener("click", () => el.deleteSessionDialog.close());
el.deleteSessionDialog.addEventListener("cancel", (event) => {
  if (deletingSessions) event.preventDefault();
});
el.confirmDeleteSessions.addEventListener("click", async () => {
  if (deletingSessions || !deleteTargets.length) return;
  deletingSessions = true;
  el.confirmDeleteSessions.textContent = "Deleting…";
  render();
  try {
    const result = await request("/api/sessions/delete", { ids: deleteTargets });
    renderSessions(result.sessions);
    el.deleteSessionDialog.close();
    notify(`${result.deleted} ${result.deleted === 1 ? "session moved" : "sessions moved"} to the app's Trash folder.`);
    receive(await request("/api/state"));
  } catch (error) {
    notify(`Could not confirm deletion: ${error.message}`, true);
    refreshSessions();
  } finally {
    deletingSessions = false;
    el.confirmDeleteSessions.textContent = deleteTargets.length === 1 ? "Delete session" : "Delete sessions";
    render();
  }
});

async function refreshSessions() {
  try {
    const data = await request("/api/sessions");
    renderSessions(Array.isArray(data) ? data : data.sessions || []);
  } catch (error) {
    console.warn("Could not refresh sessions:", error.message);
  }
}
async function refreshPorts() {
  try {
    const data = await request("/api/ports"),
      ports = Array.isArray(data) ? data : data.ports || [],
      selected = el.portSelect.value;
    el.portSelect.replaceChildren(
      new Option("Auto-detect USB receiver", "auto"),
    );
    for (const port of ports) {
      const device = typeof port === "string" ? port : port.device;
      el.portSelect.add(
        new Option(
          `${device}${port.description ? ` · ${port.description}` : ""}`,
          device,
        ),
      );
    }
    if ([...el.portSelect.options].some((option) => option.value === selected))
      el.portSelect.value = selected;
    else el.portSelect.value = "auto";
  } catch (error) {
    notify(`USB ports could not be loaded: ${error.message}`, true);
  }
}

el.sourceDemo.addEventListener("click", async () => {
  if (await action("/api/source", { source: "demo" }))
    selectedSourceControls("demo");
});
el.sourceSerial.addEventListener("click", () => {
  selectedSourceControls("serial");
  refreshPorts();
});
el.connectButton.addEventListener("click", () => {
  if (!el.portSelect.value) {
    notify("Choose a USB receiver port first.", true);
    el.portSelect.focus();
    return;
  }
  action("/api/source", {
    source: "serial",
    port: el.portSelect.value,
    protocol: el.protocolSelect.value,
  });
});
el.refreshPorts.addEventListener("click", refreshPorts);
el.configForm.addEventListener("input", () => {
  configDirty = true;
  el.configForm.classList.add("form-dirty");
});
el.configForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const upper = el.upperId.value.trim(),
    forearm = el.forearmId.value.trim();
  if (upper === forearm) {
    notify("Upper arm and forearm need different sensor IDs.", true);
    return;
  }
  const patch = {
    upper_id: upper,
    forearm_id: forearm,
    upper_length: Number(el.upperLength.value) / 100,
    forearm_length: Number(el.forearmLength.value) / 100,
  };
  if (
    await action(
      "/api/config",
      patch,
      "Sensor assignment saved. Check alignment before recording.",
    )
  ) {
    configDirty = false;
    el.configForm.classList.remove("form-dirty");
    scene?.clearTrail();
  }
});
el.tableButton.addEventListener("click", () =>
  action("/api/calibrate", { step: "table" }),
);
el.neutralButton.addEventListener("click", () =>
  action("/api/calibrate", { step: "neutral" }),
);
el.resetCalibration.addEventListener("click", () =>
  action("/api/calibrate", { step: "reset" }),
);
el.recordButton.addEventListener("click", async () => {
  if (recordRequestPending) return;
  if (recordingCountdown.pending) {
    recordingCountdown.cancel();
    notify("Countdown cancelled. No recording was started.");
    return;
  }
  if (!state?.recording?.active) {
    selectedDuration = Number(el.recordDuration.value);
    recordingCountdown.start();
    return;
  }
  await stopRecording(state.recording.session_id);
});
el.recordDuration.addEventListener("change", render);
window.addEventListener("pagehide", () => {
  pageExitCount++;
  recordingCountdown.cancel();
});
el.refreshSessions.addEventListener("click", refreshSessions);
el.returnLive.addEventListener("click", () => action("/api/live", {}));
el.helpButton.addEventListener("click", () => el.helpDialog.showModal());
el.helpDialog.addEventListener("click", (event) => {
  if (event.target === el.helpDialog) {
    const rect = el.helpDialog.getBoundingClientRect();
    if (
      event.clientX < rect.left ||
      event.clientX > rect.right ||
      event.clientY < rect.top ||
      event.clientY > rect.bottom
    )
      el.helpDialog.close();
  }
});
document.querySelectorAll("[data-view]").forEach((button) =>
  button.addEventListener("click", () => {
    document
      .querySelectorAll("[data-view]")
      .forEach((node) => node.classList.toggle("active", node === button));
    scene?.setView(button.dataset.view);
  }),
);
for (const [button, method] of [
  [el.axesToggle, "setAxes"],
  [el.trailToggle, "setTrail"],
]) {
  button.addEventListener("click", () => {
    const active = button.getAttribute("aria-pressed") !== "true";
    button.setAttribute("aria-pressed", String(active));
    button.classList.toggle("active", active);
    scene?.[method](active);
  });
}

function connectSocket() {
  clearTimeout(reconnectTimer);
  socket = new WebSocket(
    `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws`,
  );
  socket.addEventListener("message", (event) => {
    try {
      const data = JSON.parse(event.data);
      receive(data.state || data);
    } catch (error) {
      console.warn("Invalid workspace update:", error);
    }
  });
  socket.addEventListener("close", () => {
    // A recent HTTP state remains usable when WebSocket upgrades are unavailable.
    transportOnline = arrival > 0 && performance.now() - arrival < 2500;
    render();
    reconnectTimer = setTimeout(connectSocket, 1800);
  });
  socket.addEventListener("error", () => socket.close());
}
request("/api/state")
  .then(receive)
  .catch(() => {
    el.connectionLabel.textContent = "Workspace unavailable";
    el.statusMessage.textContent =
      "Waiting for the local server. The workspace will reconnect automatically.";
  });
refreshSessions();
refreshPorts();
connectSocket();
// Age indicators keep advancing even when no new packets arrive.
setInterval(render, 200);
setInterval(() => {
  if (!socket || socket.readyState !== WebSocket.OPEN)
    request("/api/state")
      .then(receive)
      .catch(() => {});
}, 200);
