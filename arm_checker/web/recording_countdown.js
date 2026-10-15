// A preparation delay, not camera synchronization. No frames are saved here.
export function recordingContext(state, online, elapsedMs) {
  if (!online || !state?.connected || !state.pose?.valid ||
      state.source === "replay" || state.recording?.active) return null;
  const config = state.config || {};
  const fresh = [config.upper_id, config.forearm_id].every((id) => {
    const sensor = state.sensors?.find((item) => String(item.id) === String(id));
    return sensor && sensor.ok !== false && Number.isFinite(sensor.age_ms) &&
      sensor.age_ms >= 0 && sensor.age_ms + elapsedMs < 250;
  });
  if (!fresh) return null;
  return JSON.stringify([state.source, config.upper_id, config.forearm_id,
    config.upper_length, config.forearm_length,
    state.calibration?.calibration_version, state.calibration?.generation]);
}

export function createRecordingCountdown({ getContext, onChange, onStart, onCancel,
  now = () => performance.now(), schedule = setTimeout, unschedule = clearTimeout }) {
  let deadline = null, timer = null, context = null;
  const countdown = {
    get pending() { return deadline !== null; },
    get seconds() { return deadline === null ? 0 : Math.max(1, Math.ceil((deadline - now()) / 1000)); },
    start() {
      if (countdown.pending) return;
      context = getContext();
      if (context === null) return;
      deadline = now() + 5000;
      tick();
    },
    cancel(message) {
      if (!countdown.pending) return;
      unschedule(timer);
      deadline = timer = null;
      onChange();
      if (message) onCancel(message);
    },
    check() {
      if (countdown.pending && getContext() !== context) {
        countdown.cancel("Countdown cancelled: sensor signal or setup changed. Check the live view and try again.");
      }
    },
  };
  function tick() {
    countdown.check();
    if (!countdown.pending) return;
    if (now() >= deadline) {
      deadline = timer = null;
      onStart(context);
      return;
    }
    onChange();
    timer = schedule(tick, 100);
  }
  return countdown;
}
