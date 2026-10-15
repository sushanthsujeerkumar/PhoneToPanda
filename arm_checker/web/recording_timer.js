// The browser requests Stop & save. Keep the page open for this timer to run.
export function createRecordingTimer({ onExpire, storage = null, now = () => Date.now() }) {
  const key = "arm-studio-recording-timer";
  let plan = null, saving = false, retryAt = 0;
  try {
    const saved = JSON.parse(storage?.getItem(key) || "null");
    if (saved && typeof saved.id === "string" && Number.isFinite(saved.deadline) &&
        [10, 15, 30].includes(saved.duration)) plan = saved;
  } catch { /* Storage may be unavailable; the current page can still time a take. */ }
  function clear() {
    plan = null;
    try { storage?.removeItem(key); } catch { /* Optional reload recovery. */ }
  }
  return {
    start(recording, duration) {
      if (!recording?.active || !recording.session_id || ![10, 15, 30].includes(duration))
        throw new Error("A timed take needs an active session and a supported duration.");
      plan = { id: recording.session_id, deadline: now() + duration * 1000, duration };
      retryAt = 0;
      try { storage?.setItem(key, JSON.stringify(plan)); } catch { /* Optional reload recovery. */ }
    },
    update(recording) {
      if (!plan) return null;
      if (!recording?.active || recording.session_id !== plan.id) {
        clear();
        return null;
      }
      const remaining = Math.max(0, Math.ceil((plan.deadline - now()) / 1000));
      const status = { ...plan, remaining, saving };
      if (!remaining && !saving && now() >= retryAt) {
        saving = true;
        const id = plan.id;
        Promise.resolve().then(() => onExpire(id)).then((saved) => {
          if (saved && plan?.id === id) clear();
          if (!saved) retryAt = now() + 2000;
        }).catch(() => { retryAt = now() + 2000; }).finally(() => { saving = false; });
      }
      return status;
    },
  };
}
