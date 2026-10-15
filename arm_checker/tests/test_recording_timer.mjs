import test from "node:test";
import assert from "node:assert/strict";
import { createRecordingTimer } from "../web/recording_timer.js";

const flush = () => new Promise((resolve) => setImmediate(resolve));
const active = (id = "take-one") => ({ active: true, session_id: id });
function fixture() {
  let clock = 0;
  const values = new Map();
  return {
    now: () => clock,
    advance: (value) => { clock = value; },
    storage: { getItem: (key) => values.get(key), setItem: (key, value) => values.set(key, value), removeItem: (key) => values.delete(key) },
  };
}

test("each duration expires once after active recording begins", async () => {
  for (const duration of [10, 15, 30]) {
    const f = fixture(), stops = [];
    const timer = createRecordingTimer({ ...f, onExpire: (id) => { stops.push(id); return true; } });
    f.advance(5000); // The preparation countdown has already finished.
    timer.start(active(), duration);
    f.advance(5000 + duration * 1000 - 1);
    assert.equal(timer.update(active()).remaining, 1);
    await flush(); assert.equal(stops.length, 0);
    f.advance(5000 + duration * 1000);
    timer.update(active()); timer.update(active()); await flush();
    assert.deepEqual(stops, ["take-one"]);
    assert.equal(timer.update(active()), null);
  }
});

test("a manual stop or replacement session cancels the owned timer", async () => {
  for (const recording of [{ active: false, session_id: "take-one" }, active("take-two")]) {
    const f = fixture(); let stops = 0;
    const timer = createRecordingTimer({ ...f, onExpire: () => { stops++; return true; } });
    timer.start(active(), 15); f.advance(16000);
    assert.equal(timer.update(recording), null);
    await flush(); assert.equal(stops, 0);
  }
});

test("reload restores deadline and an overdue matching take is saved", async () => {
  const f = fixture(); let stops = 0;
  const initial = createRecordingTimer({ ...f, onExpire: () => { throw new Error("old page"); } });
  initial.start(active(), 15); f.advance(7000);
  const restored = createRecordingTimer({ ...f, onExpire: () => { stops++; return true; } });
  assert.equal(restored.update(active()).remaining, 8);
  f.advance(17000); restored.update(active()); await flush();
  assert.equal(stops, 1);
});

test("failed saves retry with backoff and do not cancel on stale sensor data", async () => {
  const f = fixture(); let attempts = 0;
  const timer = createRecordingTimer({ ...f, onExpire: () => ++attempts > 1 });
  timer.start(active(), 10); f.advance(10000);
  timer.update(active()); await flush();
  assert.equal(attempts, 1);
  f.advance(11000); timer.update(active()); await flush();
  assert.equal(attempts, 1);
  f.advance(12000); timer.update(active()); await flush();
  assert.equal(attempts, 2);
  assert.equal(timer.update(active()), null);
});

test("bad storage and invalid starts cannot create a stop timer", () => {
  const timer = createRecordingTimer({ storage: { getItem: () => "broken JSON" }, onExpire: () => true });
  assert.equal(timer.update(active()), null);
  assert.throws(() => timer.start({ active: false }, 15));
  assert.throws(() => timer.start(active(), 5));
});
