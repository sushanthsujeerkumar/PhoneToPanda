import test from "node:test";
import assert from "node:assert/strict";
import { createRecordingCountdown, recordingContext } from "../web/recording_countdown.js";

function fixture() {
  let clock = 0, context = "original", scheduled = null, starts = 0, cancellations = 0;
  const countdown = createRecordingCountdown({
    now: () => clock, getContext: () => context,
    schedule: (callback) => { scheduled = callback; return 1; },
    unschedule: () => { scheduled = null; }, onChange: () => {},
    onStart: () => { starts++; }, onCancel: () => { cancellations++; },
  });
  return { countdown,
    get starts() { return starts; }, get cancellations() { return cancellations; },
    setContext: (value) => { context = value; },
    advance: (milliseconds) => { clock = milliseconds; const tick = scheduled; scheduled = null; tick?.(); },
  };
}

test("waits five seconds, starts once, and ignores duplicate starts", () => {
  const f = fixture();
  f.countdown.start();
  assert.equal(f.countdown.seconds, 5);
  f.advance(1000);
  f.countdown.start();
  assert.equal(f.countdown.seconds, 4);
  f.advance(4999);
  assert.equal(f.starts, 0);
  f.advance(5000);
  assert.equal(f.starts, 1);
  assert.equal(f.countdown.pending, false);
  f.advance(6000);
  assert.equal(f.starts, 1);
});

test("cancel and page exit leave no delayed start, and a fresh countdown can restart", () => {
  const f = fixture();
  f.countdown.start(); f.advance(2000); f.countdown.cancel(); f.advance(6000);
  assert.equal(f.starts, 0);
  f.countdown.start(); f.advance(10999);
  assert.equal(f.starts, 0);
  f.advance(11000);
  assert.equal(f.starts, 1);
});

test("lost input or changed calibration cancels, including at the deadline", () => {
  for (const context of [null, "changed calibration"]) {
    const f = fixture();
    f.countdown.start(); f.setContext(context); f.advance(5000);
    assert.equal(f.starts, 0);
    assert.equal(f.cancellations, 1);
    assert.equal(f.countdown.pending, false);
    f.setContext("original"); f.advance(6000);
    assert.equal(f.starts, 0);
  }
});

test("a throttled timer never starts early", () => {
  const f = fixture();
  f.countdown.start(); f.advance(7000);
  assert.equal(f.starts, 1);
});

test("context checks the assigned pair, setup, calibration, and recording state", () => {
  const state = { connected: true, source: "serial", pose: { valid: true },
    recording: { active: false }, config: { upper_id: "3", forearm_id: "4", upper_length: .3, forearm_length: .26 },
    calibration: { calibration_version: 3, generation: 1 },
    sensors: [{ id: "3", ok: true, age_ms: 2 }, { id: "4", ok: true, age_ms: 2 }, { id: "0", ok: false, age_ms: 9999 }],
  };
  const original = recordingContext(state, true, 0);
  assert.notEqual(original, null);
  assert.equal(recordingContext(state, false, 0), null);
  assert.equal(recordingContext(state, true, 250), null);
  for (const change of [
    s => { s.sensors[1].age_ms = null; }, s => { s.sensors[0].ok = false; },
    s => { s.pose.valid = false; }, s => { s.source = "replay"; },
    s => { s.recording.active = true; }, s => { s.connected = false; },
  ]) {
    const copy = structuredClone(state); change(copy);
    assert.equal(recordingContext(copy, true, 0), null);
  }
  for (const change of [
    s => { s.calibration.generation++; }, s => { s.calibration.calibration_version++; },
    s => { s.config.upper_length = .31; }, s => { s.source = "demo"; },
    s => { s.config.upper_id = "4"; s.config.forearm_id = "3"; },
  ]) {
    const copy = structuredClone(state); change(copy);
    assert.notEqual(recordingContext(copy, true, 0), original);
  }
});
