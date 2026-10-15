"""Check source/calibration boundaries and that exported sessions retain evidence."""
import json
import time
import pytest
from fastapi.testclient import TestClient
from arm_checker.app import Checker, create_app
from arm_checker.source import demo_frame


def feed(checker, elapsed, received):
    frame = demo_frame(elapsed, int(elapsed * 100), stationary=True)
    frame["received_at"] = received
    checker.ingest(frame)
    return frame


def ready(checker):
    base = time.monotonic() - 6
    feed(checker, 0, base)
    checker.motion.start_table(now=base)
    for index in range(1, 221):
        feed(checker, index / 100, base + index / 100)
    assert checker.motion.table_ready
    checker.motion.start_neutral(now=base + 2.2)
    for index in range(221, 541):
        feed(checker, index / 100, base + index / 100)
    now = time.monotonic()
    feed(checker, now - base, now)
    assert checker.pose()["valid"]
    return base


def test_recording_requires_fresh_calibrated_input(tmp_path):
    checker = Checker(sessions_dir=tmp_path)
    with pytest.raises(ValueError, match="Calibrate"):
        checker.record("start")
    ready(checker)
    checker.motion._latest["4"]["_advanced_at"] -= 1
    with pytest.raises(ValueError, match="Calibrate"):
        checker.record("start")
    assert checker.state()["sensors"][4]["ok"] is False


def test_record_download_and_replay_preserve_context(tmp_path):
    checker = Checker(sessions_dir=tmp_path)
    base = ready(checker)
    checker.record("start")
    for _ in range(3):
        now = time.monotonic()
        feed(checker, now - base, now)
    metadata = checker.record("stop")
    saved = json.loads(checker.store.path(metadata["id"]).read_text())
    assert metadata["synthetic"] is True
    assert metadata["frames"] == metadata["valid_pose_frames"] == 3
    assert saved["metadata"]["calibration"]["neutral_ready"] is True
    assert all(f["device_frame"]["synthetic"] for f in saved["frames"])
    assert all(f["host_elapsed_s"] >= 0 for f in saved["frames"])
    checker.start_replay(metadata["id"])
    state = checker.state()
    assert state["source"] == "replay"
    assert state["pose"] == saved["frames"][0]["pose"]
    assert state["sensors"] == []
    with pytest.raises(ValueError):
        checker.record("start")


def test_changes_cannot_reinterpret_active_recording(tmp_path):
    checker = Checker(sessions_dir=tmp_path)
    ready(checker)
    checker.record("start")
    for operation in (lambda: checker.configure({"upper_id": "4", "forearm_id": "3"}),
                      lambda: checker.change_source({"source": "demo"}),
                      lambda: checker.calibrate("reset")):
        with pytest.raises(ValueError, match="Stop recording"):
            operation()
    assert checker.pose()["valid"]
    checker.record("stop")


@pytest.mark.parametrize("event", [
    {"type": "connection", "connected": False, "message": "Disconnected"},
    {"type": "device_status", "status": {"event": "bias_start"}},
    {"type": "device_status", "status": {"boot_id": "new"}},
    {"type": "device_status", "status": {"orientation_generation": [0, 0, 0, 1, 0]}},
])
def test_device_reset_or_disconnect_invalidates_calibration(tmp_path, event):
    checker = Checker(sessions_dir=tmp_path)
    ready(checker)
    checker.ingest({"type": "device_status", "status": {"boot_id": "old", "orientation_generation": [0] * 5}})
    checker.ingest(event)
    assert not checker.motion.neutral_ready
    assert not checker.pose()["valid"]


def test_config_validation_is_atomic_and_mapping_requires_recalibration(tmp_path):
    checker = Checker(sessions_dir=tmp_path)
    ready(checker)
    with pytest.raises(ValueError):
        checker.configure({"upper_id": "4"})
    assert checker.config["upper_id"] == "3"
    assert checker.pose()["valid"]
    checker.configure({"upper_id": "4", "forearm_id": "3"})
    assert not checker.pose()["valid"]


def test_http_boundaries_and_session_path_validation(tmp_path):
    app = create_app(sessions_dir=tmp_path)
    client = TestClient(app)
    assert client.get("/api/state").json()["camera_synchronized"] is False
    assert client.post("/api/record", json={"action": "start"}).status_code == 400
    assert client.post("/api/source", json={"source": "unknown"}).status_code == 400
    assert client.post("/api/config", json={"upper_length": 99}).status_code == 400
    assert client.post("/api/calibrate", json={"step": "neutral"}).status_code == 400
    assert client.post("/api/replay", json={"id": "../private"}).status_code == 400
    assert client.post("/api/source", json={"source": "demo"},
                       headers={"Origin": "https://another-site.example"}).status_code == 403
    assert client.get("/api/sessions/missing/download").status_code == 404


def test_pre_start_serial_queue_is_not_included_in_recording(tmp_path):
    checker = Checker(sessions_dir=tmp_path)
    ready(checker)
    checker.record("start")
    checker.store.append({}, {}, checker.store.started - .1)
    assert checker.store.status()["frames"] == 0
    checker.record("stop")


def test_status_scalar_and_summary_share_per_sensor_generations(tmp_path):
    checker = Checker(sessions_dir=tmp_path)
    ready(checker)
    statuses = [
        {"event": "status", "orientation_generation": [1, 2, 3, 7, 8]},
        {"event": "calibration_motion", "sensor": 0, "orientation_generation": 1},
        {"event": "bias_ready", "sensor": 1, "orientation_generation": 2},
        {"event": "bias_ready", "sensor": 3, "orientation_generation": 7},
        {"event": "status", "orientation_generation": [1, 2, 3, 7, 8]},
        {"event": "status", "orientation_generation": [2, 2, 3, 7, 8]},
    ]
    before = checker.motion.generation
    for status in statuses:
        checker.ingest({"type": "device_status", "status": {"boot_id": "same", **status}})
        assert checker.motion.neutral_ready
        assert checker.motion.generation == before
    assert checker.device_identity["orientation_generation"] == {"0": 2, "1": 2, "2": 3, "3": 7, "4": 8}
    checker.ingest({"type": "device_status", "status": {"sensor": "4", "orientation_generation": 9}})
    assert not checker.motion.neutral_ready
    assert checker.motion.generation == before + 1


@pytest.mark.parametrize("event", ["sensor_init", "sample_gap", "dmp_fallback", "dmp_start", "sensor_stalled"])
def test_sensor_reset_events_only_invalidate_selected_ids(tmp_path, event):
    checker = Checker(sessions_dir=tmp_path)
    ready(checker)
    before = checker.motion.generation
    checker.ingest({"type": "device_status", "status": {"event": event, "sensor": 1, "orientation_generation": 11}})
    assert checker.motion.neutral_ready and checker.motion.generation == before
    checker.ingest({"type": "device_status", "status": {"event": event, "sensor": "3", "orientation_generation": 12}})
    assert not checker.motion.neutral_ready and checker.motion.generation == before + 1


def test_unrelated_status_does_not_cancel_active_table_capture(tmp_path):
    checker = Checker(sessions_dir=tmp_path)
    now = time.monotonic()
    feed(checker, 0, now)
    checker.motion.start_table(now=now)
    for status in ({"orientation_generation": [1, 2, 3, 4, 5]},
                   {"event": "calibration_motion", "sensor": 0, "orientation_generation": 1},
                   {"event": "sensor_init", "sensor": 2, "orientation_generation": 4},
                   {"orientation_generation": [1, 2, 4, 4, 5]}):
        checker.ingest({"type": "device_status", "status": status})
        assert checker.motion.stage == "table"


def test_unscoped_scalar_generation_remains_a_global_reset(tmp_path):
    checker = Checker(sessions_dir=tmp_path)
    ready(checker)
    for value in (3, 3):
        checker.ingest({"type": "device_status", "status": {"orientation_generation": value}})
        assert checker.motion.neutral_ready
    checker.ingest({"type": "device_status", "status": {"orientation_generation": 4}})
    assert not checker.motion.neutral_ready


@pytest.mark.parametrize("event", ["bias_start", "rescan_complete"])
def test_global_reset_events_remain_global(tmp_path, event):
    checker = Checker(sessions_dir=tmp_path)
    ready(checker)
    checker.ingest({"type": "device_status", "status": {"event": event}})
    assert not checker.motion.neutral_ready


def test_boot_change_clears_per_sensor_generation_history(tmp_path):
    checker = Checker(sessions_dir=tmp_path)
    ready(checker)
    checker.ingest({"type": "device_status", "status": {"boot_id": "old", "orientation_generation": [9] * 5}})
    checker.ingest({"type": "device_status", "status": {"boot_id": "new", "sensor": 0, "orientation_generation": 1}})
    assert not checker.motion.neutral_ready
    assert checker.device_identity["orientation_generation"] == {"0": 1}


def test_generation_filter_follows_configured_sensor_mapping(tmp_path):
    checker = Checker(sessions_dir=tmp_path)
    checker.configure({"upper_id": "0", "forearm_id": "1"})
    ready(checker)
    checker.ingest({"type": "device_status", "status": {"orientation_generation": [1] * 5}})
    checker.ingest({"type": "device_status", "status": {"event": "sensor_init", "sensor": 3, "orientation_generation": 2}})
    assert checker.motion.neutral_ready
    checker.ingest({"type": "device_status", "status": {"orientation_generation": [2, 1, 1, 2, 1]}})
    assert not checker.motion.neutral_ready
