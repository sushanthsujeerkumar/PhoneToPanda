"""Small fabricated fixtures test provenance gates, never robot or human accuracy."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from phone_to_panda import learning, simulation


def write_json(path, value):
    path.write_text(json.dumps(value))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path):
    video, session_path, events = (tmp_path / name for name in ("fixture.MOV", "session.json", "events.json"))
    video.write_bytes(b"Test fixture only, not a real video or demonstration")
    calibration = {"mapping": {"upper": "3", "forearm": "4"}, "table_ready": True,
                   "neutral_ready": True, "stage": "ready", "calibration_version": 3, "generation": 1}
    session = {"metadata": {"id": "fixture", "schema_version": 1, "source": "serial", "synthetic": False,
                            "config": {"upper_id": "3", "forearm_id": "4"}, "calibration": calibration,
                            "frames": 4, "valid_pose_frames": 4}, "frames": []}
    for i in range(4):
        stamp = 100000 + i * 10000
        session["frames"].append({"host_elapsed_s": i / 100,
            "device_frame": {"t_us": stamp, "sensors": [
                {"id": identity, "t_us": stamp - 100, "ok": True, "reason": "", "flags": 31,
                 "age_ms": .1, "q": [1, 0, 0, 0]} for identity in ("3", "4")]},
            "pose": {"valid": True, "wrist": [.1 * i, 0, -.4], "calibration_version": 3, "generation": 1}})
    write_json(session_path, session)
    write_json(events, {"schema_version": 1, "annotation_method": "manual_video_review",
                        "video_grasp_time_s": 0., "video_release_time_s": .02,
                        "imu_time_offset_s": .005, "synchronization": "provisional"})
    source = {"kind": simulation.IMU_PILOT_KIND, "video": str(video), "sha256": digest(video),
              "imu_session": str(session_path), "imu_sha256": digest(session_path),
              "events": str(events), "events_sha256": digest(events)}
    quality = {"accepted": True, "estimated_wrist": True, "experimental": True, "synchronization": "provisional"}
    return source, quality, session


def test_verified_imu_source_retains_explicit_experimental_identity(tmp_path):
    source, quality, _ = fixture(tmp_path)
    result = simulation.validate_imu_pilot_source(source, tmp_path, quality)
    assert result["session_id"] == "fixture"
    assert result["frames"] == 4
    assert result["events"]["synchronization"] == "provisional"
    assert source["kind"] != "personal_video"


@pytest.mark.parametrize("path_key", ["video", "imu_session", "events"])
def test_each_original_hash_is_required_and_rechecked(tmp_path, path_key):
    source, quality, _ = fixture(tmp_path)
    from pathlib import Path
    Path(source[path_key]).write_bytes(b"changed source")
    with pytest.raises(ValueError, match="hash differs"):
        simulation.validate_imu_pilot_source(source, tmp_path, quality)


@pytest.mark.parametrize("bad", ["synthetic", "mapping", "calibration", "invalid_pose", "pose_generation",
                                "stale_sensor", "missing_sensor", "duplicate_clock", "sample_clock",
                                "gap", "nan_wrist", "frame_count"])
def test_acquisition_claims_cannot_hide_invalid_or_synthetic_input(tmp_path, bad):
    source, quality, session = fixture(tmp_path)
    if bad == "synthetic":
        session["metadata"]["synthetic"] = True
    elif bad == "mapping":
        session["metadata"]["config"]["upper_id"] = "0"
    elif bad == "calibration":
        session["metadata"]["calibration"]["neutral_ready"] = False
    elif bad == "invalid_pose":
        session["frames"][2]["pose"]["valid"] = False
    elif bad == "pose_generation":
        session["frames"][2]["pose"]["generation"] = 99
    elif bad == "stale_sensor":
        session["frames"][2]["device_frame"]["sensors"][1]["flags"] = 23
    elif bad == "missing_sensor":
        session["frames"][2]["device_frame"]["sensors"].pop()
    elif bad == "duplicate_clock":
        session["frames"][2]["host_elapsed_s"] = .01
    elif bad == "sample_clock":
        session["frames"][2]["device_frame"]["sensors"][1]["t_us"] = 109900
    elif bad == "gap":
        session["frames"][3]["host_elapsed_s"] = .2
    elif bad == "nan_wrist":
        session["frames"][2]["pose"]["wrist"][0] = float("nan")
    else:
        session["metadata"]["frames"] = 5
    from pathlib import Path
    path = Path(source["imu_session"])
    write_json(path, session)
    source["imu_sha256"] = digest(path)
    with pytest.raises(ValueError):
        simulation.validate_imu_pilot_source(source, tmp_path, quality)


@pytest.mark.parametrize("field", ["accepted", "estimated_wrist", "experimental", "synchronization"])
def test_experimental_limitations_must_survive_in_provenance(tmp_path, field):
    source, quality, _ = fixture(tmp_path)
    del quality[field]
    with pytest.raises(ValueError, match="disclose"):
        simulation.validate_imu_pilot_source(source, tmp_path, quality)


def test_imu_path_cannot_be_declared_marker_measured(tmp_path):
    source, quality, _ = fixture(tmp_path)
    trajectory = tmp_path / "trajectory.json"
    write_json(trajectory, {"schema_version": 1, "source": source, "quality": quality,
                            "coordinate_frame": "table_marker"})
    with pytest.raises(ValueError, match="schema/frame"):
        simulation.record_demo({"_root": str(tmp_path)}, tmp_path / "episode.npz", trajectory)
    assert not (tmp_path / "episode.npz").exists()


@pytest.mark.parametrize("change", ["method", "order", "offset", "outside", "session"])
def test_hashed_annotations_still_require_valid_manual_event_meaning(tmp_path, change):
    source, quality, _ = fixture(tmp_path)
    from pathlib import Path
    path = Path(source["events"])
    events = json.loads(path.read_text())
    if change == "method":
        events["annotation_method"] = "unknown"
    elif change == "order":
        events["video_release_time_s"] = -1
    elif change == "offset":
        events["imu_time_offset_s"] = float("nan")
    elif change == "outside":
        events["imu_time_offset_s"] = 10
    else:
        events["sensor_session_id"] = "another_take"
    write_json(path, events)
    source["events_sha256"] = digest(path)
    with pytest.raises(ValueError):
        simulation.validate_imu_pilot_source(source, tmp_path, quality)


def test_replay_rejects_cut_times_that_disagree_with_verified_events(tmp_path):
    source, quality, _ = fixture(tmp_path)
    trajectory = tmp_path / "trajectory.json"
    write_json(trajectory, {"schema_version": 1, "source": source, "quality": quality,
        "coordinate_frame": "fixed_shoulder", "grasp_time_s": .01, "release_time_s": .025})
    with pytest.raises(ValueError, match="cut times"):
        simulation.record_demo({"_root": str(tmp_path)}, tmp_path / "episode.npz", trajectory)


def recorded_trajectory(source, quality, session):
    return {"schema_version": 1, "coordinate_frame": "fixed_shoulder", "source": source,
            "quality": quality, "grasp_time_s": .005, "release_time_s": .025,
            "samples": [{"t": frame["host_elapsed_s"], "xyz": frame["pose"]["wrist"].copy(),
                         "origin": "recorded_imu_wrist_estimate"} for frame in session["frames"]]}


def test_replay_path_is_bound_to_verified_recorded_wrist_samples(tmp_path):
    source, quality, session = fixture(tmp_path)
    trajectory = recorded_trajectory(source, quality, session)
    verified = simulation.validate_imu_pilot_source(source, tmp_path, quality, trajectory=trajectory)
    assert verified["frames"] == len(trajectory["samples"])
    # Small JSON floating-point round-off is allowed, with no relative tolerance.
    trajectory["samples"][1]["xyz"][0] += 1e-10
    simulation.validate_imu_pilot_source(source, tmp_path, quality, trajectory=trajectory)


@pytest.mark.parametrize("change", ["position", "timestamp", "omit", "reorder", "nan", "origin", "boolean"])
def test_tampered_imu_paths_are_rejected_before_simulation(tmp_path, monkeypatch, change):
    source, quality, session = fixture(tmp_path)
    trajectory = recorded_trajectory(source, quality, session)
    samples = trajectory["samples"]
    if change == "position":
        samples[1]["xyz"][0] += .001
    elif change == "timestamp":
        samples[1]["t"] += .001
    elif change == "omit":
        samples.pop()
    elif change == "reorder":
        samples[1], samples[2] = samples[2], samples[1]
    elif change == "nan":
        samples[1]["xyz"][0] = float("nan")
    elif change == "origin":
        samples[1]["origin"] = "invented_position"
    else:
        samples[0]["t"] = False
    path = tmp_path / "trajectory.json"
    write_json(path, trajectory)
    monkeypatch.setattr(simulation, "make_env", lambda *a, **k: pytest.fail("Tampered data must not start physics"))
    with pytest.raises(ValueError, match="IMU pilot trajectory"):
        simulation.record_demo({"_root": str(tmp_path)}, tmp_path / "episode.npz", path)
    assert not (tmp_path / "episode.npz").exists()


def test_export_gate_requires_success_and_reverifies_imu_source(tmp_path):
    source, quality, _ = fixture(tmp_path)
    episode = tmp_path / "episode.npz"
    np.savez(episode, state=np.zeros((2, 8), np.float32), action=np.zeros((2, 7), np.float32),
             image=np.zeros((2, 256, 256, 3), np.uint8), image2=np.zeros((2, 256, 256, 3), np.uint8))
    metadata = {"source": source, "trajectory_quality": quality, "success": True, "task": "Fixture"}
    write_json(episode.with_suffix(".json"), metadata)
    _, checked = learning._validate_episode(episode, tmp_path)
    assert checked["source"]["kind"] == simulation.IMU_PILOT_KIND
    metadata["success"] = False
    write_json(episode.with_suffix(".json"), metadata)
    with pytest.raises(ValueError, match="unsuccessful"):
        learning._validate_episode(episode, tmp_path)
    metadata["success"] = True
    source["imu_sha256"] = "0" * 64
    write_json(episode.with_suffix(".json"), metadata)
    with pytest.raises(ValueError, match="hash differs"):
        learning._validate_episode(episode, tmp_path)


def test_training_rechecks_sources_before_starting_any_process(tmp_path, monkeypatch):
    source, quality, _ = fixture(tmp_path)
    provenance = tmp_path / "dataset/meta/phone_to_panda_provenance.json"
    provenance.parent.mkdir(parents=True)
    source["events_sha256"] = "0" * 64
    write_json(provenance, {"provenance": [{"source": source, "trajectory_quality": quality, "success": True}]})
    monkeypatch.setattr(learning.subprocess, "run", lambda *a, **k: pytest.fail("Training must not launch"))
    config = {"_root": str(tmp_path), "paths": {"dataset": "dataset"}, "training": {"device": "cpu"}}
    with pytest.raises(ValueError, match="hash differs"):
        learning.train(config)


def test_observer_receives_reset_and_actual_post_action_observation():
    initial = {"observation.state": np.zeros(8, np.float32),
               "observation.images.image": np.zeros((2, 2, 3), np.uint8),
               "observation.images.image2": np.zeros((2, 2, 3), np.uint8)}
    env = SimpleNamespace(cube_position=np.array([0, 0, .92]), steps=0, horizon=10,
                          orientation_control="fixed_downward")
    after = deepcopy(initial)
    after["observation.state"][0] = .01
    def step(action):
        env.steps += 1
        return after, 0., False, {}
    env.step = step
    observations = []
    recorder = simulation._Recorder(env, initial,
        observer=lambda e, obs, phase: observations.append((e.steps, float(obs["observation.state"][0]), phase)))
    recorder.tick([.01, 0, 0], 1., "carry_imu_estimated_wrist")
    assert observations == [(0, 0., "reset"), (1, pytest.approx(.01), "carry_imu_estimated_wrist")]
    assert recorder.states[0][0] == 0.  # Training still pairs pre-action observation with its next command.
