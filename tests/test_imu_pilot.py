import json

import pytest

from phone_to_panda.imu_pilot import prepare_trajectory


def inputs(tmp_path):
    video, session, events = [tmp_path / name for name in ("clip.mov", "session.json", "events.json")]
    video.write_bytes(b"video fixture")
    recording = {"metadata": {"id": "recording", "source": "serial", "synthetic": False,
        "calibration": {"stage": "ready", "table_ready": True, "neutral_ready": True,
            "mapping": {"upper": "3", "forearm": "4"}, "calibration_version": 2,
            "lengths_m": {"upper": .30, "forearm": .26}}}, "frames": []}
    for index in range(8):
        recording["frames"].append({"host_elapsed_s": index * .01,
            "pose": {"valid": True, "calibration_version": 2,
                     "points": {"wrist": [.2, index * .01, -.3]}},
            "device_frame": {"sensors": [{"id": "3", "ok": True}, {"id": "4", "ok": True}]}})
    session.write_text(json.dumps(recording))
    events.write_text(json.dumps({"schema_version": 1, "annotation_method": "manual_video_review",
        "synchronization": "provisional", "video_grasp_time_s": .01,
        "video_release_time_s": .05, "imu_time_offset_s": .01}))
    return video, session, events, tmp_path / "trajectory.json"


def test_preserves_estimates_and_shifted_manual_events(tmp_path):
    paths = inputs(tmp_path)
    result = prepare_trajectory(*paths)
    saved = json.loads(paths[-1].read_text())
    assert result["grasp_time_s"] == pytest.approx(.02)
    assert saved["release_time_s"] == pytest.approx(.06)
    assert saved["samples"][3]["xyz"] == [.2, .03, -.3]
    assert saved["coordinate_frame"] == "fixed_shoulder"
    assert saved["quality"]["camera_position_fusion"] is False
    with pytest.raises(FileExistsError):
        prepare_trajectory(*paths)


@pytest.mark.parametrize("field,value", [("sensor_session_id", "different_take"), ("video_file", "other.mov")])
def test_rejects_events_for_a_different_recording_before_writing(tmp_path, field, value):
    paths = inputs(tmp_path)
    labels = json.loads(paths[2].read_text())
    labels[field] = value
    paths[2].write_text(json.dumps(labels))
    with pytest.raises(ValueError, match="different"):
        prepare_trajectory(*paths)
    assert not paths[-1].exists()


def test_accepts_explicit_event_identifiers_for_the_selected_pair(tmp_path):
    paths = inputs(tmp_path)
    labels = json.loads(paths[2].read_text())
    labels.update(sensor_session_id="recording", video_file=paths[0].name)
    paths[2].write_text(json.dumps(labels))
    result = prepare_trajectory(*paths)
    assert result["source"]["session_id"] == "recording"


@pytest.mark.parametrize("defect", ["synthetic", "invalid_pose", "sensor_missing", "time_gap", "event_outside"])
def test_rejects_unusable_inputs_without_writing(tmp_path, defect):
    paths = inputs(tmp_path)
    recording = json.loads(paths[1].read_text())
    if defect == "synthetic":
        recording["metadata"]["synthetic"] = True
    elif defect == "invalid_pose":
        recording["frames"][3]["pose"]["valid"] = False
    elif defect == "sensor_missing":
        recording["frames"][3]["device_frame"]["sensors"].pop()
    elif defect == "time_gap":
        recording["frames"][-1]["host_elapsed_s"] = .20
    else:
        labels = json.loads(paths[2].read_text()); labels["imu_time_offset_s"] = 1.
        paths[2].write_text(json.dumps(labels))
    paths[1].write_text(json.dumps(recording))
    with pytest.raises(ValueError):
        prepare_trajectory(*paths)
    assert not paths[-1].exists()
