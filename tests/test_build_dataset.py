"""Dataset-build safety checks use tiny fixtures, never task training."""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from scripts import build_dataset as builder


def fixture(tmp_path):
    paths = {"raw": "raw", "processed": "processed", "annotations": "annotations", "calibration": "camera.json", "robot": "robot", "dataset": "dataset"}
    for name in ("raw", "processed", "annotations"):
        (tmp_path / name).mkdir()
    video = tmp_path / "raw/clip.MOV"
    video.write_bytes(b"test fixture, not recorded video")
    calibration, events = tmp_path / "camera.json", tmp_path / "annotations/clip.json"
    calibration.write_text("{}")
    events.write_text("{}")
    config = {"_root": str(tmp_path), "paths": paths, "task": "Place cube in box.", "batch": {"clips": ["clip"]},
              "video": {"table_marker_size_m": .05, "object_marker_size_m": .03, "sample_hz": 30, "max_gap_s": .2, "tracking_mode": "multi_marker", "reference_markers": []},
              "simulation": {"control_hz": 20, "orientation_control": "fixed_downward"},
              "demonstrations": {"replays_per_clip": 2, "seed_start": 0, "minimum_successes": 1, "minimum_successes_per_clip": 1}}
    trajectory = tmp_path / "processed/clip.json"
    track = {"source": {"kind": "personal_video", "video": "raw/clip.MOV", "sha256": builder.digest(video), "calibration_sha256": builder.digest(calibration), "events_sha256": builder.digest(events)},
             "markers": {"table_marker_size_m": .05, "object_marker_size_m": .03, "reference_markers": [], "object_id": 1},
             "quality": {"accepted": True, "tracking_mode": "multi_marker", "sample_hz": 30, "max_allowed_gap_s": .2, "measured_fraction": 1., "max_pose_reprojection_error_px": .2, "max_table_corner_drift_px": 1.}}
    builder.write_json(trajectory, track)
    return config, trajectory, video, events, calibration


def test_changed_annotation_is_rejected_before_any_replay(tmp_path):
    config, trajectory, video, events, calibration = fixture(tmp_path)
    builder.validate_track(config, "clip", trajectory, video, events, calibration)
    events.write_text('{"grasp_time_s": 9}')
    with patch.object(builder, "record_demo") as replay:
        with pytest.raises(ValueError, match="events_sha256"):
            builder.build_dataset(config, skip_prepare=True)
        replay.assert_not_called()


def test_failed_replays_are_preserved_outside_export_and_resume_is_checked(tmp_path):
    config, trajectory, *_ = fixture(tmp_path)
    source = builder.read_json(trajectory)["source"]
    def replay(settings, output, trajectory_path, seed):
        np.savez(output, state=np.zeros((2, 8), np.float32), action=np.zeros((2, 7), np.float32),
                 image=np.zeros((2, 256, 256, 3), np.uint8), image2=np.zeros((2, 256, 256, 3), np.uint8))
        return {"success": seed == 0, "seed": seed, "frames": 2, "failure": None if seed == 0 else "contact timeout",
                "source": source, "task": config["task"], "simulation_settings": config["simulation"]}
    def export(settings):
        episodes = sorted((tmp_path / "robot").glob("*.npz"))
        assert [p.name for p in episodes] == ["clip_seed_0.npz"]
        metadata = builder.read_json(episodes[0].with_suffix(".json"))
        report = {"episodes": 1, "provenance": [{"episode_index": 0, "frames": 2, **metadata}]}
        builder.write_json(tmp_path / "dataset/meta/phone_to_panda_provenance.json", report)
        return report
    with patch.object(builder, "record_demo", side_effect=replay) as record, patch.object(builder, "export_dataset", side_effect=export) as writer:
        result = builder.build_dataset(config, skip_prepare=True)
        assert result["exported_episodes"] == 1
        assert (tmp_path / "robot/build/rejected_replays/clip_seed_1.npz").exists()
        assert builder.read_json(tmp_path / "robot/build/replays.json")[1]["failure"] == "contact timeout"
        builder.build_dataset(config, skip_prepare=True)
        assert record.call_count == 2
        assert writer.call_count == 1
        config["simulation"]["randomize_start_m"] = .1
        with pytest.raises(ValueError, match="Replay inputs changed"):
            builder.build_dataset(config, skip_prepare=True)
