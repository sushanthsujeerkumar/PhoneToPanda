"""Small replay fixtures exercise selection/provenance; no simulator or training."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from scripts import build_selected_dataset as builder


def episode(path, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, state=np.zeros((2, 8), np.float32), action=np.zeros((2, 7), np.float32),
                        image=np.zeros((2, 256, 256, 3), np.uint8), image2=np.zeros((2, 256, 256, 3), np.uint8))
    builder.write_json(path.with_suffix(".json"), metadata)


def fixture(tmp_path, clips=("one",)):
    config = {"_root": str(tmp_path), "task": "Place cube in box.", "batch": {"clips": list(clips)},
              "paths": {"raw": "raw", "processed": "processed", "annotations": "annotations", "calibration": "camera.json", "robot": "selected", "dataset": "dataset"},
              "video": {"table_marker_size_m": .05, "object_marker_size_m": .03, "sample_hz": 30, "max_gap_s": .2, "tracking_mode": "multi_marker", "reference_markers": []},
              "simulation": {"control_hz": 20, "randomize_start_m": .035, "orientation_control": "fixed_downward"}}
    for name in ("raw", "processed", "annotations"):
        (tmp_path / name).mkdir()
    camera = tmp_path / "camera.json"
    camera.write_text("{}")
    sources = {}
    for ci, clip in enumerate(clips):
        video, events = tmp_path / f"raw/{clip}.MOV", tmp_path / f"annotations/{clip}.json"
        video.write_bytes(b"fixture bytes, never new personal recording evidence")
        events.write_text("{}")
        source = {"kind": "personal_video", "video": f"raw/{clip}.MOV", "sha256": builder.digest(video), "calibration_sha256": builder.digest(camera), "events_sha256": builder.digest(events)}
        sources[clip] = source
        track = {"source": source, "markers": {"table_marker_size_m": .05, "object_marker_size_m": .03, "reference_markers": [], "object_id": 1},
                 "quality": {"accepted": True, "tracking_mode": "multi_marker", "sample_hz": 30, "max_allowed_gap_s": .2, "measured_fraction": 1., "max_pose_reprojection_error_px": .2, "max_table_corner_drift_px": 1.}}
        builder.write_json(tmp_path / f"processed/{clip}.json", track)
        for offset in range(4):
            seed = ci * 4 + offset
            episode(tmp_path / f"data/robot/main/{clip}_seed_{seed}.npz",
                    {"seed": seed, "success": True, "source": source, "task": config["task"], "simulation_settings": config["simulation"]})
    plan = {"selection_name": "targeted", "successes_per_clip": 1,
            "clips": {clip: [{"seed": 100 + ci * 10 + k, "cube_start_xy": [.01 * k, -.17]} for k in range(3)] for ci, clip in enumerate(clips)}}
    return config, plan, sources


def replay_function(sources, failed=()):
    def replay(settings, output, trajectory_path, seed):
        report = {"success": seed not in failed, "seed": seed, "frames": 2,
                  "failure": "failed grasp" if seed in failed else None, "source": sources[trajectory_path.stem],
                  "task": settings["task"], "simulation_settings": deepcopy(settings["simulation"])}
        episode(output, report)
        return report
    return replay


def fake_export(settings, demos_dir, output_dir):
    provenance = []
    for path in sorted(demos_dir.glob("*.npz")):
        arrays, metadata = builder._validate_episode(path, Path(settings["_root"]))
        provenance.append({"episode_index": len(provenance), "frames": len(arrays["action"]), **metadata})
    report = {"episodes": len(provenance), "provenance": provenance}
    builder.write_json(output_dir / "meta/phone_to_panda_provenance.json", report)
    return report


def test_exact_starts_balanced_first_success_and_failure_retention(tmp_path):
    config, plan, sources = fixture(tmp_path, ("one", "two"))
    before = deepcopy(config)
    plan["clips"]["one"][1]["box_xy"] = [.02, .19]
    with patch.object(builder, "record_demo", side_effect=replay_function(sources, failed={100})) as replay:
        result = builder.build_selected_dataset(config, plan, skip_export=True)
    assert result["complete"] is True
    assert result["new_successes_per_clip"] == {"one": 1, "two": 1}
    assert result["attempted_candidates"] == 3
    assert result["failed_candidates"] == 1
    assert [call.kwargs["seed"] for call in replay.call_args_list] == [100, 101, 110]
    assert config == before
    metadata = builder.read_json(tmp_path / "selected/one_seed_101.json")
    assert metadata["simulation_settings"]["cube_start_xy"] == [.01, -.17]
    assert metadata["simulation_settings"]["box_xy"] == [.02, .19]
    assert metadata["simulation_settings"]["randomize_start_m"] == 0
    assert metadata["selection"]["candidate_index"] == 1
    assert metadata["source"] == sources["one"]
    assert len(list((tmp_path / "selected").glob("*.npz"))) == 10
    assert not (tmp_path / "selected/one_seed_100.npz").exists()
    assert (tmp_path / "selected/build/candidates/one_seed_100.npz").is_file()
    assert not (tmp_path / "selected/build/candidates/one_seed_102.npz").exists()


def test_bounded_resume_and_export_only_never_rerun_simulation(tmp_path):
    config, plan, sources = fixture(tmp_path)
    with patch.object(builder, "record_demo", side_effect=replay_function(sources, failed={100})) as replay:
        result = builder.build_selected_dataset(config, plan, skip_export=True, max_jobs=1)
        assert result["complete"] is False
        assert not list((tmp_path / "selected").glob("*.npz"))
        with pytest.raises(ValueError, match="coverage"):
            builder.build_selected_dataset(config, plan, export_only=True)
        assert replay.call_count == 1
        result = builder.build_selected_dataset(config, plan, skip_export=True, max_jobs=1)
        assert result["complete"] is True
        assert replay.call_count == 2
    with patch.object(builder, "record_demo") as replay, patch.object(builder, "export_dataset", side_effect=fake_export) as export:
        result = builder.build_selected_dataset(config, plan, export_only=True)
        assert result["exported_episodes"] == 5
        builder.build_selected_dataset(config, plan, export_only=True)
        replay.assert_not_called()
        assert export.call_count == 1


@pytest.mark.parametrize("change", ["plan", "task", "simulation", "original", "trajectory"])
def test_changed_inputs_are_rejected_without_replay(tmp_path, change):
    config, plan, sources = fixture(tmp_path)
    with patch.object(builder, "record_demo", side_effect=replay_function(sources)):
        builder.build_selected_dataset(config, plan, skip_export=True)
    if change == "plan":
        plan["clips"]["one"][0]["cube_start_xy"] = [.09, -.18]
    elif change == "task":
        config["task"] = "Changed task."
    elif change == "simulation":
        config["simulation"]["horizon"] = 500
    elif change == "original":
        path = tmp_path / "data/robot/main/one_seed_0.json"
        metadata = builder.read_json(path)
        metadata["extra"] = "changed original"
        builder.write_json(path, metadata)
    elif change == "trajectory":
        path = tmp_path / "processed/one.json"
        metadata = builder.read_json(path)
        metadata["extra"] = "changed trajectory"
        builder.write_json(path, metadata)
    with patch.object(builder, "record_demo") as replay:
        with pytest.raises(ValueError, match="changed|differs"):
            builder.build_selected_dataset(config, plan, skip_export=True)
        replay.assert_not_called()


@pytest.mark.parametrize("artifact", ["candidate", "copy", "incomplete"])
def test_changed_and_incomplete_artifacts_are_not_overwritten(tmp_path, artifact):
    config, plan, sources = fixture(tmp_path)
    with patch.object(builder, "record_demo", side_effect=replay_function(sources)):
        builder.build_selected_dataset(config, plan, skip_export=True)
    if artifact == "candidate":
        path = tmp_path / "selected/build/candidates/one_seed_100.npz"
        path.write_bytes(b"changed")
    elif artifact == "copy":
        path = tmp_path / "selected/one_seed_100.json"
        path.write_text("{}")
    else:
        path = tmp_path / "selected/build/candidates/one_seed_100.receipt.json"
        path.unlink()
    with patch.object(builder, "record_demo") as replay:
        with pytest.raises(ValueError, match="differs|Incomplete"):
            builder.build_selected_dataset(config, plan, skip_export=True)
        replay.assert_not_called()


def test_insufficient_successes_never_export_and_retain_all_attempts(tmp_path):
    config, plan, sources = fixture(tmp_path)
    with patch.object(builder, "record_demo", side_effect=replay_function(sources, failed={100, 101, 102})), patch.object(builder, "export_dataset") as export:
        with pytest.raises(ValueError, match="coverage"):
            builder.build_selected_dataset(config, plan)
        export.assert_not_called()
    report = builder.read_json(tmp_path / "selected/build/summary.json")
    assert report["failed_candidates"] == 3
    assert len(list((tmp_path / "selected/build/candidates").glob("*.npz"))) == 3


@pytest.mark.parametrize("bad_xy", [[float("nan"), 0], [float("inf"), 0], [True, 0], [0], "0,1"])
def test_invalid_exact_starts_rejected(bad_xy):
    plan = {"selection_name": "uniform", "successes_per_clip": 1, "clips": {"one": [{"seed": 50, "cube_start_xy": bad_xy}]}}
    with pytest.raises(ValueError, match="finite numbers"):
        builder.validate_plan(plan, ["one"])


def test_original_seed_collision_and_overwrite_rejected(tmp_path):
    config, plan, sources = fixture(tmp_path)
    plan["clips"]["one"][0]["seed"] = 0
    with patch.object(builder, "record_demo") as replay:
        with pytest.raises(ValueError, match="overlap"):
            builder.build_selected_dataset(config, plan, skip_export=True)
        with pytest.raises(ValueError, match="preserve original"):
            builder.build_selected_dataset(config, plan, robot_dir="data/robot/main", skip_export=True)
        replay.assert_not_called()
