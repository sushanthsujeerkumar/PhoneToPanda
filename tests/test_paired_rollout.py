"""Synthetic-only validation of selected-seed video comparison and timing."""
import importlib.util
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("render_paired_rollout", Path(__file__).resolve().parents[1] / "scripts/render_paired_rollout.py")
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_fixture(root, phase="development"):
    plan = {"development": {"known_wide": {"seed": 3010, "episodes": 2, "radius_m": .06}},
            "final_test": {"wide": {"seed": 7100, "episodes": 2, "radius_m": .06}}}
    write_json(root / "study_plan.json", plan)
    group = "known_wide" if phase == "development" else "wide"
    settings = plan[phase][group]
    selected_seed = settings["seed"] + 1
    plan_hash = renderer.digest(root / "study_plan.json")
    identities = {name: {"checkpoint_files": {"model.safetensors": name}, "n_action_steps": 5} for name in ("baseline", "candidate")}
    context = {"config_sha256": hashlib.sha256(b"synthetic base configuration").hexdigest(),
               "source_sha256": {"phone_to_panda/learning.py": hashlib.sha256(b"synthetic policy code").hexdigest(),
                                 "phone_to_panda/simulation.py": hashlib.sha256(b"synthetic simulator code").hexdigest()}}
    if phase == "final_test":
        write_json(root / "candidate_lock.json", {"study_plan_sha256": plan_hash, "candidates": identities,
            "reference_candidate": "baseline", "selected_candidate": "candidate",
            "contrast_roles": {"candidate": "Primary selected adaptation"},
            "selection_sidecars_sha256": {"selection_rule.json": hashlib.sha256(b"synthetic selection rule").hexdigest()}, **context})
    for name in identities:
        folder = root / phase / name / group
        config = {"simulation": {"control_hz": 20, "image_size": 256, "horizon": 600, "orientation_control": "fixed_downward"}}
        write_json(folder / "manifest.json", {"candidate": name, "phase": phase, "group": group,
            "episode_plan": settings, "study_plan_sha256": plan_hash, "identity": identities[name], "config": config,
            "source_sha256": context["source_sha256"],
            **({"base_config_sha256": context["config_sha256"],
                "candidate_lock_sha256": renderer.digest(root / "candidate_lock.json")} if phase == "final_test" else {})})
        success, steps = (False, 2) if name == "baseline" else (True, 4)
        # Episode 000 intentionally has no trace/video: the renderer must only
        # read selected seed 001 while checking the group's report seed list.
        episodes = [{"seed": settings["seed"], "success": False, "steps": 2, "video": "episode_000.mp4"},
                    {"seed": selected_seed, "success": success, "steps": steps, "video": "episode_001.mp4"}]
        write_json(folder / "evaluation/report.json", {"kind": "trained_policy_physics_evaluation",
            "seed_start": settings["seed"], "episodes": episodes, "randomize_start_m": .06,
            "orientation_control": "fixed_downward", "policy_settings": {"n_action_steps": 5}, "video_recorded": True})
        trace = [{"step": index, "episode_seed": selected_seed, "cube_xyz": [0., -.17, .92],
                  "observation_state": [0., -.15, 1.1, 0., 0., 0., .04, -.04],
                  "success": success and index == steps - 1} for index in range(steps)]
        write_json(folder / "evaluation/episode_001_trace.json", trace)
        with imageio.get_writer(folder / "evaluation/episode_001.mp4", fps=20, codec="libx264", quality=8,
                                macro_block_size=1, ffmpeg_params=["-threads", "1"]) as writer:
            for index in range(steps + 1):
                frame = np.empty((256, 512, 3), dtype=np.uint8)
                frame[:] = [30 + index * 30, 25, 35] if name == "baseline" else [20, 50 + index * 30, 35]
                frame[:, 256:] = frame[:, :256] // 2
                writer.append_data(frame)
    return root, group, selected_seed


def test_renderer_uses_only_selected_seed_preserves_fps_and_holds_last_frame(tmp_path):
    study, group, seed = make_fixture(tmp_path / "study")
    output = tmp_path / "paired.mp4"
    evidence = renderer.render_pair(study, "development", "baseline", "candidate", group, seed, output)
    assert evidence["paired_initial_cube_and_robot_states_verified"]
    assert evidence["output_frames"] == 5
    assert evidence["models"][0]["success"] is False
    assert evidence["models"][1]["success"] is True
    assert evidence["models"][0]["video_frames"] == 3
    assert evidence["output_sha256"] == renderer.digest(output)
    with imageio.get_reader(output) as reader:
        assert reader.get_meta_data()["fps"] == 20
        assert reader.count_frames() == 5
        last = reader.get_data(4)
        assert last.shape == (renderer.HEIGHT, 512, 3)
        # Baseline's final camera pixels remain held while the candidate runs.
        np.testing.assert_allclose(last[180, 100].astype(float), reader.get_data(2)[180, 100].astype(float), atol=5)
    with pytest.raises(ValueError, match="never overwritten"):
        renderer.render_pair(study, "development", "baseline", "candidate", group, seed, output)


def test_initial_state_mismatch_rejected_before_output(tmp_path):
    study, group, seed = make_fixture(tmp_path / "study")
    path = study / "development/candidate" / group / "evaluation/episode_001_trace.json"
    trace = json.loads(path.read_text())
    trace[0]["observation_state"][0] = 1e-12
    write_json(path, trace)
    with pytest.raises(ValueError, match="initial_observation_state differs"):
        renderer.render_pair(study, "development", "baseline", "candidate", group, seed, tmp_path / "bad.mp4")
    assert not (tmp_path / "bad.mp4").exists()


def test_final_lock_and_report_video_identity_checked(tmp_path):
    study, group, seed = make_fixture(tmp_path / "study", phase="final_test")
    selected, _ = renderer.validate_pair(study, "final_test", "baseline", "candidate", group, seed)
    assert selected[0]["seed"] == seed
    path = study / "candidate_lock.json"
    lock = json.loads(path.read_text())
    lock["candidates"]["candidate"]["n_action_steps"] = 1
    write_json(path, lock)
    with pytest.raises(ValueError, match="differs from lock"):
        renderer.validate_pair(study, "final_test", "baseline", "candidate", group, seed)
    lock["candidates"]["candidate"]["n_action_steps"] = 5
    write_json(path, lock)
    path = study / "final_test/candidate" / group / "evaluation/report.json"
    report = json.loads(path.read_text())
    report["episodes"][1]["video"] = "../../unrelated.mp4"
    write_json(path, report)
    with pytest.raises(ValueError, match="canonical video"):
        renderer.validate_pair(study, "final_test", "baseline", "candidate", group, seed)


def test_mismatched_video_frame_count_is_not_silently_stretched(tmp_path):
    study, group, seed = make_fixture(tmp_path / "study")
    path = study / "development/candidate" / group / "evaluation/episode_001.mp4"
    imageio.mimwrite(path, [np.zeros((256, 512, 3), dtype=np.uint8)], fps=20, codec="libx264", macro_block_size=1)
    with pytest.raises(ValueError, match="frame count differs"):
        renderer.render_pair(study, "development", "baseline", "candidate", group, seed, tmp_path / "bad.mp4")
    assert not (tmp_path / "bad.mp4").exists()


@pytest.mark.parametrize("field,message", [("source_sha256", "source code"), ("base_config_sha256", "base configuration")])
@pytest.mark.parametrize("remove", [True, False])
def test_final_render_requires_locked_source_and_base_config(tmp_path, field, message, remove):
    study, group, seed = make_fixture(tmp_path / "study", phase="final_test")
    path = study / "final_test/candidate" / group / "manifest.json"
    manifest = json.loads(path.read_text())
    if remove:
        del manifest[field]
    else:
        manifest[field] = "changed"
    write_json(path, manifest)
    with pytest.raises(ValueError, match=message):
        renderer.validate_pair(study, "final_test", "baseline", "candidate", group, seed)


@pytest.mark.parametrize("field", ["source_sha256", "config_sha256"])
def test_final_render_rejects_old_lock_missing_context(tmp_path, field):
    study, group, seed = make_fixture(tmp_path / "study", phase="final_test")
    path = study / "candidate_lock.json"
    lock = json.loads(path.read_text())
    del lock[field]
    write_json(path, lock)
    with pytest.raises(ValueError, match="differs from lock"):
        renderer.validate_pair(study, "final_test", "baseline", "candidate", group, seed)


def test_final_reference_cannot_be_swapped_and_selection_cannot_change(tmp_path):
    study, group, seed = make_fixture(tmp_path / "study", phase="final_test")
    with pytest.raises(ValueError, match="locked reference_candidate"):
        renderer.validate_pair(study, "final_test", "candidate", "baseline", group, seed)
    path = study / "candidate_lock.json"
    lock = json.loads(path.read_text())
    lock["selected_candidate"] = "baseline"
    write_json(path, lock)
    with pytest.raises(ValueError, match="selection differs from lock"):
        renderer.validate_pair(study, "final_test", "baseline", "candidate", group, seed)


def test_final_secondary_comparison_retains_original_selection_label(tmp_path):
    study, group, seed = make_fixture(tmp_path / "study", phase="final_test")
    path = study / "candidate_lock.json"
    lock = json.loads(path.read_text())
    lock["selected_candidate"] = "baseline"
    lock["contrast_roles"]["candidate"] = "Secondary exploratory adaptation"
    write_json(path, lock)
    for manifest_path in (study / "final_test").glob("*/*/manifest.json"):
        manifest = json.loads(manifest_path.read_text())
        manifest["candidate_lock_sha256"] = renderer.digest(path)
        write_json(manifest_path, manifest)
    selected, _ = renderer.validate_pair(study, "final_test", "baseline", "candidate", group, seed)
    assert selected[1]["selected_candidate"] == "baseline"
    assert selected[1]["selected_contrast"] is False
    assert selected[1]["contrast_role"] == "Secondary exploratory adaptation"
