"""Prefix provenance and label-boundary checks without model inference."""
import importlib.util
from copy import deepcopy
from pathlib import Path
import time

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("build_recovery_dataset", Path(__file__).resolve().parents[1] / "scripts/build_recovery_dataset.py")
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


class PrefixEnv:
    def __init__(self):
        self.reset_count, self.steps = 0, 0
        self.cube_position = np.array([0., -.17, .92])
        self._max_cube_height = .92
        self.actions = []
    def observation(self):
        return {"observation.state": np.array([self.steps * .01, 0., 1., 0., 0., 0., .04, -.04], np.float32)}
    def reset(self):
        self.reset_count += 1
        self.steps = 0
        return self.observation()
    def step(self, action):
        self.actions.append(action.copy())
        self.steps += 1
        return self.observation(), 0., False, {"applied_action": action.tolist()}
    def success(self):
        return False


def trace_fixture():
    env = PrefixEnv()
    trace = []
    for index in range(4):
        env.steps = index
        action = [index * .1, 0., 0., 0., 0., 0., -1.]
        trace.append({"step": index, "observation_state": env.observation()["observation.state"].tolist(),
                      "cube_xyz": env.cube_position.tolist(), "policy_action": action, "applied_action": action.copy()})
    return trace


def test_prefix_replays_exact_actions_once_and_recorder_starts_empty_at_takeover():
    env, trace = PrefixEnv(), trace_fixture()
    frames = []
    obs, evidence = pilot.replay_prefix(env, trace, 3, deadline=time.monotonic() + 10,
                                       write_frame=lambda obs, phase, step: frames.append((phase, step)))
    assert env.reset_count == 1 and env.steps == 3
    np.testing.assert_array_equal(env.actions, np.asarray([row["policy_action"] for row in trace[:3]], np.float32))
    assert evidence["takeover_state_check"]["max_state_error"] == 0.
    assert evidence["policy_actions_sha256"] == pilot.action_digest([row["policy_action"] for row in trace[:3]])
    recorder = pilot.PilotRecorder(env, obs, time.monotonic() + 10, None)
    assert recorder.actions == [] and recorder.states == []
    np.testing.assert_array_equal(recorder.obs["observation.state"], trace[3]["observation_state"])
    assert frames == [("UNLABELLED POLICY PREFIX", index) for index in range(3)]


def test_prefix_state_or_action_mismatch_stops_before_expert_labels():
    trace = trace_fixture()
    trace[3]["cube_xyz"][0] += 1e-4
    with pytest.raises(ValueError, match="state reproduction failed"):
        pilot.replay_prefix(PrefixEnv(), trace, 3, deadline=time.monotonic() + 10)
    trace = trace_fixture()
    trace[1]["applied_action"][0] += .1
    with pytest.raises(ValueError, match="Applied prefix actions differ"):
        pilot.replay_prefix(PrefixEnv(), trace, 3, deadline=time.monotonic() + 10)


def test_deadline_prevents_even_reset_and_takeover_requires_next_state():
    env = PrefixEnv()
    with pytest.raises(TimeoutError):
        pilot.replay_prefix(env, trace_fixture(), 3, deadline=time.monotonic() - 1)
    assert env.reset_count == 0
    with pytest.raises(ValueError, match="contiguous"):
        pilot.replay_prefix(env, trace_fixture(), 4, deadline=time.monotonic() + 10)


def test_full_mode_timer_covers_blocking_work_and_restores_handler():
    import signal
    previous = signal.getsignal(signal.SIGALRM)
    with pytest.raises(TimeoutError, match="collection wall-time"):
        with pilot.wall_timer(.01):
            time.sleep(.1)
    assert signal.getsignal(signal.SIGALRM) == previous
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)


def test_recovery_expert_uses_reached_cube_and_never_resets_environment():
    from types import SimpleNamespace
    events = []
    cube = np.array([.02, -.19, .92])
    env = SimpleNamespace(settings={}, control_hz=20, cube_position=cube,
                          box_position=np.array([0., .18, .9]), success=lambda: True,
                          reset=lambda: pytest.fail("Expert must never reset after takeover"))
    recorder = SimpleNamespace(env=env, obs={"observation.state": np.array([-.01, -.12, .97, 0., 0., 0., 0., 0.])})
    def move(xyz, grip, phase, **kwargs):
        events.append((phase, np.asarray(xyz).copy(), grip))
        if phase == "lift":
            # Test stand-in for physics, not an assignment in the implementation.
            env.cube_position = np.array([.02, -.19, 1.03])
    recorder.move = move
    recorder.tick = lambda xyz, grip, phase: events.append((phase, np.asarray(xyz), grip))
    recorder.close_until_grasp = lambda xyz, **kwargs: events.append(("grasp", np.asarray(xyz), 1.))
    recorder.open_until_released = lambda xyz: events.append(("release", np.asarray(xyz), -1.))
    track = {"samples": [{"t": i, "xyz": [i * .1, .02 if i in (1, 2) else 0., .05]} for i in range(4)],
             "grasp_time_s": 0., "release_time_s": 3.}
    pilot.expert_recovery(recorder, track)
    assert events[0][0] == "recovery_open_retract" and events[0][2] == -1.
    np.testing.assert_allclose(events[1][1][:2], [.02, -.19])
    assert any(event[0] == "carry_personal_video" for event in events)
    assert any(event[0] == "release" for event in events)


def collection_plan():
    return pilot.read_json(Path(__file__).resolve().parents[1] / "configs/improvement_20261011/recovery_selection.json")


def test_collection_plan_is_exact_distinct_and_balanced():
    plan = collection_plan()
    checked = pilot.validate_collection_plan(plan, pilot.RECOVERY_CLIPS)
    assert len({(job["seed"], job["prefix_steps"]) for job in checked["jobs"]}) == 20
    assert {clip: sum(job["clip"] == clip for job in checked["jobs"]) for clip in pilot.RECOVERY_CLIPS} == dict.fromkeys(pilot.RECOVERY_CLIPS, 4)
    for field, value in (("seed", 4100), ("prefix_steps", 79), ("clip", "IMG_2097")):
        changed = deepcopy(plan)
        changed["jobs"][0][field] = value
        with pytest.raises(ValueError, match="fixed 20"):
            pilot.validate_collection_plan(changed, pilot.RECOVERY_CLIPS)
    changed = deepcopy(plan)
    changed["jobs"].append(changed["jobs"][0])
    with pytest.raises(ValueError, match="fixed 20"):
        pilot.validate_collection_plan(changed, pilot.RECOVERY_CLIPS)


def collection_fixture(tmp_path, monkeypatch, *, failed_pair=None):
    """Mock only physics and input discovery; real arrays, hashes and copies."""
    monkeypatch.setattr(pilot, "ROOT", tmp_path)
    raw = tmp_path / "personal.MOV"
    raw.write_bytes(b"synthetic test bytes, never study data")
    source = {"kind": "personal_video", "video": "personal.MOV", "sha256": pilot.digest(raw)}
    settings = {"control_hz": 20, "orientation_control": "fixed_downward"}
    config = {"_root": str(tmp_path), "task": "Place cube", "simulation": settings}
    jobs = [{**job, "source_phase": "development", "source_group": "known_wide", "trace_sha256": "test-trace"}
            for job in collection_plan()["jobs"]]
    originals = []
    def save_episode(path, report):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, state=np.zeros((2, 8), np.float32), action=np.zeros((2, 7), np.float32),
                            image=np.zeros((2, 256, 256, 3), np.uint8), image2=np.zeros((2, 256, 256, 3), np.uint8))
        pilot.write_json(path.with_suffix(".json"), report)
    for index in range(20):
        name = f"original_{index:02d}.npz"
        path = tmp_path / "data/robot/main" / name
        save_episode(path, {"seed": index, "source": source, "success": True, "task": config["task"], "frames": 2})
        originals.append({"name": name, "sha256": pilot.digest(path), "metadata_sha256": pilot.digest(path.with_suffix('.json'))})
    manifest = {"kind": "fixed_recovery_collection", "jobs": jobs, "original": originals,
                "takeover_tolerance": 1e-6, "test_config_hash": "frozen"}
    monkeypatch.setattr(pilot, "_prepare_collection", lambda *args: (deepcopy(manifest), deepcopy(config)))
    calls = []
    def collect(config, job, destination, deadline, tolerance):
        calls.append((job["seed"], job["prefix_steps"]))
        success = calls[-1] != failed_pair
        report = {"seed": job["seed"], "source": source, "success": success, "frames": 2 if success else 0,
                  "task": config["task"], "failure": None if success else "test physical grasp failure",
                  "prefix_provenance": {key: value for key, value in job.items() if key != "clip"}}
        path = destination / (pilot._job_stem(job) + ".npz")
        path.with_suffix(".mp4").write_bytes(b"synthetic test video artifact")
        if success:
            save_episode(path, report)
        else:
            pilot.write_json(path.with_suffix(".json"), report)
        return report
    monkeypatch.setattr(pilot, "collect_job", collect)
    def export(config, demos_dir, output_dir):
        provenance = []
        for path in sorted(demos_dir.glob("*.npz")):
            arrays, metadata = pilot._validate_episode(path, tmp_path)
            provenance.append({"episode_index": len(provenance), "frames": len(arrays["action"]), **metadata})
        report = {"episodes": len(provenance), "provenance": provenance}
        pilot.write_json(output_dir / "meta/phone_to_panda_provenance.json", report)
        (output_dir / "payload.test").write_bytes(b"test export payload")
        return report
    monkeypatch.setattr(pilot, "export_dataset", export)
    def run(**kwargs):
        return pilot.build_recovery_collection(config, tmp_path / "selection.json", tmp_path / "config.yaml",
            tmp_path / "study", tmp_path / "robot", tmp_path / "dataset", deadline=time.monotonic() + 100, **kwargs)
    return run, calls, manifest


def test_collection_copies_originals_unchanged_exports_seed_provenance_and_resumes(tmp_path, monkeypatch):
    run, calls, manifest = collection_fixture(tmp_path, monkeypatch)
    result = run(skip_export=True)
    assert result["complete"] and result["successes"] == 20 and len(calls) == 20
    assert len(list((tmp_path / "robot").glob("*.npz"))) == 40
    assert not (tmp_path / "dataset").exists()
    for original in manifest["original"]:
        assert pilot.digest(tmp_path / "robot" / original["name"]) == original["sha256"]
        assert pilot.digest((tmp_path / "robot" / original["name"]).with_suffix(".json")) == original["metadata_sha256"]
    exported = run(export_only=True)
    assert exported["exported_episodes"] == 40 and len(calls) == 20
    provenance = pilot.read_json(tmp_path / "dataset/meta/phone_to_panda_provenance.json")["provenance"]
    assert {ep["seed"] for ep in provenance if "prefix_provenance" in ep} == set(pilot.RECOVERY_SEEDS)
    assert all(ep["seed"] == ep["prefix_provenance"]["seed"] for ep in provenance if "prefix_provenance" in ep)
    run(export_only=True)
    assert len(calls) == 20
    (tmp_path / "dataset/payload.test").write_bytes(b"modified export")
    with pytest.raises(ValueError, match="Existing export changed"):
        run(export_only=True)


def test_failed_recovery_is_retained_and_never_retried_or_exported(tmp_path, monkeypatch):
    run, calls, _ = collection_fixture(tmp_path, monkeypatch, failed_pair=(3010, 80))
    with pytest.raises(ValueError, match="fixed recovery attempt failed"):
        run()
    assert calls == [(3010, 60), (3010, 80)]
    assert not list((tmp_path / "robot").glob("*.npz")) and not (tmp_path / "dataset").exists()
    failed = tmp_path / "robot/build/candidates/IMG_2093_seed_3010_prefix_80.json"
    assert pilot.read_json(failed)["success"] is False and failed.with_suffix(".mp4").exists()
    with pytest.raises(ValueError, match="fixed recovery attempt failed"):
        run()
    assert len(calls) == 2


def test_verified_partial_collection_resumes_only_unattempted_jobs(tmp_path, monkeypatch):
    run, calls, _ = collection_fixture(tmp_path, monkeypatch)
    def stop_after_two(deadline):
        if len(calls) == 2:
            raise TimeoutError("test cutoff before third attempt")
    monkeypatch.setattr(pilot, "check_deadline", stop_after_two)
    with pytest.raises(TimeoutError):
        run(skip_export=True)
    assert len(calls) == 2
    monkeypatch.setattr(pilot, "check_deadline", lambda deadline: None)
    assert run(skip_export=True)["successes"] == 20
    assert len(calls) == len(set(calls)) == 20


def test_incomplete_attempt_and_changed_inputs_are_not_retried(tmp_path, monkeypatch):
    run, calls, manifest = collection_fixture(tmp_path, monkeypatch)
    original_collect = pilot.collect_job
    def interrupted(*args):
        raise OSError("test interruption before report")
    monkeypatch.setattr(pilot, "collect_job", interrupted)
    with pytest.raises(OSError):
        run(skip_export=True)
    monkeypatch.setattr(pilot, "collect_job", original_collect)
    with pytest.raises(ValueError, match="Incomplete recovery attempt"):
        run(skip_export=True)
    assert calls == []
    manifest["test_config_hash"] = "changed"
    with pytest.raises(ValueError, match="Frozen recovery inputs changed"):
        run(skip_export=True)


def test_export_only_cannot_collect_and_receipts_detect_changed_artifacts(tmp_path, monkeypatch):
    run, calls, _ = collection_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="No verified recovery plan"):
        run(export_only=True)
    assert calls == []
    run(skip_export=True)
    victim = next((tmp_path / "robot/build/candidates").glob("*.mp4"))
    victim.write_bytes(b"changed video")
    with pytest.raises(ValueError, match="Recovery artifact changed"):
        run(export_only=True)
    assert len(calls) == 20
