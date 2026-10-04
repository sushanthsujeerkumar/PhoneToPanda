"""Contract tests plus an opt-in real contact-physics integration check."""
import os

import numpy as np
import pytest

from phone_to_panda.simulation import PandaBoxEnv, _Recorder, record_demo, retarget_carried_path


def measured_track(curvature=.03):
    return {
        "samples": [
            {"t": 1., "xyz": [0., 0., .02]},
            {"t": 2., "xyz": [.1, curvature, .08]},
            {"t": 3., "xyz": [.2, curvature, .10]},
            {"t": 4., "xyz": [.3, 0., .02]},
        ],
        "grasp_time_s": 1., "release_time_s": 4.,
    }


def test_measured_curvature_and_timing_change_robot_carry():
    start, end = [0, -.17, 1.03], [0, .18, 1.03]
    times, path = retarget_carried_path(measured_track(.03), start, end, {})
    _, straight = retarget_carried_path(measured_track(0), start, end, {})
    assert times.tolist() == [0, 1, 2, 3]
    np.testing.assert_allclose(path[[0, -1], :2], np.array([start, end])[:, :2], atol=1e-12)
    assert abs(path[1, 0] - straight[1, 0]) > .025
    assert path[2, 2] > path[0, 2] + .08
    faster, _ = retarget_carried_path(measured_track(), start, end, {"video_speed": 2})
    np.testing.assert_allclose(faster, times / 2)


def test_rejects_unusable_track():
    track = measured_track()
    track["samples"][2]["t"] = track["samples"][1]["t"]
    with pytest.raises(ValueError, match="strictly increasing"):
        retarget_carried_path(track, [0, -.17, 1.03], [0, .18, 1.03], {})


def test_synthetic_episode_cannot_enter_training_data(tmp_path):
    with pytest.raises(ValueError, match="Synthetic setup"):
        record_demo({"_root": str(tmp_path)}, tmp_path / "data" / "robot" / "demo.npz")


def contact_recorder(force_samples):
    """A small contact sequence tests transition conditions without MuJoCo."""
    from types import SimpleNamespace
    recorder = _Recorder.__new__(_Recorder)
    recorder.actions, recorder.transitions = [], []
    recorder.env = SimpleNamespace(grasp_contact_forces=lambda: np.asarray(force_samples[len(recorder.actions) - 1]))
    recorder.tick = lambda xyz, grip, phase: recorder.actions.append((grip, phase))
    return recorder


def test_contact_grasp_requires_load_on_both_fingers_and_has_no_extra_wait():
    recorder = contact_recorder([[0, 0], [2, 0], [2, .8], [2, 1.2]])
    recorder.close_until_grasp([0, 0, 1], max_steps=4)
    assert len(recorder.actions) == 4
    assert recorder.transitions[0]["finger_normal_forces_n"] == [2, 1.2]
    assert recorder.transitions[0]["frame"] == 4


def test_one_sided_contact_times_out_instead_of_exporting_a_lift():
    recorder = contact_recorder([[2, 0]] * 3)
    with pytest.raises(RuntimeError, match="No firm bilateral grasp"):
        recorder.close_until_grasp([0, 0, 1], max_steps=3)
    assert len(recorder.actions) == 3
    assert recorder.transitions == []


def test_contact_release_requires_open_fingers_and_no_object_contact():
    from types import SimpleNamespace
    recorder = _Recorder.__new__(_Recorder)
    recorder.actions, recorder.transitions, recorder.obs = [], [], {}
    samples = [(.02, False), (.04, True), (.035, False)]
    recorder.env = SimpleNamespace(cube_touches_gripper=lambda: samples[len(recorder.actions) - 1][1])
    def tick(xyz, grip, phase):
        aperture, _ = samples[len(recorder.actions)]
        recorder.obs["observation.state"] = np.r_[np.zeros(6), aperture, -aperture]
        recorder.actions.append((grip, phase))
    recorder.tick = tick
    recorder.open_until_released([0, 0, 1], max_steps=3)
    assert len(recorder.actions) == 3
    assert recorder.transitions[0]["finger_aperture_m"] == pytest.approx(.070)


@pytest.mark.parametrize("mode", ["policy", "fixed_downward"])
def test_orientation_hold_changes_only_rotation_and_does_not_mutate_input(monkeypatch, mode):
    from types import SimpleNamespace
    import phone_to_panda.simulation as simulation
    env = PandaBoxEnv.__new__(PandaBoxEnv)
    env.orientation_control, env.steps, env._max_cube_height = mode, 0, .92
    env._raw_obs = {"robot0_eef_quat": np.array([1., 0, 0, 0])}
    env._convert, env.success = lambda raw: raw, lambda: False
    captured = []
    def raw_step(action):
        captured.append(action.copy())
        return {}, 0., False, {}
    env.raw_env = SimpleNamespace(step=raw_step, obj_body_id={"cube_1": 0},
                                  sim=SimpleNamespace(data=SimpleNamespace(body_xpos=np.array([[0, 0, .92]]))))
    correction = np.array([.1, -.2, .3])
    monkeypatch.setattr(simulation, "_downward_rotation_action", lambda quaternion: correction)
    action = np.array([.2, -.3, .4, .5, -.6, .7, -1.], np.float32)
    original = action.copy()
    _, _, _, info = env.step(action)
    np.testing.assert_array_equal(action, original)
    np.testing.assert_allclose(captured[0][[0, 1, 2, 6]], original[[0, 1, 2, 6]])
    np.testing.assert_allclose(captured[0][3:6], correction if mode == "fixed_downward" else original[3:6])
    assert info["orientation_control"] == mode


@pytest.mark.skipif(os.environ.get("PHONE_PHYSICS_TEST") != "1", reason="Run PHONE_PHYSICS_TEST=1 for the EGL/MuJoCo integration test")
@pytest.mark.parametrize("transition,orientation", [("timed", "policy"), ("contact", "fixed_downward")])
def test_real_contact_physics_pick_and_place(tmp_path, transition, orientation):
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    result = record_demo({"_root": str(root), "simulation": {
        "image_size": 64, "randomize_start_m": 0, "demo_transition_mode": transition,
        "orientation_control": orientation, "save_replay_video": False,
    }}, tmp_path / "setup.npz", seed=0)
    assert result["success"], result
    assert result["max_cube_height_m"] > .97
    with np.load(tmp_path / "setup.npz") as episode:
        assert episode["state"].shape[1:] == (8,)
        assert episode["action"].shape[1:] == (7,)
        assert episode["image"].shape[1:] == (64, 64, 3)
        assert len(episode["state"]) == len(episode["action"])
        if orientation == "fixed_downward":
            np.testing.assert_array_equal(episode["action"][:, 3:6], 0)
            assert min(result["phase_transition_evidence"][0]["finger_normal_forces_n"]) >= 1.
    assert result["video"] is None
    assert not (tmp_path / "setup.mp4").exists()
