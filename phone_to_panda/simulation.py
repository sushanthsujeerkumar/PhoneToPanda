"""Custom LIBERO/Panda cube-to-box task and contact-based demonstration replay.

The setup controller is an oracle, not a learned policy. Personal-video replay uses
measured object motion for the carried path; approach, grasp, release, and minimum
clearance are scripted robot-specific adaptations. No object is attached to the
hand, and object poses are only assigned during reset.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

TASK = "Pick up the red cube and place it inside the blue box."
TABLE_Z = 0.90
CUBE_HALF = 0.020
STATE_NAMES = ["eef_x", "eef_y", "eef_z", "eef_axisangle_x", "eef_axisangle_y", "eef_axisangle_z", "gripper_left", "gripper_right"]
ACTION_NAMES = ["delta_x", "delta_y", "delta_z", "delta_axisangle_x", "delta_axisangle_y", "delta_axisangle_z", "gripper"]


def _downward_rotation_action(quaternion) -> np.ndarray:
    """Normalized OSC correction, with no translation or gripper intervention."""
    from robosuite.utils.transform_utils import quat2mat
    from robosuite.utils.control_utils import orientation_error
    rotation = orientation_error(np.diag([1., -1., -1.]), quat2mat(quaternion)) / .5
    return np.clip(rotation * .6, -.7, .7).astype(np.float32)


def _root(config: dict) -> Path:
    return Path(config.get("_root", Path(__file__).resolve().parents[1])).resolve()


def _prepare_libero(root: Path):
    """Localize LIBERO's interactive config and use the bundled scene assets."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
    cfg_dir = root / ".cache" / "libero"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    # Set before import: upstream otherwise asks an interactive first-run question.
    os.environ["LIBERO_CONFIG_PATH"] = str(cfg_dir)
    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("LIBERO is not installed. Run bash setup.sh first.")
    package_root = Path(next(iter(spec.submodule_search_locations))) / "libero"
    assets = root / "assets" / "libero"
    if not (assets / "scenes" / "libero_tabletop_base_style.xml").exists():
        raise FileNotFoundError(f"Bundled LIBERO scene missing under {assets}")
    import yaml
    (cfg_dir / "config.yaml").write_text(yaml.safe_dump({
        "benchmark_root": str(package_root),
        "bddl_files": str(package_root / "bddl_files"),
        "init_states": str(package_root / "init_files"),
        "datasets": str(root / "data"), "assets": str(assets),
    }))
    import libero.libero as libero
    # hf-libero 0.1.4's asset resolver ignores config.yaml and otherwise downloads
    # all benchmark assets. Its cache is the smallest explicit integration point.
    libero._assets_path_cache = str(assets)


def _make_raw_env(root: Path, settings: dict):
    _prepare_libero(root)
    import robosuite
    from robosuite.models.objects import MujocoXMLObject
    from libero.libero.envs.base_object import OBJECTS_DICT
    from libero.libero.envs.bddl_base_domain import TASK_MAPPING
    # Upstream registration decorator stores the class but returns None.
    Libero_Tabletop_Manipulation = TASK_MAPPING["libero_tabletop_manipulation"]

    class TaskObject(MujocoXMLObject):
        def __init__(self, name, file, category, joints="default"):
            super().__init__(str(root / "assets" / file), name=name, joints=joints,
                             obj_type="all", duplicate_collision_geoms=True)
            self.category_name = category
            self.rotation = (0, 0)
            self.rotation_axis = "z"
            self.object_properties = {"vis_site_names": {}}

    class Cube(TaskObject):
        def __init__(self, name, joints="default"):
            super().__init__(name, "cube.xml", "phone_cube", joints)

    class OpenBox(TaskObject):
        def __init__(self, name, joints=None):
            super().__init__(name, "open_box.xml", "phone_open_box", joints)

    OBJECTS_DICT.update(phone_cube=Cube, phone_open_box=OpenBox)

    class PickPlaceDomain(Libero_Tabletop_Manipulation):
        def _assert_problem_name(self):
            # Reuse LIBERO's tabletop BDDL parser and region samplers.
            assert self.parsed_problem["problem_name"] == "libero_tabletop_manipulation"

        def _check_success(self):
            # Physical placement criterion; wrapper additionally requires a lift.
            if not hasattr(self, "obj_body_id"):
                return False
            cube_id, box_id = self.obj_body_id["cube_1"], self.obj_body_id["box_1"]
            delta = self.sim.data.body_xpos[cube_id] - self.sim.data.body_xpos[box_id]
            cube_speed = np.linalg.norm(self.sim.data.get_body_xvelp(self.objects_dict["cube_1"].root_body))
            in_box = abs(delta[0]) < .050 and abs(delta[1]) < .050 and .026 < delta[2] < .052
            gripper = self.robots[0].gripper
            released = not self.check_contact(gripper, self.objects_dict["cube_1"])
            return bool(in_box and released and cube_speed < .05)

    controller = robosuite.load_controller_config(default_controller="OSC_POSE")
    return PickPlaceDomain(
        str(root / "assets" / "pick_place.bddl"), robots=["Panda"],
        controller_configs=controller, initialization_noise=None,
        use_camera_obs=True, use_object_obs=True, has_renderer=False,
        has_offscreen_renderer=True, camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=int(settings.get("image_size", 256)),
        camera_widths=int(settings.get("image_size", 256)),
        control_freq=int(settings.get("control_hz", 20)), horizon=int(settings.get("horizon", 600)),
        ignore_done=False, hard_reset=False,
    )


class PandaBoxEnv:
    """Small adapter with the same state/action/camera contract used by SmolVLA."""
    def __init__(self, config: dict, seed: int = 0):
        self.config, self.seed = config, int(seed)
        self.settings = config.get("simulation", {})
        self.orientation_control = self.settings.get("orientation_control", "policy")
        if self.orientation_control not in {"policy", "fixed_downward"}:
            raise ValueError("simulation.orientation_control must be policy or fixed_downward.")
        self.raw_env = _make_raw_env(_root(config), self.settings)
        self.control_hz = int(self.settings.get("control_hz", 20))
        self.horizon = int(self.settings.get("horizon", 600))
        self.rng = np.random.default_rng(seed)
        self._raw_obs = None
        self._max_cube_height = TABLE_Z
        self.steps = 0

    def _convert(self, raw: dict) -> dict:
        from robosuite.utils.transform_utils import quat2axisangle
        state = np.concatenate((raw["robot0_eef_pos"], quat2axisangle(raw["robot0_eef_quat"]), raw["robot0_gripper_qpos"]))
        return {
            "observation.state": state.astype(np.float32),
            "observation.images.image": np.ascontiguousarray(raw["agentview_image"][::-1], dtype=np.uint8),
            "observation.images.image2": np.ascontiguousarray(raw["robot0_eye_in_hand_image"][::-1], dtype=np.uint8),
        }

    def reset(self) -> dict:
        self.raw_env.seed(self.seed)
        self.raw_env.reset()
        env = self.raw_env
        xy = np.asarray(self.settings.get("cube_start_xy", [0., -.17]), dtype=float)
        jitter = float(self.settings.get("randomize_start_m", .035))
        xy += self.rng.uniform(-jitter, jitter, size=2)
        box_xy = np.asarray(self.settings.get("box_xy", [0., .18]), dtype=float)
        # Only initial reset assigns poses. Every subsequent movement is physics.
        env.sim.data.set_joint_qpos(env.objects_dict["cube_1"].joints[0], np.r_[xy, TABLE_Z + CUBE_HALF + .001, 1., 0., 0., 0.])
        env.sim.model.body_pos[env.obj_body_id["box_1"]] = np.r_[box_xy, TABLE_Z]
        env.sim.forward()
        for _ in range(8):
            raw, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1.]))
        # Exclude settling from the episode horizon and exported training sequence.
        env.timestep = 0
        env.done = False
        self.steps = 0
        self._max_cube_height = float(self.cube_position[2])
        self._raw_obs = raw
        return self._convert(raw)

    @property
    def cube_position(self) -> np.ndarray:
        return self.raw_env.sim.data.body_xpos[self.raw_env.obj_body_id["cube_1"]].copy()

    @property
    def box_position(self) -> np.ndarray:
        return self.raw_env.sim.data.body_xpos[self.raw_env.obj_body_id["box_1"]].copy()

    def grasp_contact_forces(self) -> np.ndarray:
        """Measured normal force on each finger pad against the cube, in newtons."""
        import mujoco
        env, forces = self.raw_env, np.zeros(2)
        gripper = env.robots[0].gripper
        pads = [{env.sim.model.geom_name2id(name) for name in gripper.important_geoms[key]}
                for key in ("left_fingerpad", "right_fingerpad")]
        cube_geoms = {env.sim.model.geom_name2id(name) for name in env.objects_dict["cube_1"].contact_geoms}
        for index in range(env.sim.data.ncon):
            contact = env.sim.data.contact[index]
            for side, geoms in enumerate(pads):
                if ((contact.geom1 in geoms and contact.geom2 in cube_geoms)
                        or (contact.geom2 in geoms and contact.geom1 in cube_geoms)):
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(env.sim.model._model, env.sim.data._data, index, wrench)
                    forces[side] += max(0., float(wrench[0]))
        return forces

    def cube_touches_gripper(self) -> bool:
        return bool(self.raw_env.check_contact(self.raw_env.robots[0].gripper,
                                               self.raw_env.objects_dict["cube_1"]))

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("Panda action must be seven finite numbers.")
        applied_action = np.clip(action, -1., 1.)
        if self.orientation_control == "fixed_downward":
            applied_action[3:6] = _downward_rotation_action(self._raw_obs["robot0_eef_quat"])
        raw, reward, done, info = self.raw_env.step(applied_action)
        self.steps += 1
        self._raw_obs = raw
        self._max_cube_height = max(self._max_cube_height, float(self.cube_position[2]))
        info = dict(info, success=self.success(), cube_xyz=self.cube_position.tolist(),
                    orientation_control=self.orientation_control, applied_action=applied_action.tolist())
        return self._convert(raw), float(reward), bool(done), info

    def success(self) -> bool:
        return bool(self._max_cube_height > TABLE_Z + .070 and self.raw_env._check_success())

    def close(self):
        self.raw_env.close()


def make_env(config: dict, seed: int = 0) -> PandaBoxEnv:
    return PandaBoxEnv(config, seed=seed)


def retarget_carried_path(trajectory: dict, start_xyz, end_xyz, settings: dict) -> tuple[np.ndarray, np.ndarray]:
    """Map measured object motion to the simulator, retaining timing and curvature.

    Align start->end translation in the table plane, scale to robot displacement,
    and preserve measured deviations from the endpoint line. A robot clearance
    floor and workspace ceiling deliberately modify unsafe vertical portions.
    """
    samples = trajectory.get("samples", [])
    if len(samples) < 4:
        raise ValueError("At least four measured trajectory samples are required.")
    times = np.asarray([s["t"] for s in samples], dtype=float)
    xyz = np.asarray([s["xyz"] for s in samples], dtype=float)
    if xyz.shape != (len(times), 3) or not np.isfinite(xyz).all() or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Trajectory needs finite XYZ values and strictly increasing times.")
    grasp, release = float(trajectory["grasp_time_s"]), float(trajectory["release_time_s"])
    if not times[0] <= grasp < release <= times[-1]:
        raise ValueError("Grasp and release must lie inside measured sample times.")
    inside = (times > grasp) & (times < release)
    carry_t = np.r_[grasp, times[inside], release]
    carry = np.stack([np.interp(carry_t, times, xyz[:, axis]) for axis in range(3)], axis=1)
    human_d = carry[-1, :2] - carry[0, :2]
    if np.linalg.norm(human_d) < .025:
        raise ValueError("Measured pick/place separation is under 2.5 cm; check markers and annotations.")
    start, end = np.asarray(start_xyz, float), np.asarray(end_xyz, float)
    robot_d = end[:2] - start[:2]
    angle = np.arctan2(robot_d[1], robot_d[0]) - np.arctan2(human_d[1], human_d[0])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    scale = np.linalg.norm(robot_d) / np.linalg.norm(human_d)
    path = np.empty_like(carry)
    path[:, :2] = start[:2] + (carry[:, :2] - carry[0, :2]) @ rotation.T * scale
    progress = (carry_t - grasp) / (release - grasp)
    # Remove endpoint height offset, retaining the measured lift profile.
    base_z = carry[0, 2] * (1 - progress) + carry[-1, 2] * progress
    measured_lift = (carry[:, 2] - base_z) * scale
    clearance = float(settings.get("min_carry_height_m", .13))
    ceiling = float(settings.get("max_carry_height_m", .30))
    path[:, 2] = np.clip(start[2] + measured_lift, TABLE_Z + clearance, TABLE_Z + ceiling)
    # Reject grossly out-of-workspace tracks instead of silently distorting them.
    if np.any(abs(path[:, 0]) > .30) or np.any(abs(path[:, 1]) > .40):
        raise ValueError("Retargeted path leaves the supported workspace; inspect the measured track.")
    speed = float(settings.get("video_speed", 1.0))
    if speed <= 0:
        raise ValueError("video_speed must be positive.")
    return (carry_t - grasp) / speed, path


class _Recorder:
    def __init__(self, env: PandaBoxEnv, initial: dict):
        self.env, self.obs = env, initial
        self.states, self.images, self.images2, self.actions, self.phases = [], [], [], [], []
        self.max_cube_height = float(env.cube_position[2])
        self.transitions = []

    def action_to(self, xyz, gripper):
        delta = (np.asarray(xyz) - self.obs["observation.state"][:3]) / .05
        rotation = (np.zeros(3) if self.env.orientation_control == "fixed_downward"
                    else _downward_rotation_action(self.env._raw_obs["robot0_eef_quat"]))
        return np.r_[np.clip(delta * .65, -.65, .65), rotation, gripper].astype(np.float32)

    def tick(self, xyz, gripper, phase):
        if self.env.steps >= self.env.horizon:
            raise RuntimeError("Replay exceeded simulation.horizon; increase it or speed up the recorded carry.")
        action = self.action_to(xyz, gripper)
        self.states.append(self.obs["observation.state"])
        self.images.append(self.obs["observation.images.image"])
        self.images2.append(self.obs["observation.images.image2"])
        self.actions.append(action)
        self.phases.append(phase)
        self.obs, _, _, _ = self.env.step(action)
        self.max_cube_height = max(self.max_cube_height, float(self.env.cube_position[2]))

    def move(self, xyz, gripper, phase, min_steps=8, max_steps=80, tolerance=.009):
        for i in range(max_steps):
            self.tick(xyz, gripper, phase)
            if i + 1 >= min_steps and np.linalg.norm(self.obs["observation.state"][:3] - xyz) < tolerance:
                return
        raise RuntimeError(f"Robot failed to reach {phase}; check workspace limits.")

    def close_until_grasp(self, xyz, max_steps=40, min_force_n=1.):
        """Advance as soon as both fingers exert force; the count is only a timeout."""
        if max_steps < 1 or not np.isfinite(min_force_n) or min_force_n <= 0:
            raise ValueError("Contact grasp requires a positive timeout and normal force threshold.")
        for _ in range(max_steps):
            self.tick(xyz, 1., "grasp")
            forces = self.env.grasp_contact_forces()
            if np.all(forces >= min_force_n):
                self.transitions.append({"phase": "grasp_to_lift", "frame": len(self.actions),
                                         "finger_normal_forces_n": forces.tolist()})
                return
        raise RuntimeError("No firm bilateral grasp before the contact timeout; episode rejected.")

    def open_until_released(self, xyz, max_steps=40, aperture_m=.060):
        if max_steps < 1 or not 0 < aperture_m < .080:
            raise ValueError("Contact release requires a positive timeout and aperture below 80 mm.")
        for _ in range(max_steps):
            self.tick(xyz, -1., "release")
            aperture = float(np.abs(self.obs["observation.state"][6:8]).sum())
            if not self.env.cube_touches_gripper() and aperture >= aperture_m:
                self.transitions.append({"phase": "release_to_retreat", "frame": len(self.actions),
                                         "finger_aperture_m": aperture})
                return
        raise RuntimeError("Cube did not release before the contact timeout; episode rejected.")


def record_demo(config: dict, output_path, trajectory_path=None, seed: int = 0) -> dict[str, Any]:
    """Export one physics replay; absence of a trajectory means synthetic setup-only."""
    output = Path(output_path).expanduser().resolve()
    if output.suffix != ".npz":
        output = output.with_suffix(".npz")
    trajectory = None
    source = {"kind": "synthetic_setup_controller", "video": None, "sha256": None}
    if trajectory_path is not None:
        trajectory_path = Path(trajectory_path).expanduser().resolve()
        trajectory = json.loads(trajectory_path.read_text())
        if trajectory.get("schema_version") != 1 or trajectory.get("coordinate_frame") != "table_marker":
            raise ValueError("Unsupported trajectory schema/frame.")
        source = dict(trajectory.get("source", {}))
        if source.get("kind") != "personal_video" or not source.get("video") or len(source.get("sha256", "")) != 64:
            raise ValueError("Replay requires a personal-video source with a SHA256 provenance hash.")
        video = Path(source["video"]).expanduser()
        if not video.is_absolute():
            video = _root(config) / video
        if not video.exists():
            raise FileNotFoundError(f"Source video is missing: {video}")
        with video.open("rb") as stream:
            actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual_hash != source["sha256"]:
            raise ValueError("Video hash does not match the measured trajectory provenance.")
    elif output.is_relative_to(_root(config) / "data"):
        raise ValueError("Synthetic setup demos belong in runs/setup_check or work, not data/.")
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"Refusing to overwrite an existing episode: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    transition_mode = config.get("simulation", {}).get("demo_transition_mode", "timed")
    if transition_mode not in {"timed", "contact"}:
        raise ValueError("simulation.demo_transition_mode must be timed or contact.")
    contact_mode = transition_mode == "contact"
    env = make_env(config, seed=seed)
    recorder = None
    failure = None
    try:
        recorder = _Recorder(env, env.reset())
        cube = env.cube_position
        box = env.box_position
        grasp = cube + np.array([0., 0., .003])
        high = TABLE_Z + float(env.settings.get("min_carry_height_m", .13))
        recorder.move(np.r_[cube[:2], high + .04], -1., "approach", min_steps=1 if contact_mode else 8)
        recorder.move(grasp, -1., "descend", min_steps=1 if contact_mode else 8)
        if contact_mode:
            recorder.close_until_grasp(grasp, min_force_n=float(env.settings.get("grasp_min_normal_force_n", 1.)))
        else:
            for _ in range(22):
                recorder.tick(grasp, 1., "grasp")
        recorder.move(np.r_[cube[:2], high], 1., "lift", min_steps=1 if contact_mode else 20)
        if env.cube_position[2] < TABLE_Z + .060:
            raise RuntimeError("The physical grasp did not lift the cube; episode is a failure.")
        end = np.r_[box[:2], high]
        if trajectory is None:
            times = np.linspace(0, 3., 61)
            progress = times / times[-1]
            path = (1 - progress[:, None]) * np.r_[cube[:2], high] + progress[:, None] * end
            path[:, 2] += .025 * np.sin(progress * np.pi)
        else:
            times, path = retarget_carried_path(trajectory, np.r_[cube[:2], high], end, env.settings)
        for time in np.arange(0, times[-1] + 1 / env.control_hz, 1 / env.control_hz):
            point = np.array([np.interp(time, times, path[:, k]) for k in range(3)])
            recorder.tick(point, 1., "carry_personal_video" if trajectory is not None else "carry_synthetic")
        recorder.move(end, 1., "over_box", min_steps=1 if contact_mode else 8)
        release = np.r_[box[:2], TABLE_Z + .052]
        recorder.move(release, 1., "lower_into_box", min_steps=1 if contact_mode else 12)
        if contact_mode:
            recorder.open_until_released(release)
        else:
            for _ in range(20):
                recorder.tick(release, -1., "release")
        recorder.move(np.r_[box[:2], high + .06], -1., "retreat", min_steps=1 if contact_mode else 20)
        for _ in range(20):
            if contact_mode and env.success():
                break
            recorder.tick(np.r_[box[:2], high + .06], -1., "settle")
    except (RuntimeError, ValueError) as exc:
        failure = str(exc)
    finally:
        success = env.success() if recorder else False
        final_cube = env.cube_position.tolist() if recorder else None
        if recorder is not None and recorder.actions:
            np.savez_compressed(output, state=np.asarray(recorder.states, np.float32),
                                image=np.asarray(recorder.images, np.uint8), image2=np.asarray(recorder.images2, np.uint8),
                                action=np.asarray(recorder.actions, np.float32))
            report = {
                "schema_version": 1, "source": source, "success": bool(success and failure is None),
                "seed": int(seed), "task": config.get("task", TASK),
                "frames": len(recorder.actions), "fps": env.control_hz,
                "state_names": STATE_NAMES, "action_names": ACTION_NAMES,
                "action_contract": ("Hybrid OSC_POSE: xyz*0.05m; rotation action slots zero and ignored in favor of downward feedback; gripper -1 open,+1 close"
                                    if env.orientation_control == "fixed_downward" else
                                    "OSC_POSE normalized world deltas; xyz*0.05m, rotation*0.5rad; gripper -1 open,+1 close"),
                "max_cube_height_m": recorder.max_cube_height, "final_cube_xyz": final_cube,
                "trajectory": str(trajectory_path) if trajectory_path else None,
                "trajectory_quality": trajectory.get("quality", {}) if trajectory else None,
                "simulation_settings": env.settings,
                "failure": failure, "phases": recorder.phases,
                "demo_transition_mode": transition_mode, "phase_transition_evidence": recorder.transitions,
                "orientation_control": env.orientation_control,
                "controller": ("Hybrid: XYZ and gripper commands plus fixed downward orientation feedback; rotation action slots ignored"
                               if env.orientation_control == "fixed_downward" else "Seven-dimensional OSC_POSE commands"),
                "retargeting": "Measured carried object path, horizontal endpoint alignment/scaling, timed replay, safety height floor/ceiling; scripted approach/grasp/release" if trajectory else "Synthetic scripted setup check; not personal demonstration data or learned-policy performance",
                "physics": "MuJoCo contacts and friction only; no attachment or object-pose assignment after reset",
                "episode": str(output), "video": str(output.with_suffix(".mp4")),
            }
            if not env.settings.get("save_replay_video", True):
                report["video"] = None
            else:
                try:
                    import imageio.v2 as imageio
                    import cv2
                    frames = []
                    label = "PERSONAL VIDEO REPLAY (scripted grasp)" if trajectory else "SYNTHETIC SETUP CHECK (scripted controller)"
                    for a, b in zip(recorder.images, recorder.images2):
                        frame = np.concatenate((a, b), axis=1)
                        canvas = np.zeros((frame.shape[0] + 28, frame.shape[1], 3), np.uint8)
                        canvas[:frame.shape[0]] = frame
                        cv2.putText(canvas, label, (7, frame.shape[0] + 19), cv2.FONT_HERSHEY_SIMPLEX,
                                    min(.45, frame.shape[1] / 1150), (240, 240, 240), 1, cv2.LINE_AA)
                        frames.append(canvas)
                    imageio.mimsave(output.with_suffix(".mp4"), frames, fps=env.control_hz, codec="libx264", quality=7, macro_block_size=1)
                except Exception as exc:
                    report["video"] = None
                    report["video_error"] = str(exc)
            output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        else:
            report = {"success": False, "failure": failure or "No frames recorded", "source": source}
        env.close()
    return report
