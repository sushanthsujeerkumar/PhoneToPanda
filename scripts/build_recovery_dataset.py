"""Bounded recovery collection from saved development action prefixes.

This collects no new human recordings and performs no policy inference/training.
Only expert actions after takeover receive training labels. The saved policy
prefix is unlabelled context, replayed through physics without a second reset.
Without --collection-plan, the independent three-episode pilot is unchanged.
Full mode requires all 20 fixed jobs to succeed, then copies the original 20
episodes byte-for-byte and optionally exports the standard LeRobot dataset.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
from phone_to_panda.common import load_config, project_path
from phone_to_panda.learning import _validate_episode, export_dataset
from phone_to_panda.simulation import _Recorder, TABLE_Z, STATE_NAMES, ACTION_NAMES, make_env, retarget_carried_path
from scripts.build_dataset import prepare_tracks
from scripts.build_selected_dataset import _copy_identical, _original_manifest, _signature

RECOVERY_SEEDS = (3010, 3012, 3013, 3014, 3015)
RECOVERY_PREFIXES = (60, 80, 100, 120)
RECOVERY_CLIPS = ("IMG_2092", "IMG_2093", "IMG_2095", "IMG_2096", "IMG_2098")


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def action_digest(actions):
    return hashlib.sha256(json.dumps(actions, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Recovery pilot wall-time limit reached")


@contextmanager
def wall_timer(seconds):
    """Bound full mode too, including environment creation, copying and export."""
    def timeout_handler(signum, frame):
        raise TimeoutError("Recovery collection wall-time limit reached")
    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    started = time.monotonic()
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, max(.000001, previous_timer[0] - (time.monotonic() - started)), previous_timer[1])


def verify_state(obs, cube, expected, tolerance):
    state = np.asarray(obs["observation.state"], dtype=np.float64)
    wanted_state = np.asarray(expected["observation_state"], dtype=np.float64)
    wanted_cube = np.asarray(expected["cube_xyz"], dtype=np.float64)
    cube = np.asarray(cube, dtype=np.float64)
    if state.shape != (8,) or wanted_state.shape != (8,) or cube.shape != (3,) or wanted_cube.shape != (3,):
        raise ValueError("Reproduction state has an invalid shape")
    state_error, cube_error = float(np.max(np.abs(state - wanted_state))), float(np.max(np.abs(cube - wanted_cube)))
    if not np.isfinite([state_error, cube_error]).all() or max(state_error, cube_error) > tolerance:
        raise ValueError(f"Saved-prefix state reproduction failed: state {state_error:.3g}, cube {cube_error:.3g} m; tolerance {tolerance:g}")
    return {"max_state_error": state_error, "max_cube_xyz_error_m": cube_error, "absolute_tolerance": tolerance,
            "observation_state": state.tolist(), "cube_xyz": cube.tolist()}


def replay_prefix(env, trace, prefix_steps, *, deadline, tolerance=1e-6, write_frame=None):
    """One reset only; original raw policy actions retain their recorded order."""
    if not 0 < prefix_steps < len(trace) or [entry.get("step") for entry in trace[:prefix_steps + 1]] != list(range(prefix_steps + 1)):
        raise ValueError("A prefix requires contiguous saved actions and the next pre-action state")
    check_deadline(deadline)
    obs = env.reset()
    initial = verify_state(obs, env.cube_position, trace[0], tolerance)
    actions = []
    max_applied_error = 0.
    for entry in trace[:prefix_steps]:
        check_deadline(deadline)
        action = np.asarray(entry["policy_action"], dtype=np.float32)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("Saved prefix has an invalid policy action")
        if write_frame is not None:
            write_frame(obs, "UNLABELLED POLICY PREFIX", entry["step"])
        # Match run_policy's clipping and PandaBoxEnv's fixed-wrist application.
        obs, _, done, info = env.step(np.clip(action, -1., 1.))
        applied = np.asarray(info["applied_action"], dtype=np.float32)
        expected = np.asarray(entry["applied_action"], dtype=np.float32)
        error = float(np.max(np.abs(applied - expected)))
        max_applied_error = max(max_applied_error, error)
        if not np.isfinite(error) or error > tolerance:
            raise ValueError("Applied prefix actions differ from the preserved policy trace")
        actions.append(entry["policy_action"])
        if done or env.success():
            raise ValueError("Prefix already terminated or succeeded; it is not a failed-grasp recovery state")
    takeover = verify_state(obs, env.cube_position, trace[prefix_steps], tolerance)
    return obs, {"initial_state_check": initial, "takeover_state_check": takeover,
                 "prefix_steps": prefix_steps, "takeover_pre_action_step": prefix_steps,
                 "policy_actions_sha256": action_digest(actions), "max_applied_action_error": max_applied_error,
                 "unlabelled_policy_actions": actions}


def expert_recovery(recorder, trajectory):
    """Open/retract from the reached state, then use the existing replay recipe."""
    env = recorder.env
    high = TABLE_Z + float(env.settings.get("min_carry_height_m", .13))
    eef = recorder.obs["observation.state"][:3]
    recorder.move(np.r_[eef[:2], max(high + .04, float(eef[2]))], -1., "recovery_open_retract", min_steps=1)
    cube, box = env.cube_position, env.box_position
    if not TABLE_Z < cube[2] < TABLE_Z + .07:
        raise ValueError("Recovery requires the cube to remain on the table")
    grasp = cube + np.array([0., 0., .003])
    recorder.move(np.r_[cube[:2], high + .04], -1., "recovery_approach", min_steps=1)
    recorder.move(grasp, -1., "recovery_descend", min_steps=1)
    recorder.close_until_grasp(grasp, min_force_n=float(env.settings.get("grasp_min_normal_force_n", 1.)))
    recorder.move(np.r_[cube[:2], high], 1., "lift", min_steps=1)
    if env.cube_position[2] < TABLE_Z + .060:
        raise RuntimeError("Recovered grasp did not physically lift the cube")
    end = np.r_[box[:2], high]
    times, path = retarget_carried_path(trajectory, np.r_[cube[:2], high], end, env.settings)
    for instant in np.arange(0, times[-1] + 1 / env.control_hz, 1 / env.control_hz):
        recorder.tick(np.array([np.interp(instant, times, path[:, axis]) for axis in range(3)]), 1., "carry_personal_video")
    recorder.move(end, 1., "over_box", min_steps=1)
    recorder.move(np.r_[box[:2], TABLE_Z + .052], 1., "lower_into_box", min_steps=1)
    recorder.open_until_released(np.r_[box[:2], TABLE_Z + .052])
    recorder.move(np.r_[box[:2], high + .06], -1., "retreat", min_steps=1)
    for _ in range(20):
        if env.success():
            break
        recorder.tick(np.r_[box[:2], high + .06], -1., "settle")


class PilotRecorder(_Recorder):
    def __init__(self, env, initial, deadline, write_frame):
        super().__init__(env, initial)
        self.deadline, self.write_frame = deadline, write_frame

    def tick(self, xyz, gripper, phase):
        check_deadline(self.deadline)
        if self.write_frame:
            self.write_frame(self.obs, f"EXPERT LABEL: {phase}", self.env.steps)
        super().tick(xyz, gripper, phase)


def collect_job(config, job, destination, deadline, tolerance):
    import imageio.v2 as imageio
    import cv2
    stem = f"{job['clip']}_seed_{job['seed']}_prefix_{job['prefix_steps']}"
    video_path, report_path = destination / f"{stem}.mp4", destination / f"{stem}.json"
    output = destination / f"{stem}.npz"
    trajectory, trace = read_json(job["trajectory"]), read_json(job["trace"])
    source = trajectory["source"]
    video = ROOT / source["video"]
    if source.get("kind") != "personal_video" or not video.is_file() or digest(video) != source.get("sha256"):
        raise ValueError("Measured carry path has no valid personal-video provenance")
    if any(path.exists() for path in (video_path, report_path, output)):
        raise FileExistsError(f"Preserve existing recovery pilot artifacts: {stem}")
    if int(config["simulation"].get("control_hz", 20)) != 20 or config["simulation"].get("orientation_control") != "fixed_downward":
        raise ValueError("Pilot requires the source 20 Hz fixed-downward hybrid controller")
    report = {"schema_version": 1, "source": source, "seed": job["seed"], "task": config["task"],
              "success": False, "failure": None, "fps": 20, "frames": 0,
              "state_names": STATE_NAMES, "action_names": ACTION_NAMES,
              "simulation_settings": config["simulation"], "orientation_control": "fixed_downward",
              "demo_transition_mode": "contact", "label_source": "scripted_recovery_expert",
              "human_contribution": "Existing measured carry-path curvature and timing only; recovery, grasp and release are simulator-expert labels.",
              "prefix_provenance": {key: value for key, value in job.items() if key != "clip"},
              "physics": "One initial reset; no reset, object pose assignment or attachment after policy-prefix replay starts.",
              "video": str(video_path), "episode": None,
              "action_contract": "Hybrid OSC_POSE: xyz*0.05m; zero/ignored rotation slots with downward feedback; gripper -1 open,+1 close",
              "retargeting": "Original measured carry with endpoint alignment/scaling and height limits; expert open/retract/reapproach before grasp.",
              "success_criterion": "Cube centre previously above table+0.07m, now inside box, no gripper contact and speed<0.05m/s"}
    env = recorder = writer = None
    started = time.monotonic()
    try:
        check_deadline(deadline)
        random.seed(job["seed"])
        np.random.seed(job["seed"])
        env = make_env(config, seed=job["seed"])
        writer = imageio.get_writer(video_path, fps=20, codec="libx264", quality=7, macro_block_size=1, ffmpeg_params=["-threads", "2"])
        def write_frame(obs, phase, step):
            images = np.concatenate((obs["observation.images.image"], obs["observation.images.image2"]), axis=1)
            canvas = np.zeros((images.shape[0] + 32, images.shape[1], 3), np.uint8)
            canvas[32:] = images
            cv2.putText(canvas, f"seed {job['seed']} | step {step} | {phase}", (7, 21), cv2.FONT_HERSHEY_SIMPLEX, .42, (240, 240, 240), 1, cv2.LINE_AA)
            writer.append_data(canvas)
        obs, prefix = replay_prefix(env, trace, job["prefix_steps"], deadline=deadline, tolerance=tolerance, write_frame=write_frame)
        report["prefix_reproduction"] = prefix
        report["takeover_diagnostics"] = env.grasp_diagnostics()
        if env._max_cube_height > TABLE_Z + .07 or report["takeover_diagnostics"]["firm_bilateral_grasp"]:
            raise ValueError("Takeover already lifted or firmly grasped; this pilot targets empty-gripper failures")
        recorder = PilotRecorder(env, obs, deadline, write_frame)
        expert_recovery(recorder, trajectory)
        report["success"] = bool(env.success() and recorder.max_cube_height > TABLE_Z + .07)
        if not report["success"]:
            report["failure"] = "Expert rollout finished without physical placement success"
    except (RuntimeError, ValueError, OSError) as error:
        report["failure"] = f"{type(error).__name__}: {error}"
    finally:
        if recorder is not None:
            report.update(frames=len(recorder.actions), phases=recorder.phases,
                          phase_transition_evidence=recorder.transitions,
                          max_cube_height_m=recorder.max_cube_height,
                          labelled_frame_start="First expert action after prefix; no policy actions in exported arrays")
            if report["success"]:
                np.savez_compressed(output, state=np.asarray(recorder.states, np.float32),
                    image=np.asarray(recorder.images, np.uint8), image2=np.asarray(recorder.images2, np.uint8),
                    action=np.asarray(recorder.actions, np.float32))
                report["episode"] = str(output)
        if env is not None:
            report["final_cube_xyz"] = env.cube_position.tolist()
            report["final_diagnostics"] = env.grasp_diagnostics()
            if writer is not None and env._raw_obs is not None:
                final_obs = env._convert(env._raw_obs)
                write_frame(final_obs, "RECOVERY PLACED" if report["success"] else "RECOVERY FAILED", env.steps)
            env.close()
        if writer is not None:
            writer.close()
        report["wall_seconds"] = time.monotonic() - started
        write_json(report_path, report)
    if report["success"]:
        arrays, _ = _validate_episode(output, ROOT)
        if len(arrays["action"]) != report["frames"]:
            raise RuntimeError("Exported recovery frame count mismatch")
        del arrays
    return report


def validate_collection_plan(plan, clips):
    """This experiment has exactly 20 predeclared jobs, with no replacements."""
    expected_jobs = [{"seed": seed, "prefix_steps": prefix,
                      "clip": RECOVERY_CLIPS[index % len(RECOVERY_CLIPS)]}
                     for index, (seed, prefix) in enumerate(
                         (seed, prefix) for seed in RECOVERY_SEEDS for prefix in RECOVERY_PREFIXES)]
    expected = {"selection_name": "recovery", "require_all_successful": True,
                "source_phase": "development", "source_candidate": "original_h5",
                "source_group": "known_wide", "jobs": expected_jobs}
    if tuple(clips) != RECOVERY_CLIPS or plan != expected:
        raise ValueError("Recovery collection requires the fixed 20 seed/prefix pairs and cyclic five-clip assignment")
    if any(type(job["seed"]) is not int or type(job["prefix_steps"]) is not int for job in plan["jobs"]):
        raise ValueError("Recovery seeds and prefixes must be integers")
    return deepcopy(expected)


def _job_stem(job):
    return f"{job['clip']}_seed_{job['seed']}_prefix_{job['prefix_steps']}"


def _prepare_collection(config, plan_path, config_path, study, destination, dataset, original_dir):
    plan = validate_collection_plan(read_json(plan_path), config["batch"]["clips"])
    training = config["training"]
    required = {"steps": 2000, "batch_size": 4, "seed": 42, "optimizer_lr": 1e-5,
                "preserve_normalization": True, "n_action_steps": 5, "chunk_size": 20}
    if any(training.get(key) != value for key, value in required.items()):
        raise ValueError("Recovery training must keep the predeclared parent, updates, optimizer and normalization")
    checkpoint = project_path(config, training["policy_path"]).resolve()
    if checkpoint != (ROOT / "checkpoints/main").resolve():
        raise ValueError("Recovery training parent must remain checkpoints/main")
    study_plan = read_json(study / "study_plan.json")
    decision = study / "recovery_plan.json"
    if not decision.is_file():
        raise ValueError("Write the recovery_plan.json decision sidecar before freezing collection")
    folder = study / "development/original_h5/known_wide"
    manifest_path, report_path = folder / "manifest.json", folder / "evaluation/report.json"
    source_manifest, source_report = read_json(manifest_path), read_json(report_path)
    if (source_manifest["study_plan_sha256"] != digest(study / "study_plan.json")
            or source_manifest["identity"]["checkpoint_files"] != study_plan["original_checkpoint_files"]
            or source_manifest["identity"]["n_action_steps"] != 5
            or (source_manifest["phase"], source_manifest["group"], source_manifest["candidate"])
            != ("development", "known_wide", "original_h5")):
        raise ValueError("Saved prefixes do not identify the frozen original development run")
    for name, wanted in study_plan["original_checkpoint_files"].items():
        if digest(checkpoint / name) != wanted:
            raise ValueError(f"Original checkpoint file changed: {name}")
    for name, wanted in source_manifest["source_sha256"].items():
        if digest(ROOT / name) != wanted:
            raise ValueError(f"Source evaluator changed since prefix recording: {name}")
    replay_config = deepcopy(config)
    replay_config["simulation"] = deepcopy(source_manifest["config"]["simulation"])
    if replay_config["task"] != source_manifest["config"]["task"]:
        raise ValueError("Recovery task differs from the saved policy prefix")
    tracks = prepare_tracks(config, skip_prepare=True)
    original = _original_manifest(config, original_dir, tracks)
    if set(RECOVERY_SEEDS) & {episode["seed"] for episode in original}:
        raise ValueError("Recovery source seeds overlap original training seeds")
    protected = {settings["seed"] + index
                 for phase in ("development", "final_test")
                 for group, settings in study_plan[phase].items() if group != "known_wide"
                 for index in range(settings["episodes"])}
    if set(RECOVERY_SEEDS) & protected:
        raise ValueError("Recovery source seeds overlap selection or final evaluation seeds")
    raw_hashes, trajectory_hashes, jobs = {}, {}, []
    for clip, path in tracks.items():
        source = read_json(path)["source"]
        raw = project_path(config, source["video"])
        raw_hashes[source["video"]] = digest(raw)
        trajectory_hashes[clip] = digest(path)
    if raw_hashes != study_plan["source_videos"]:
        raise ValueError("Recovery sources differ from the five frozen personal recordings")
    for item in plan["jobs"]:
        matching = [(index, ep) for index, ep in enumerate(source_report["episodes"]) if ep["seed"] == item["seed"]]
        if len(matching) != 1 or matching[0][1]["success"] is not False:
            raise ValueError("Recovery requires a recorded failed original development episode")
        index = matching[0][0]
        trace_path = folder / "evaluation" / f"episode_{index:03d}_trace.json"
        trace = read_json(trace_path)
        prefix = item["prefix_steps"]
        if len(trace) <= prefix or [row.get("step") for row in trace[:prefix + 1]] != list(range(prefix + 1)):
            raise ValueError("Recovery trace does not cover its complete fixed prefix")
        diagnostics = trace[prefix]["diagnostics"]
        if (diagnostics["ever_lifted"] or diagnostics["firm_bilateral_grasp"]
                or diagnostics["gripper_contact"] or not TABLE_Z < trace[prefix]["cube_xyz"][2] < TABLE_Z + .07):
            raise ValueError("Takeover is not a pre-lift empty-gripper state near the table")
        jobs.append({**item, "source_phase": "development", "source_candidate": "original_h5", "source_group": "known_wide",
                     "trace": str(trace_path), "trace_sha256": digest(trace_path),
                     "source_report_sha256": digest(report_path), "source_manifest_sha256": digest(manifest_path),
                     "checkpoint_files": study_plan["original_checkpoint_files"],
                     "trajectory": str(tracks[item["clip"]]), "trajectory_sha256": trajectory_hashes[item["clip"]]})
    source_files = ("scripts/build_recovery_dataset.py", "scripts/build_dataset.py", "scripts/build_selected_dataset.py",
                    "phone_to_panda/common.py", "phone_to_panda/learning.py", "phone_to_panda/simulation.py")
    manifest = {"schema_version": 2, "kind": "fixed_recovery_collection", "selection_plan": plan, "jobs": jobs,
                "selection_plan_sha256": digest(plan_path), "config_file_sha256": digest(config_path),
                "config": {key: value for key, value in config.items() if key != "_root"},
                "replay_simulation": replay_config["simulation"], "task": config["task"],
                "study_plan_sha256": digest(study / "study_plan.json"), "decision_sha256": digest(decision),
                "source_sha256": {name: digest(ROOT / name) for name in source_files},
                "checkpoint_files": study_plan["original_checkpoint_files"], "raw_video_sha256": raw_hashes,
                "trajectory_sha256": trajectory_hashes, "original": original, "original_dir": str(original_dir),
                "robot_dir": str(destination), "dataset_dir": str(dataset), "takeover_tolerance": 1e-6,
                "labels": "Only expert actions after takeover; preserved policy prefix is unlabelled context",
                "evaluation": "No rescue controller in evaluation; source seeds recorded at top level to prevent overlap"}
    return manifest, replay_config


def _resume_report(stem, signature):
    receipt_path = stem.with_suffix(".receipt.json")
    artifacts = (".attempt.json", ".npz", ".json", ".mp4")
    if not receipt_path.exists():
        if any(stem.with_suffix(suffix).exists() for suffix in artifacts):
            raise ValueError(f"Incomplete recovery attempt retained; do not retry or overwrite: {stem}")
        return None
    receipt = read_json(receipt_path)
    if receipt.get("signature") != signature or set(receipt.get("artifacts", {})) != set(artifacts):
        raise ValueError("Recovery receipt differs from the frozen job")
    for suffix, wanted in receipt["artifacts"].items():
        path = stem.with_suffix(suffix)
        if (wanted is None and path.exists()) or (wanted is not None and (not path.is_file() or digest(path) != wanted)):
            raise ValueError(f"Recovery artifact changed: {path}")
    report = read_json(stem.with_suffix(".json"))
    if type(report.get("success")) is not bool or bool(receipt["artifacts"][".npz"]) != report["success"]:
        raise ValueError("Recovery success and episode artifact disagree")
    return report


def _dataset_files(dataset):
    return {str(path.relative_to(dataset)): digest(path) for path in sorted(dataset.rglob("*")) if path.is_file()}


def _export_collection(config, destination, dataset, plan_hash):
    receipt_path = destination / "build/export_receipt.json"
    if dataset.exists():
        if not receipt_path.is_file():
            raise ValueError("Incomplete or unrelated dataset retained without an export receipt")
        receipt = read_json(receipt_path)
        if receipt.get("plan_sha256") != plan_hash or receipt.get("files") != _dataset_files(dataset):
            raise ValueError("Existing export changed; preserve it and inspect before resuming")
        exported = read_json(dataset / "meta/phone_to_panda_provenance.json")
    else:
        if receipt_path.exists():
            raise ValueError("Previously exported dataset is missing")
        exported = export_dataset(config, demos_dir=destination, output_dir=dataset)
    current = []
    for path in sorted(destination.glob("*.npz")):
        arrays, metadata = _validate_episode(path, ROOT)
        current.append({"episode_index": len(current), "frames": len(arrays["action"]), **metadata})
        del arrays
    if exported.get("provenance") != current or exported.get("episodes") != 40:
        raise ValueError("Exported provenance must match exactly 20 original plus 20 recovery episodes")
    if not receipt_path.exists():
        write_json(receipt_path, {"plan_sha256": plan_hash, "files": _dataset_files(dataset)})
    return exported


def build_recovery_collection(config, plan_path, config_path, study, robot_dir, dataset_dir, *,
                              skip_export=False, export_only=False, deadline=None):
    if skip_export and export_only:
        raise ValueError("--skip-export and --export-only are mutually exclusive")
    destination, dataset = Path(robot_dir).resolve(), Path(dataset_dir).resolve()
    original_dir = (ROOT / "data/robot/main").resolve()
    if (destination == original_dir or destination.is_relative_to(original_dir) or original_dir.is_relative_to(destination)
            or destination == dataset or destination.is_relative_to(dataset) or dataset.is_relative_to(destination)
            or dataset == (ROOT / "data/datasets/main").resolve()):
        raise ValueError("Use separate new robot/dataset directories and preserve original data")
    manifest, replay_config = _prepare_collection(config, plan_path, config_path, study, destination, dataset, original_dir)
    build = destination / "build"
    build.mkdir(parents=True, exist_ok=True)
    with (build / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("This recovery directory is already being built") from error
        plan_path = build / "plan.json"
        if plan_path.exists():
            if read_json(plan_path) != manifest:
                raise ValueError("Frozen recovery inputs changed; preserve this collection")
        else:
            if export_only or dataset.exists() or any(p.name != ".lock" for p in build.iterdir()) or any(p.name != "build" for p in destination.iterdir()):
                raise ValueError("No verified recovery plan exists for these output directories")
            write_json(plan_path, manifest)
        plan_hash = _signature(manifest)
        candidates = build / "candidates"
        candidates.mkdir(exist_ok=True)
        expected = {episode["name"] for episode in manifest["original"]} | {_job_stem(job) + ".npz" for job in manifest["jobs"]}
        if any(path.name != "build" and (path.suffix not in (".npz", ".json") or path.with_suffix(".npz").name not in expected) for path in destination.iterdir()):
            raise ValueError("Unexpected files in recovery export directory")
        allowed_candidates = {_job_stem(job) + suffix for job in manifest["jobs"] for suffix in (".attempt.json", ".receipt.json", ".npz", ".json", ".mp4")}
        if any(path.name not in allowed_candidates for path in candidates.iterdir()):
            raise ValueError("Unexpected or incomplete files in recovery candidate directory")
        reports, successful = [], []
        for job in manifest["jobs"]:
            stem = candidates / _job_stem(job)
            signature = _signature({"plan_sha256": plan_hash, "job": job})
            report = _resume_report(stem, signature)
            if report is None:
                if export_only:
                    break
                if deadline is None:
                    raise ValueError("Collection requires an explicit wall-time deadline")
                check_deadline(deadline)
                write_json(stem.with_suffix(".attempt.json"), {"signature": signature, "job": job})
                report = collect_job(replay_config, job, candidates, deadline, manifest["takeover_tolerance"])
                write_json(stem.with_suffix(".receipt.json"), {"signature": signature, "artifacts": {
                    suffix: digest(stem.with_suffix(suffix)) if stem.with_suffix(suffix).exists() else None
                    for suffix in (".attempt.json", ".npz", ".json", ".mp4")}})
            if report.get("seed") != job["seed"] or report.get("prefix_provenance") != {key: value for key, value in job.items() if key != "clip"}:
                raise ValueError("Recovery metadata does not preserve the source seed and fixed job provenance")
            reports.append({"clip": job["clip"], "seed": job["seed"], "prefix_steps": job["prefix_steps"],
                            **{key: report.get(key) for key in ("success", "frames", "failure")}})
            if report["success"]:
                arrays, _ = _validate_episode(stem.with_suffix(".npz"), ROOT)
                del arrays
                successful.append(stem.with_suffix(".npz"))
            summary = {"kind": "fixed_recovery_collection", "complete": len(successful) == 20,
                       "plan_sha256": plan_hash, "planned": 20, "attempted": len(reports), "successes": len(successful),
                       "original_successes": 20, "source_recordings": 5, "new_human_recordings": 0,
                       "training_started": False, "reports": reports, "dataset": str(dataset)}
            write_json(build / "summary.json", summary)
            print(json.dumps(reports[-1]), flush=True)
            if not report["success"]:
                raise ValueError("A fixed recovery attempt failed; retained artifacts, no replacement or dataset export")
        if len(successful) != 20:
            raise ValueError("All 20 fixed recovery jobs must succeed before copying or exporting training data")
        for path in [original_dir / episode["name"] for episode in manifest["original"]] + successful:
            for suffix in (".npz", ".json"):
                _copy_identical(path.with_suffix(suffix), (destination / path.name).with_suffix(suffix))
        if not skip_export:
            exported = _export_collection(config, destination, dataset, plan_hash)
            summary["exported_episodes"] = exported["episodes"]
            write_json(build / "summary.json", summary)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="runs/improvement_20261011")
    parser.add_argument("--output", default="runs/improvement_20261011/recovery_pilot")
    parser.add_argument("--seeds", nargs="+", type=int, default=[3010, 3012, 3013])
    parser.add_argument("--prefix-steps", type=int, default=80)
    parser.add_argument("--collection-plan", help="Opt into the fixed 20-job collection instead of the independent pilot")
    parser.add_argument("--config", default="configs/improvement_20261011/recovery.yaml")
    parser.add_argument("--robot-dir", help="New robot directory for 20 unchanged originals plus 20 recoveries")
    parser.add_argument("--dataset-dir", help="New standard LeRobot dataset output")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--skip-export", action="store_true")
    mode.add_argument("--export-only", action="store_true")
    parser.add_argument("--wall-seconds", type=int, default=1200)
    parser.add_argument("--deadline-utc", help="Optional absolute ISO timestamp, further limiting the wall budget")
    args = parser.parse_args(argv)
    if not 1 <= len(args.seeds) <= 3 or len(set(args.seeds)) != len(args.seeds) or not 0 < args.wall_seconds <= 1200:
        parser.error("Pilot is limited to three unique seeds and at most 1200 wall seconds")
    remaining = args.wall_seconds
    if args.deadline_utc:
        remaining = min(remaining, (datetime.fromisoformat(args.deadline_utc.replace("Z", "+00:00")) - datetime.now(timezone.utc)).total_seconds())
    if remaining <= 0:
        parser.error("Pilot deadline has already passed")
    deadline = time.monotonic() + remaining
    study, destination = (ROOT / args.study).resolve(), (ROOT / args.output).resolve()
    if args.collection_plan:
        try:
            with wall_timer(remaining):
                config_path = project_path({"_root": str(ROOT)}, args.config)
                config = load_config(config_path)
                result = build_recovery_collection(config, project_path(config, args.collection_plan), config_path, study,
                    project_path(config, args.robot_dir or config["paths"]["robot"]),
                    project_path(config, args.dataset_dir or config["paths"]["dataset"]),
                    skip_export=args.skip_export, export_only=args.export_only, deadline=deadline)
            print(json.dumps(result, indent=2))
            return 0
        except (ValueError, OSError, RuntimeError) as error:
            print(f"Recovery collection: {error}", file=sys.stderr)
            return 1
    if args.robot_dir or args.dataset_dir or args.skip_export or args.export_only:
        parser.error("Collection/export options require --collection-plan")
    if not destination.is_relative_to(ROOT / "runs"):
        parser.error("Pilot output must be a new directory under runs/")
    plan = read_json(study / "study_plan.json")
    folder = study / "development/original_h5/known_wide"
    manifest, evaluation = read_json(folder / "manifest.json"), read_json(folder / "evaluation/report.json")
    if (manifest["study_plan_sha256"] != digest(study / "study_plan.json")
            or manifest["identity"]["checkpoint_files"] != plan["original_checkpoint_files"]):
        parser.error("Source development run does not match the frozen original model/study")
    for path, expected in manifest["source_sha256"].items():
        if digest(ROOT / path) != expected:
            parser.error(f"Source evaluator changed since prefix recording: {path}")
    config = deepcopy(manifest["config"])
    config["_root"] = str(ROOT)
    clips = load_config(ROOT / "configs/main.yaml")["batch"]["clips"]
    jobs = []
    for index, seed in enumerate(args.seeds):
        episode_index = next((i for i, episode in enumerate(evaluation["episodes"]) if episode["seed"] == seed), None)
        if episode_index is None or evaluation["episodes"][episode_index]["success"]:
            parser.error("Only recorded failed development seeds are eligible")
        trace_path = folder / "evaluation" / f"episode_{episode_index:03d}_trace.json"
        trajectory = ROOT / "data/processed" / f"{clips[index]}.json"
        jobs.append({"seed": seed, "clip": clips[index], "prefix_steps": args.prefix_steps,
                     "source_phase": "development", "source_candidate": "original_h5", "source_group": "known_wide",
                     "trace": str(trace_path), "trace_sha256": digest(trace_path),
                     "source_report_sha256": digest(folder / "evaluation/report.json"),
                     "source_manifest_sha256": digest(folder / "manifest.json"),
                     "checkpoint_files": manifest["identity"]["checkpoint_files"],
                     "trajectory": str(trajectory), "trajectory_sha256": digest(trajectory)})
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "plan.json", {"kind": "recovery_pilot_only", "jobs": jobs, "prefix_steps": args.prefix_steps,
        "script_sha256": digest(Path(__file__)), "study_plan_sha256": digest(study / "study_plan.json"),
        "wall_limit_seconds": remaining, "takeover_tolerance": 1e-6, "go_threshold": "At least two successful physical recoveries; no automatic expansion or training"})
    def timeout_handler(signum, frame):
        raise TimeoutError("Recovery pilot wall-time limit reached")
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    reports = []
    try:
        for job in jobs:
            if time.monotonic() >= deadline:
                break
            result = collect_job(config, job, destination, deadline, 1e-6)
            reports.append({key: result.get(key) for key in ("seed", "success", "failure", "frames", "wall_seconds", "episode", "video")})
            summary = {"kind": "recovery_pilot_only", "attempted": len(reports), "planned": len(jobs),
                       "successes": sum(report["success"] for report in reports), "reports": reports,
                       "go_threshold_met": sum(report["success"] for report in reports) >= 2,
                       "training_started": False, "new_human_recordings": 0}
            write_json(destination / "summary.json", summary)
            print(json.dumps(reports[-1]), flush=True)
        return 0 if reports and sum(report["success"] for report in reports) >= 2 else 2
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


if __name__ == "__main__":
    raise SystemExit(main())
