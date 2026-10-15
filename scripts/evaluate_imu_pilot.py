"""Evaluate one fixed IMU-pilot model on both planned simulator-start groups.

Completed runs are reusable only when their inputs and retained outputs match.
Interrupted runs are kept and refused, never overwritten or silently repeated.
This small same-scene pilot is not model selection or a sensor-fusion study.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from phone_to_panda.common import load_config
from phone_to_panda.learning import run_policy
from scripts.evaluate_candidate import digest, validate_report

STUDY = Path("results/imu_pilot_20261015")
MODELS = ("original", "continuation", "augmented")
GROUPS = ("narrow", "wide")
EVENTS = ("contact_step", "bilateral_contact_step", "firm_grasp_step", "lift_step",
          "release_after_lift_step")


def read_json(path):
    return json.loads(Path(path).read_text())


def retain_json(path, value):
    """Write once, or verify an existing identical record without changing it."""
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Existing record differs; keep it for review: {path}")
        return
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def file_hashes(folder):
    return {str(path.relative_to(folder)): digest(path)
            for path in sorted(folder.rglob("*")) if path.is_file()}


def input_identity(model, plan, config_path, config, checkpoint):
    """Snapshot the actual weights, scene, code, dataset provenance and take."""
    weights = file_hashes(checkpoint)
    if any(name not in weights for name in ("model.safetensors", "config.json", "train_config.json")):
        raise ValueError("A completed checkpoint with weights and training configuration is required")
    if "SETUP_TEST_ONLY.json" in weights:
        raise ValueError("Setup-test weights cannot be used in this pilot")
    code = ["phone_to_panda/common.py", "phone_to_panda/learning.py",
            "phone_to_panda/simulation.py", "scripts/evaluate_candidate.py",
            "scripts/evaluate_imu_pilot.py"]
    trajectory_path = ROOT / "data/experimental/imu_20261015/trajectory.json"
    trajectory = read_json(trajectory_path)
    source = trajectory["source"]
    if (source.get("kind") != "personal_video_imu_pilot"
            or Path(source["video"]).name != plan["paired_video"]
            or source.get("session_id") != plan["sensor_session_id"]):
        raise ValueError("Pilot trajectory differs from the planned paired recording")
    recording = {}
    for path_field, hash_field in (("video", "sha256"), ("imu_session", "imu_sha256"),
                                   ("events", "events_sha256")):
        path = (ROOT / source[path_field]).resolve()
        actual = digest(path)
        if actual != source[hash_field]:
            raise ValueError(f"Pilot recording hash differs: {path_field}")
        recording[path_field] = {"path": str(path), "sha256": actual}
    events = read_json(ROOT / source["events"])
    if (events.get("imu_time_offset_s") != plan["primary_offset_s"]
            or events.get("synchronization") != plan["offset_status"]):
        raise ValueError("Pilot timing differs from the planned provisional offset")
    evidence_paths = [trajectory_path, ROOT / STUDY / "dataset_inputs.json",
                      ROOT / STUDY / "dataset_summary.json", ROOT / STUDY / "replays.json",
                      ROOT / config["paths"]["dataset"] / "meta/phone_to_panda_provenance.json"]
    # run_policy can fall back to the saved training dataset's provenance.
    if "phone_to_panda_provenance.json" not in weights:
        saved = read_json(checkpoint / "train_config.json")
        evidence_paths.append(ROOT / saved["dataset"]["root"] / "meta/phone_to_panda_provenance.json")
    return {
        "model": model, "checkpoint": str(checkpoint), "checkpoint_files_sha256": weights,
        "plan_sha256": digest(ROOT / STUDY / "plan.json"),
        "base_config": str(config_path), "base_config_sha256": digest(config_path),
        "source_sha256": {name: digest(ROOT / name) for name in code},
        "scene_assets_sha256": file_hashes(ROOT / "assets"),
        "recording": recording,
        "evidence_sha256": {str(path.resolve()): digest(path) for path in evidence_paths},
    }


def failure_category(episode):
    """Describe the last observed stage; these labels are not causal diagnoses."""
    if episode["success"]:
        return None
    events = episode["first_events"]
    if events["lift_step"] is not None:
        return "lifted_without_successful_placement"
    if events["firm_grasp_step"] is not None:
        return "firm_grasp_without_lift"
    if events["contact_step"] is not None:
        return "contact_without_firm_grasp"
    return "no_gripper_contact"


def checked_episodes(report, directory):
    """Require every saved video and a complete, matching action trace."""
    episodes = read_json(directory / "episodes.json")
    if episodes != report["episodes"]:
        raise ValueError("Saved episode list differs from the completed report")
    rows = []
    for index, episode in enumerate(episodes):
        steps = episode.get("steps")
        if type(steps) is not int or steps < 1:
            raise ValueError("Episode must contain a positive number of physics steps")
        events = episode.get("first_events")
        if (not isinstance(events, dict) or set(events) != set(EVENTS)
                or any(value is not None and (type(value) is not int or not 0 <= value < steps)
                       for value in events.values())):
            raise ValueError("Episode is missing valid contact and lift event indices")
        height = episode.get("max_cube_height_m")
        if type(height) not in (int, float) or not math.isfinite(height):
            raise ValueError("Episode must report a finite maximum cube height")
        if episode["success"] and events["lift_step"] is None:
            raise ValueError("Successful placement must have a recorded lift")
        video_name = f"episode_{index:03d}.mp4"
        video = directory / video_name
        if episode.get("video") != video_name or not video.is_file() or video.stat().st_size == 0:
            raise ValueError("Episode video is missing or empty")
        trace_name = f"episode_{index:03d}_trace.json"
        trace = read_json(directory / trace_name)
        if (not isinstance(trace, list) or len(trace) != steps
                or any(not isinstance(row, dict) for row in trace)
                or [row.get("step") for row in trace] != list(range(steps))
                or any(row.get("episode_seed") != episode["seed"] for row in trace)
                or trace[-1].get("success") is not episode["success"]):
            raise ValueError("Action trace is incomplete or differs from the episode")
        rows.append({**episode, "failure_category": failure_category(episode),
                     "trace": trace_name})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    args = parser.parse_args(argv)
    plan = read_json(ROOT / STUDY / "plan.json")
    evaluation = plan["evaluation"]
    if (evaluation.get("models") != list(MODELS) or evaluation.get("n_action_steps") != 5
            or any(evaluation[group].get("episodes") != 5 for group in GROUPS)):
        raise ValueError("Expected the fixed three-model, five-seed-per-group pilot plan")
    config_name = "augmented" if args.model == "augmented" else "continuation"
    config_path = ROOT / f"configs/imu_pilot_20261015/{config_name}.yaml"
    config = load_config(config_path)
    checkpoint = (ROOT / (plan["training"]["initial_checkpoint"] if args.model == "original" else
                         f"runs/imu_pilot_20261015/train_{args.model}/checkpoints/last/pretrained_model")).resolve()
    identity = input_identity(args.model, plan, config_path, config, checkpoint)
    destination = ROOT / STUDY / "evaluation" / args.model
    manifest_path = destination / "identity.json"
    if destination.exists() and not manifest_path.is_file():
        raise ValueError(f"Existing evaluation has no identity; retain it for review: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    retain_json(manifest_path, identity)
    prepared = []
    # Check both groups before starting anything, including when resuming.
    for group in GROUPS:
        settings = evaluation[group]
        active = deepcopy(config)
        active["simulation"]["randomize_start_m"] = settings["radius_m"]
        active["evaluation"].update(episodes=settings["episodes"], seed=settings["seed"],
                                    n_action_steps=evaluation["n_action_steps"], save_video=True, save_trace=True)
        folder = destination / group
        manifest = {"identity_sha256": digest(manifest_path), "group": group, "episode_plan": settings,
                    "config": {key: value for key, value in active.items() if key != "_root"}}
        if folder.exists() and not (folder / "manifest.json").is_file():
            raise ValueError(f"Existing group has no manifest; retain it for review: {folder}")
        folder.mkdir(exist_ok=True)
        retain_json(folder / "manifest.json", manifest)
        output = folder / "evaluation"
        if output.exists():
            if not (output / "report.json").is_file() or not (folder / "outputs.json").is_file():
                raise ValueError(f"Incomplete evaluation retained; do not overwrite: {output}")
            if read_json(folder / "outputs.json") != file_hashes(output):
                raise ValueError(f"Saved evaluation outputs changed: {output}")
            cached = read_json(output / "report.json")
            validate_report(cached, checkpoint=checkpoint, action_steps=5, settings=settings, config=active)
            checked_episodes(cached, output)
        prepared.append((group, settings, active, folder, output))
    reports = {}
    for group, settings, active, folder, output in prepared:
        if input_identity(args.model, plan, config_path, config, checkpoint) != identity:
            raise ValueError("Evaluation inputs changed during the run; all outputs retained")
        report_path = output / "report.json"
        if not output.exists():
            report = run_policy(active, checkpoint, output, episodes=settings["episodes"], seed=settings["seed"])
            if read_json(report_path) != report:
                raise ValueError("Returned evaluation differs from the saved report")
        else:
            report = read_json(report_path)
        validate_report(report, checkpoint=checkpoint, action_steps=5, settings=settings, config=active)
        rows = checked_episodes(report, output)
        if input_identity(args.model, plan, config_path, config, checkpoint) != identity:
            raise ValueError("Evaluation inputs changed during the run; all outputs retained")
        retain_json(folder / "outputs.json", file_hashes(output))
        reports[group] = {
            "radius_m": settings["radius_m"], "successes": report["successes"], "episodes": len(rows),
            "success_rate": report["success_rate"], "episode_results": rows,
            "failure_categories": dict(Counter(row["failure_category"] for row in rows if not row["success"])),
            "wall_seconds": report["wall_seconds"], "report": str(report_path.relative_to(ROOT)),
            "report_sha256": digest(report_path),
        }
        print(json.dumps({"model": args.model, "group": group, "successes": report["successes"],
                          "episodes": len(rows), "report": str(report_path)}), flush=True)
    summary = {
        "kind": "single_recording_imu_assisted_policy_evaluation", "model": args.model,
        "identity_sha256": digest(manifest_path), "groups": reports,
        "successes": sum(group["successes"] for group in reports.values()), "episodes": 10,
        "selection": evaluation["selection"], "interpretation": evaluation["interpretation"],
        "scope": plan["scope"], "source_human_recordings": plan["source_human_recordings"],
        "limitations": "Five starts per group in the same simulated scene. No model selection, held-out human sessions, real-robot result, or sensor-fusion claim. The policy consumes simulated cameras and robot state.",
        "failure_category_meaning": "Last observed contact/lift stage, not a proven failure cause.",
    }
    retain_json(destination / "summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
