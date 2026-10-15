"""Evaluate one candidate against the frozen development or final-test plan.

Every run records model/config/source hashes. Completed matching runs may be
reused; incomplete or changed runs are retained and rejected, never overwritten.
Final tests require a candidate_lock.json written before looking at test results.
The lock schema is::

    {"study_plan_sha256": "...", "config_sha256": "...",
     "source_sha256": {"phone_to_panda/learning.py": "...",
                       "phone_to_panda/simulation.py": "..."},
     "reference_candidate": "original_h5", "selected_candidate": "uniform_h5",
     "contrast_roles": {"uniform_h5": "Primary selected adaptation"},
     "selection_sidecars_sha256": {"selection_rule.json": "..."},
     "candidates": {"name": {"checkpoint_files": {"model.safetensors": "...", ...},
                              "n_action_steps": 5}}}

``config_sha256`` hashes the unmodified YAML file supplied through ``--config``.
The locked study plan supplies each group's radius and seeds; action horizon is
locked per candidate. All final groups must use the same locked source code.
Selection sidecar paths are relative to the study directory. Final manifests
record ``base_config_sha256`` and ``candidate_lock_sha256`` so summaries and
videos cannot silently adopt a changed selection decision or comparison role.
"""
from __future__ import annotations

import argparse
import hashlib
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


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_hashes():
    return {str(path.relative_to(ROOT)): digest(path) for path in
            [ROOT / "phone_to_panda/learning.py", ROOT / "phone_to_panda/simulation.py"]}


def validate_final_lock(lock, *, study_plan_sha256, config_sha256, source_sha256, candidate, identity):
    """Check the whole final-test context before any evaluation is launched."""
    for field, expected in (("study_plan_sha256", study_plan_sha256),
                            ("config_sha256", config_sha256), ("source_sha256", source_sha256)):
        if lock.get(field) != expected:
            raise ValueError(f"Final-test {field} differs from the locked study context")
    if lock.get("candidates", {}).get(candidate) != identity:
        raise ValueError("Candidate differs from the model/settings locked before final testing")
    candidates = lock["candidates"]
    reference, selected = lock.get("reference_candidate"), lock.get("selected_candidate")
    if reference not in candidates or selected not in candidates:
        raise ValueError("Final lock must name its reference_candidate and selected_candidate")
    roles = lock.get("contrast_roles")
    if (not isinstance(roles, dict) or not (set(candidates) - {reference}) <= set(roles)
            or not set(roles) <= set(candidates)
            or any(not isinstance(value, str) or not value.strip() for value in roles.values())):
        raise ValueError("Final contrast_roles must describe every non-reference candidate")
    if not isinstance(lock.get("selection_sidecars_sha256"), dict) or not lock["selection_sidecars_sha256"]:
        raise ValueError("Final lock requires selection_sidecars_sha256")


def validate_selection_sidecars(study, lock):
    """Selection evidence uses portable study-relative paths, never result files."""
    for name, expected in lock["selection_sidecars_sha256"].items():
        path = (study / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(study.resolve()):
            raise ValueError("Selection sidecar paths must stay inside the study directory")
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Selection sidecar differs from final lock: {name}")


def validate_report(report, *, checkpoint, action_steps, settings, config):
    """Reject mismatched or inconsistent completed reports before reusing them."""
    if report.get("kind") != "trained_policy_physics_evaluation":
        raise ValueError("Evaluation report is not a trained-policy physics result")
    recorded_checkpoint = report.get("checkpoint")
    if not isinstance(recorded_checkpoint, str) or Path(recorded_checkpoint).resolve() != Path(checkpoint).resolve():
        raise ValueError("Evaluation report checkpoint differs from the candidate")
    if (report.get("randomize_start_m") != settings["radius_m"]
            or report.get("seed_start") != settings["seed"]):
        raise ValueError("Evaluation report radius or starting seed differs from the plan")
    horizon = report.get("policy_settings", {}).get("n_action_steps")
    if type(horizon) is not int or horizon != action_steps:
        raise ValueError("Evaluation report action horizon differs from the candidate")
    if report.get("orientation_control") != config["simulation"].get("orientation_control", "policy"):
        raise ValueError("Evaluation report orientation controller differs from the configuration")
    if report.get("held_out_from_training_seeds") is not True:
        raise ValueError("Evaluation report did not verify held-out seeds")
    if report.get("video_recorded") is not bool(config["evaluation"].get("save_video", True)):
        raise ValueError("Evaluation report video setting differs from the configuration")
    episodes = report.get("episodes")
    expected = list(range(settings["seed"], settings["seed"] + settings["episodes"]))
    if (not isinstance(episodes, list) or not expected
            or any(not isinstance(ep, dict) for ep in episodes)
            or [ep.get("seed") for ep in episodes] != expected):
        raise ValueError("Evaluation did not complete every planned seed")
    if any(type(ep.get("success")) is not bool for ep in episodes):
        raise ValueError("Evaluation episode success must be a boolean")
    successes = sum(ep["success"] for ep in episodes)
    if type(report.get("successes")) is not int or report["successes"] != successes:
        raise ValueError("Evaluation success count disagrees with its episodes")
    rate = report.get("success_rate")
    if (type(rate) not in (int, float) or not math.isfinite(rate)
            or not math.isclose(rate, successes / len(episodes), rel_tol=0, abs_tol=1e-12)):
        raise ValueError("Evaluation success rate disagrees with its episodes")
    duration = report.get("wall_seconds")
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
        raise ValueError("Evaluation duration must be finite and nonnegative")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="runs/improvement_20261011")
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--action-steps", type=int, required=True)
    parser.add_argument("--phase", choices=("development", "final_test"), default="development")
    parser.add_argument("--groups", nargs="+")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args(argv)
    if not args.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.name):
        parser.error("Use a simple candidate name containing letters, digits, '_' or '-'.")
    study = (ROOT / args.study).resolve()
    checkpoint = (ROOT / args.checkpoint).resolve()
    plan = json.loads((study / "study_plan.json").read_text())
    groups = plan[args.phase]
    selected_groups = args.groups or list(groups)
    if any(g not in groups for g in selected_groups) or len(set(selected_groups)) != len(selected_groups):
        parser.error("Groups must be unique names present in the selected study phase.")
    checkpoint_files = {p.name: digest(p) for p in sorted(checkpoint.iterdir()) if p.is_file()}
    if "model.safetensors" not in checkpoint_files:
        raise ValueError("Missing checkpoint weights")
    identity = {"checkpoint_files": checkpoint_files, "n_action_steps": args.action_steps}
    if args.phase == "final_test":
        lock_path = study / "candidate_lock.json"
        lock_bytes = lock_path.read_bytes()
        lock = json.loads(lock_bytes)
        lock_hash = hashlib.sha256(lock_bytes).hexdigest()
        validate_final_lock(lock, study_plan_sha256=digest(study / "study_plan.json"),
                            config_sha256=digest(ROOT / args.config), source_sha256=source_hashes(),
                            candidate=args.name, identity=identity)
        validate_selection_sidecars(study, lock)
    reports = {}
    for group in selected_groups:
        settings = groups[group]
        if args.phase == "final_test":
            if digest(lock_path) != lock_hash:
                raise ValueError("Candidate selection lock changed during final evaluation")
            validate_final_lock(lock, study_plan_sha256=digest(study / "study_plan.json"),
                                config_sha256=digest(ROOT / args.config), source_sha256=source_hashes(),
                                candidate=args.name, identity=identity)
            validate_selection_sidecars(study, lock)
        config = load_config(ROOT / args.config)
        config["simulation"]["randomize_start_m"] = settings["radius_m"]
        config["evaluation"].update(n_action_steps=args.action_steps, save_video=args.video, save_trace=True)
        manifest = {"candidate": args.name, "phase": args.phase, "group": group,
                    "identity": identity, "config": {k: v for k, v in config.items() if k != "_root"},
                    "episode_plan": settings, "study_plan_sha256": digest(study / "study_plan.json"),
                    "source_sha256": source_hashes()}
        if args.phase == "final_test":
            manifest["base_config_sha256"] = digest(ROOT / args.config)
            manifest["candidate_lock_sha256"] = lock_hash
        destination = study / args.phase / args.name / group
        manifest_path = destination / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError(f"Inputs changed for {destination}; preserve it and choose a new candidate name")
        else:
            destination.mkdir(parents=True, exist_ok=False)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        report_path = destination / "evaluation/report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
        else:
            report = run_policy(config, checkpoint, destination / "evaluation",
                                episodes=settings["episodes"], seed=settings["seed"])
        validate_report(report, checkpoint=checkpoint, action_steps=args.action_steps,
                        settings=settings, config=config)
        reports[group] = {"successes": report["successes"], "episodes": len(report["episodes"]),
                          "wall_seconds": report["wall_seconds"], "report": str(report_path.relative_to(ROOT))}
        print(json.dumps({"candidate": args.name, "phase": args.phase, "group": group, **reports[group]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
