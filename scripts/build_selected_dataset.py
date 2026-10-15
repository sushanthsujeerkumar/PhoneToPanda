"""Build a balanced, resumable selection of replays from existing personal videos.

Plan format (all configured clips are required)::

    {"selection_name": "targeted", "successes_per_clip": 4,
     "clips": {"IMG_2092": [{"seed": 1000, "cube_start_xy": [0.01, -0.18]}]}}

Jobs are attempted in their listed order until each clip has its first N successful
replays. A job may also specify ``box_xy``. Starts are exact: seed-dependent start
jitter is disabled. Every attempted replay, including failures, stays under
``robot-dir/build/candidates``; only selected successes and the original validated
replays enter the export directory. Existing inputs and outputs are never replaced.
This is data generation only, not learned-policy evaluation or new human recordings.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_dataset import digest, prepare_tracks, read_json, write_json
from phone_to_panda.common import load_config, project_path
from phone_to_panda.learning import _validate_episode, export_dataset
from phone_to_panda.simulation import record_demo


def _signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _xy(value, label):
    if (not isinstance(value, list) or len(value) != 2
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value)):
        raise ValueError(f"{label} must contain two finite numbers.")
    return [float(x) for x in value]


def validate_plan(plan, clips):
    """Normalize an explicit immutable candidate plan; never synthesize jobs."""
    if not isinstance(plan, dict) or set(plan) != {"selection_name", "successes_per_clip", "clips"}:
        raise ValueError("Plan must have selection_name, successes_per_clip, and clips only.")
    if not isinstance(plan["selection_name"], str) or not plan["selection_name"].strip():
        raise ValueError("selection_name must be a nonempty string.")
    quota = plan["successes_per_clip"]
    if isinstance(quota, bool) or not isinstance(quota, int) or quota < 1:
        raise ValueError("successes_per_clip must be a positive integer.")
    if not clips or len(clips) != len(set(clips)) or any(Path(c).name != c or c in {".", ".."} for c in clips):
        raise ValueError("Configured source clips must be unique simple filenames.")
    if not isinstance(plan["clips"], dict) or set(plan["clips"]) != set(clips):
        raise ValueError("Plan must include exactly the configured source clips.")
    normalized = {"selection_name": plan["selection_name"], "successes_per_clip": quota, "clips": {}}
    seeds = set()
    for clip in clips:
        candidates = plan["clips"][clip]
        if not isinstance(candidates, list) or len(candidates) < quota:
            raise ValueError(f"{clip}: provide at least {quota} ordered candidate jobs.")
        normalized["clips"][clip] = []
        for candidate in candidates:
            if (not isinstance(candidate, dict) or not {"seed", "cube_start_xy"}.issubset(candidate)
                    or set(candidate) - {"seed", "cube_start_xy", "box_xy"}):
                raise ValueError(f"{clip}: jobs require seed and cube_start_xy, with optional box_xy only.")
            seed = candidate["seed"]
            if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32 or seed in seeds:
                raise ValueError("Candidate seeds must be unique integers in [0, 2**32).")
            seeds.add(seed)
            job = {"seed": seed, "cube_start_xy": _xy(candidate["cube_start_xy"], "cube_start_xy")}
            if "box_xy" in candidate:
                job["box_xy"] = _xy(candidate["box_xy"], "box_xy")
            normalized["clips"][clip].append(job)
    return normalized


def _copy_identical(source, destination):
    """Copy into a new location or verify an already completed copy."""
    if destination.exists():
        if not destination.is_file() or digest(source) != digest(destination):
            raise ValueError(f"Existing replay copy differs: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".copying")
    if temporary.exists():
        raise ValueError(f"Incomplete copy retained at {temporary}; inspect it before resuming.")
    shutil.copy2(source, temporary)
    temporary.rename(destination)


def _original_manifest(config, directory, tracks):
    original = []
    for path in sorted(directory.glob("*.npz")):
        arrays, metadata = _validate_episode(path, Path(config["_root"]))
        del arrays
        clip = Path(metadata["source"]["video"]).stem
        if clip not in tracks or metadata["source"] != read_json(tracks[clip])["source"]:
            raise ValueError(f"{path}: original source differs from the accepted trajectory source.")
        if metadata.get("task") != config["task"]:
            raise ValueError(f"{path}: original task differs from the selected build task.")
        original.append({"name": path.name, "sha256": digest(path), "metadata_sha256": digest(path.with_suffix(".json")),
                         "seed": metadata["seed"], "clip": clip})
    coverage = {clip: sum(ep["clip"] == clip for ep in original) for clip in tracks}
    if any(count != 4 for count in coverage.values()) or len(original) != len(tracks) * 4:
        raise ValueError(f"Expected four original successful replays per accepted source clip: {coverage}")
    return original


def _checked_report(path, signature):
    receipt_path = path.with_suffix(".receipt.json")
    if not receipt_path.exists():
        if any(path.with_suffix(suffix).exists() for suffix in (".npz", ".json", ".mp4")):
            raise ValueError(f"Incomplete replay retained at {path}; inspect before resuming.")
        return None
    receipt = read_json(receipt_path)
    if receipt.get("signature") != signature:
        raise ValueError(f"Existing replay has changed inputs: {path}")
    for suffix, expected in receipt["artifacts"].items():
        artifact = path.with_suffix(suffix)
        if (expected is None and artifact.exists()) or (expected is not None and (not artifact.is_file() or digest(artifact) != expected)):
            raise ValueError(f"Existing replay artifact differs: {artifact}")
    return read_json(path.with_suffix(".json"))


def _build(config, plan, destination, dataset, original_dir, skip_export, export_only, max_jobs):
    tracks = prepare_tracks(config, skip_prepare=True)
    plan = validate_plan(plan, list(tracks))
    quota = plan["successes_per_clip"]
    original = _original_manifest(config, original_dir, tracks)
    seeds = [job["seed"] for jobs in plan["clips"].values() for job in jobs] + [ep["seed"] for ep in original]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Candidate and original replay seeds overlap.")
    manifest = {"schema_version": 1, "selection_plan": plan, "task": config["task"],
                "config_sha256": _signature({key: value for key, value in config.items() if key != "_root"}),
                "simulation": config["simulation"], "video_settings": config["video"],
                "trajectory_sha256": {clip: digest(path) for clip, path in tracks.items()},
                "original": original, "original_dir": str(original_dir),
                "robot_dir": str(destination), "dataset_dir": str(dataset),
                "export_repo_id": config.get("training", {}).get("repo_id", "local/phone_to_panda")}
    build = destination / "build"
    plan_path = build / "plan.json"
    allowed = {ep["name"] for ep in original} | {f"{clip}_seed_{job['seed']}.npz" for clip, jobs in plan["clips"].items() for job in jobs}
    unexpected = [p.name for p in destination.iterdir() if p.name != "build" and
                  (p.suffix not in {".npz", ".json"} or p.with_suffix(".npz").name not in allowed)]
    if unexpected:
        raise ValueError(f"Unexpected files in robot directory: {unexpected}")
    if plan_path.exists():
        if read_json(plan_path) != manifest:
            raise ValueError("Replay inputs changed. Preserve the build and choose new robot and dataset paths.")
    else:
        if export_only or any(p.name != ".lock" for p in build.iterdir()) or any(p.name != "build" for p in destination.iterdir()):
            raise ValueError("No verified selection plan exists for this output directory.")
        if dataset.exists():
            raise ValueError("Dataset output already exists without this build's plan; choose a new path.")
        write_json(plan_path, manifest)
    plan_hash = _signature(manifest)
    results, selected, attempts = [], [], 0
    counts = {clip: 0 for clip in tracks}
    for clip, jobs in plan["clips"].items():
        for index, job in enumerate(jobs):
            if counts[clip] >= quota:
                break
            settings = deepcopy(config)
            settings["simulation"]["cube_start_xy"] = job["cube_start_xy"].copy()
            settings["simulation"]["randomize_start_m"] = 0.
            if "box_xy" in job:
                settings["simulation"]["box_xy"] = job["box_xy"].copy()
            selection = {"name": plan["selection_name"], "plan_sha256": plan_hash, "clip": clip,
                         "candidate_index": index, "successes_per_clip": quota, **job,
                         "trajectory_sha256": manifest["trajectory_sha256"][clip],
                         "source_recordings": len(tracks)}
            signature = _signature({"selection": selection, "task": config["task"], "simulation": settings["simulation"]})
            output = build / "candidates" / f"{clip}_seed_{job['seed']}.npz"
            report = _checked_report(output, signature)
            if report is None:
                if export_only or (max_jobs is not None and attempts >= max_jobs):
                    continue
                output.parent.mkdir(parents=True, exist_ok=True)
                report = record_demo(settings, output, trajectory_path=tracks[clip], seed=job["seed"])
                if report.get("source") != read_json(tracks[clip])["source"]:
                    raise ValueError(f"Replay source differs from its measured trajectory: {output}")
                if report.get("success") is True:
                    if report.get("simulation_settings") != settings["simulation"] or report.get("task") != config["task"]:
                        raise ValueError(f"Replay settings differ from the selected job: {output}")
                report["selection"] = selection
                report["build_signature"] = signature
                write_json(output.with_suffix(".json"), report)
                if report.get("success") is True:
                    arrays, _ = _validate_episode(output, Path(config["_root"]))
                    del arrays
                receipt = {"signature": signature, "artifacts": {
                    suffix: digest(output.with_suffix(suffix)) if output.with_suffix(suffix).exists() else None
                    for suffix in (".npz", ".json", ".mp4")}}
                write_json(output.with_suffix(".receipt.json"), receipt)
                attempts += 1
            if report.get("success") is True:
                arrays, _ = _validate_episode(output, Path(config["_root"]))
                del arrays
                counts[clip] += 1
                selected.append(output)
            results.append({"clip": clip, **job, "candidate_index": index,
                            **{key: report.get(key) for key in ("success", "frames", "failure")}})
            write_json(build / "replays.json", results)
            print(f"{clip} {counts[clip]}/{quota} selected: {json.dumps(results[-1])}", flush=True)
    complete = all(count == quota for count in counts.values())
    summary = {"selection_name": plan["selection_name"], "plan_sha256": plan_hash, "complete": complete,
               "source_recordings": len(tracks), "original_successes": len(original),
               "new_successes_per_clip": counts, "new_successes": sum(counts.values()),
               "attempted_candidates": len(results), "failed_candidates": sum(r["success"] is not True for r in results),
               "new_attempts_this_run": attempts, "candidate_directory": str(build / "candidates"),
               "selected_episodes": [p.name for p in selected], "dataset": str(dataset)}
    write_json(build / "summary.json", summary)
    if not complete:
        if export_only or max_jobs is None:
            raise ValueError(f"Balanced replay coverage not met: {counts}; failures retained, no dataset exported.")
        return summary
    expected = {ep["name"] for ep in original} | {p.name for p in selected}
    if any(p.with_suffix(".npz").name not in expected for p in destination.iterdir() if p.name != "build"):
        raise ValueError("Robot directory contains an episode outside the selected successful collection.")
    for path in [original_dir / ep["name"] for ep in original] + selected:
        for suffix in (".npz", ".json"):
            _copy_identical(path.with_suffix(suffix), (destination / path.name).with_suffix(suffix))
    if not skip_export:
        sidecar = dataset / "meta/phone_to_panda_provenance.json"
        if dataset.exists():
            if not sidecar.is_file():
                raise ValueError("Incomplete or unrelated dataset retained; choose a new dataset output path.")
            exported = read_json(sidecar)
            current = []
            for path in sorted(destination.glob("*.npz")):
                arrays, metadata = _validate_episode(path, Path(config["_root"]))
                current.append({"episode_index": len(current), "frames": len(arrays["action"]), **metadata})
                del arrays
            if exported.get("provenance") != current:
                raise ValueError("Existing dataset provenance differs from the selected successful collection.")
        else:
            exported = export_dataset(config, demos_dir=destination, output_dir=dataset)
        if exported["episodes"] != len(expected):
            raise ValueError("Exported count differs from the balanced successful collection.")
        summary["exported_episodes"] = exported["episodes"]
        write_json(build / "summary.json", summary)
    return summary


def build_selected_dataset(config, plan, robot_dir=None, dataset_dir=None, original_dir="data/robot/main", *,
                           skip_export=False, export_only=False, max_jobs=None):
    if skip_export and export_only:
        raise ValueError("--skip-export and --export-only are mutually exclusive.")
    if max_jobs is not None and (isinstance(max_jobs, bool) or not isinstance(max_jobs, int) or max_jobs < 1):
        raise ValueError("max_jobs must be a positive integer.")
    destination = project_path(config, robot_dir or config["paths"]["robot"]).resolve()
    dataset = project_path(config, dataset_dir or config["paths"]["dataset"]).resolve()
    original = project_path(config, original_dir).resolve()
    if destination == original or dataset == destination or dataset.is_relative_to(destination) or destination.is_relative_to(dataset):
        raise ValueError("Use a new robot directory and separate dataset directory; preserve original replays.")
    build = destination / "build"
    build.mkdir(parents=True, exist_ok=True)
    with (build / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("This robot selection directory is already being built.") from error
        return _build(config, plan, destination, dataset, original, skip_export, export_only, max_jobs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--plan", required=True, help="JSON plan containing ordered per-clip candidate jobs")
    parser.add_argument("--robot-dir", required=True, help="New selected robot-demo directory")
    parser.add_argument("--dataset-dir", required=True, help="New LeRobot export directory")
    parser.add_argument("--original-dir", default="data/robot/main")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--skip-export", action="store_true", help="Generate/select replays now; export separately later")
    mode.add_argument("--export-only", action="store_true", help="Verify completed selection and export; never run simulations")
    parser.add_argument("--max-jobs", type=int, help="Maximum new candidate attempts in this invocation; rerun same command to resume")
    args = parser.parse_args(argv)
    try:
        config = load_config(project_path({"_root": str(ROOT)}, args.config))
        result = build_selected_dataset(config, read_json(project_path(config, args.plan)), args.robot_dir, args.dataset_dir,
                                        args.original_dir, skip_export=args.skip_export, export_only=args.export_only,
                                        max_jobs=args.max_jobs)
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Selected dataset build: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
