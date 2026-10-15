"""Prepare measured recordings, check physics replays, and export successful demos.

Existing tracks are reused only after their recording, annotation, calibration,
marker settings and quality checks agree. Changed inputs are never overwritten.
This command builds data only; it does not start training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from phone_to_panda.common import load_config, project_path
from phone_to_panda.learning import _validate_episode, export_dataset
from phone_to_panda.simulation import record_demo
from phone_to_panda.video import extract_video


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_track(config, clip, trajectory, video, events, calibration):
    track = read_json(trajectory)
    source, quality = track.get("source", {}), track.get("quality", {})
    settings, markers = config["video"], track.get("markers", {})
    stale = f"{clip}: existing track differs from its inputs. Preserve it and choose a new processed path."
    for key, path in (("sha256", video), ("events_sha256", events), ("calibration_sha256", calibration)):
        if source.get(key) != digest(path):
            raise ValueError(f"{stale} Changed {key}.")
    if source.get("kind") != "personal_video" or project_path(config, source.get("video", "")).resolve() != video.resolve():
        raise ValueError(f"{clip}: track does not identify the original personal recording.")
    for key in ("table_marker_size_m", "object_marker_size_m", "reference_markers"):
        if markers.get(key) != settings.get(key):
            raise ValueError(f"{stale} Changed {key}.")
    if (markers.get("object_id") != settings.get("object_marker_id", 1)
            or quality.get("tracking_mode", "single_marker") != settings.get("tracking_mode", "single_marker")
            or quality.get("sample_hz") != settings["sample_hz"]
            or quality.get("max_allowed_gap_s") != settings["max_gap_s"]):
        raise ValueError(stale)
    if source.get("reference_board_sha256"):
        board = project_path(config, quality["reference_board"])
        if digest(board) != source["reference_board_sha256"]:
            raise ValueError(f"{stale} Changed reference board.")
    if (quality.get("accepted") is not True
            or quality.get("measured_fraction", 0) < settings.get("min_valid_fraction", .90)
            or quality.get("max_pose_reprojection_error_px", float("inf")) > settings.get("max_pose_reprojection_error_px", 3.)
            or quality.get("max_table_corner_drift_px", float("inf")) > settings.get("max_table_drift_px", 10.)):
        raise ValueError(f"{clip}: measured track fails the unchanged tracking quality gates.")
    resting = quality.get("resting_height_check")
    if resting:
        low, high = settings.get("resting_height_range_m", [-.01, .15])
        if not low <= resting["median_marker_top_height_m"] <= high:
            raise ValueError(f"{clip}: resting marker height fails the configured range.")
    return track


def prepare_tracks(config, skip_prepare=False):
    raw = project_path(config, config["paths"]["raw"])
    calibration = project_path(config, config["paths"]["calibration"])
    tracks = {}
    for clip in config["batch"]["clips"]:
        matches = [p for p in raw.iterdir() if p.stem == clip and p.suffix.lower() in (".mov", ".mp4", ".avi", ".mkv")]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one original video for {clip} in {raw}.")
        video = matches[0]
        events = project_path(config, config["paths"]["annotations"]) / f"{clip}.json"
        trajectory = project_path(config, config["paths"]["processed"]) / f"{clip}.json"
        if not trajectory.exists():
            if skip_prepare:
                raise FileNotFoundError(f"Missing measured track: {trajectory}; omit --skip-prepare to extract it.")
            settings = config["video"]
            print(f"Preparing {clip}", flush=True)
            extract_video(video, calibration, events, trajectory,
                table_marker_size_m=settings["table_marker_size_m"], object_marker_size_m=settings["object_marker_size_m"],
                sample_hz=settings["sample_hz"], max_gap_s=settings["max_gap_s"],
                tracking_mode=settings.get("tracking_mode", "multi_marker"), reference_markers=settings.get("reference_markers"),
                object_marker_id=settings.get("object_marker_id", 1), write_preview=settings.get("write_previews", False),
                resting_height_range_m=settings.get("resting_height_range_m", [-.01, .15]),
                min_valid_fraction=settings.get("min_valid_fraction", .90),
                max_pose_reprojection_error_px=settings.get("max_pose_reprojection_error_px", 3.),
                max_table_drift_px=settings.get("max_table_drift_px", 10.))
        validate_track(config, clip, trajectory, video, events, calibration)
        tracks[clip] = trajectory
    return tracks


def build_dataset(config, skip_prepare=False):
    """Build the planned simulator replays from the accepted phone trajectories.

    Replays with different seeds reuse the same recordings; they are additional
    robot episodes, not additional human demonstrations. Keep failed attempts in
    rejected_replays, and export successful episodes only after coverage checks.
    """
    tracks = prepare_tracks(config, skip_prepare)
    settings = config["demonstrations"]
    repetitions, start = int(settings["replays_per_clip"]), int(settings["seed_start"])
    if repetitions < 1 or len(tracks) != len(config["batch"]["clips"]):
        raise ValueError("Use unique batch clips and a positive replays_per_clip.")
    destination = project_path(config, config["paths"]["robot"])
    dataset = project_path(config, config["paths"]["dataset"])
    build = destination / "build"
    rejected = build / "rejected_replays"
    jobs = [{"clip": clip, "seed": start + index * repetitions + offset,
             "trajectory_sha256": digest(trajectory)}
            for index, (clip, trajectory) in enumerate(tracks.items()) for offset in range(repetitions)]
    original = []
    if settings.get("include_original"):
        source = project_path(config, settings["include_original"])
        originals = sorted(source.glob("*.npz"))
        if not originals:
            raise ValueError("Build the main dataset first; no original successful replays were found.")
        for path in originals:
            arrays, metadata = _validate_episode(path, Path(config["_root"]))
            del arrays
            original.append({"name": path.name, "sha256": digest(path), "metadata_sha256": digest(path.with_suffix(".json")),
                             "seed": metadata["seed"], "clip": Path(metadata["source"]["video"]).stem})
        coverage = {clip: sum(ep["clip"] == clip for ep in original) for clip in tracks}
        if any(value != 4 for value in coverage.values()) or len(original) != 4 * len(tracks):
            raise ValueError(f"Expected the main four successful replays per clip: {coverage}")
    seeds = [job["seed"] for job in jobs] + [ep["seed"] for ep in original]
    if len(seeds) != len(set(seeds)):
        raise ValueError("New and original replay seeds overlap.")
    plan = {"task": config["task"], "simulation": config["simulation"], "demonstrations": settings,
            "new_replays": jobs, "original": original}
    plan_path = build / "plan.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError(f"Replay inputs changed. Preserve {destination} and choose new robot and dataset paths.")
    expected = {f"{job['clip']}_seed_{job['seed']}.npz" for job in jobs} | {ep["name"] for ep in original}
    unexpected = [p.name for p in destination.glob("*.npz") if p.name not in expected]
    if unexpected:
        raise ValueError(f"Unexpected episodes in robot folder: {unexpected}")
    if not plan_path.exists():
        write_json(plan_path, plan)
    for entry in original:
        for suffix in (".npz", ".json"):
            src = (source / entry["name"]).with_suffix(suffix)
            dst = (destination / entry["name"]).with_suffix(suffix)
            if dst.exists() and digest(dst) != digest(src):
                raise ValueError(f"Existing original replay copy differs: {dst}")
            if not dst.exists():
                shutil.copy2(src, dst)
    results = []
    for job in jobs:
        output = destination / f"{job['clip']}_seed_{job['seed']}.npz"
        failed = rejected / output.name
        signature = hashlib.sha256(json.dumps({**job, "simulation": config["simulation"], "task": config["task"]}, sort_keys=True).encode()).hexdigest()
        saved = output if output.with_suffix(".json").exists() else failed
        if saved.with_suffix(".json").exists():
            report = read_json(saved.with_suffix(".json"))
            if report.get("build_signature") != signature:
                raise ValueError(f"Existing replay has different or unverified inputs: {saved}")
        else:
            if output.exists() or failed.exists():
                raise ValueError(f"Incomplete replay retained at {output} or {failed}; inspect before rebuilding.")
            report = record_demo(config, output, trajectory_path=tracks[job["clip"]], seed=job["seed"])
            report["build_signature"] = signature
            write_json(output.with_suffix(".json"), report)
        if report.get("success") is not True:
            rejected.mkdir(parents=True, exist_ok=True)
            for suffix in (".npz", ".json", ".mp4"):
                src, dst = output.with_suffix(suffix), failed.with_suffix(suffix)
                if src.exists():
                    if dst.exists():
                        raise ValueError(f"Both successful-folder and rejected copies exist: {dst}")
                    src.rename(dst)
        elif not output.is_file():
            raise ValueError(f"Success report has no robot episode: {output}")
        results.append({**job, **{key: report.get(key) for key in ("success", "frames", "failure")}})
        write_json(build / "replays.json", results)
        print(f"Replay {len(results)}/{len(jobs)}: {json.dumps(results[-1])}", flush=True)
    counts = {clip: sum(ep["success"] is True and ep["clip"] == clip for ep in results) for clip in tracks}
    summary = {"new_successes_per_clip": counts, "new_successes": sum(counts.values()), "original_successes": len(original),
               "source_recordings": len(tracks), "failed_replays": str(rejected.relative_to(Path(config["_root"]))) if rejected.is_relative_to(Path(config["_root"])) else str(rejected)}
    write_json(build / "summary.json", summary)
    if sum(counts.values()) < settings["minimum_successes"] or min(counts.values()) < settings["minimum_successes_per_clip"]:
        raise ValueError(f"Replay coverage gate failed: {counts}. Failed replays retained; no dataset exported.")
    sidecar = dataset / "meta/phone_to_panda_provenance.json"
    if dataset.exists():
        if not sidecar.is_file():
            raise ValueError(f"Incomplete or unrelated dataset retained at {dataset}; choose a new output path.")
        exported = read_json(sidecar)
        current = []
        for path in sorted(destination.glob("*.npz")):
            arrays, metadata = _validate_episode(path, Path(config["_root"]))
            current.append({"episode_index": len(current), "frames": len(arrays["action"]), **metadata})
        if exported.get("provenance") != current:
            raise ValueError("Existing dataset provenance differs from the replay collection; choose a new dataset path.")
    else:
        exported = export_dataset(config)
    if exported["episodes"] != len(original) + sum(counts.values()):
        raise ValueError("Exported episode count differs from validated replays.")
    summary.update(dataset=str(dataset), exported_episodes=exported["episodes"])
    write_json(build / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/main.yaml", help="Project-relative YAML configuration")
    parser.add_argument("--skip-prepare", action="store_true", help="Require existing measured tracks; still verify their input hashes and quality")
    args = parser.parse_args(argv)
    try:
        result = build_dataset(load_config(ROOT / args.config), args.skip_prepare)
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Dataset build: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
