"""Check four physics replays of one IMU take, then export a separate dataset.

The original 20 robot episodes are copied unchanged. Four additional episodes
reuse one human recording at different simulated starts; they are not four new
human demonstrations. Any failed replay remains available for inspection.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
from phone_to_panda.common import load_config, project_path
from phone_to_panda.imu_pilot import digest
from phone_to_panda.learning import _validate_episode, export_dataset
from phone_to_panda.simulation import record_demo


def build(config_path, trajectory_path):
    config = load_config(config_path)
    trajectory_path = Path(trajectory_path).resolve()
    trajectory = json.loads(trajectory_path.read_text())
    results = ROOT / "results/imu_pilot_20261015"
    plan = json.loads((results / "plan.json").read_text())
    replays = ROOT / "runs/imu_pilot_20261015/replays"
    destination = project_path(config, config["paths"]["robot"])
    dataset = project_path(config, config["paths"]["dataset"])
    if destination.exists() or dataset.exists():
        raise FileExistsError("Keep the existing pilot collection; choose new destinations before rebuilding.")
    reports = []
    for seed in plan["replay_seeds"]:
        output = replays / f"IMG2276_seed{seed}.npz"
        sidecar = output.with_suffix(".json")
        if sidecar.exists():
            report = json.loads(sidecar.read_text())
            if (report.get("seed") != seed or report.get("source") != trajectory["source"]
                    or Path(report.get("trajectory", "")).resolve() != trajectory_path
                    or report.get("trajectory_quality") != trajectory["quality"]
                    or report.get("simulation_settings") != config["simulation"]):
                raise ValueError("Existing replay differs from the fixed pilot inputs.")
        else:
            report = record_demo(config, output, trajectory_path=trajectory_path, seed=seed)
        reports.append({"seed": seed, "success": report.get("success"), "frames": report.get("frames"),
                        "failure": report.get("failure"), "report": str(sidecar.relative_to(ROOT))})
        (results / "replays.json").write_text(json.dumps(reports, indent=2) + "\n")
        print(json.dumps(reports[-1]), flush=True)
    if len(reports) != 4 or any(r["success"] is not True for r in reports):
        raise ValueError("The fixed four-success replay gate failed. Failed outputs retained; no training dataset exported.")
    originals = sorted((ROOT / "data/robot/main").glob("*.npz"))
    if len(originals) != 20:
        raise ValueError("Expected the original 20 successful robot demonstrations.")
    additions = [replays / f"IMG2276_seed{seed}.npz" for seed in plan["replay_seeds"]]
    manifests = []
    # Validate everything before creating the combined collection.
    for path in originals + additions:
        arrays, metadata = _validate_episode(path, ROOT)
        del arrays
        manifests.append({"file": path.name, "npz_sha256": digest(path),
                          "metadata_sha256": digest(path.with_suffix(".json")),
                          "source_kind": metadata["source"]["kind"], "seed": metadata["seed"]})
    destination.mkdir(parents=True)
    for path in originals + additions:
        for suffix in (".npz", ".json"):
            source = path.with_suffix(suffix)
            shutil.copy2(source, destination / source.name)
    (results / "dataset_inputs.json").write_text(json.dumps(manifests, indent=2) + "\n")
    summary = export_dataset(config)
    (results / "dataset_summary.json").write_text(json.dumps({
        "episodes": summary["episodes"], "frames": summary["frames"],
        "original_video_demonstrations": 20, "imu_assisted_demonstrations": 4,
        "original_human_recordings": 5, "new_paired_human_recordings": 1,
        "dataset": str(dataset.relative_to(ROOT)), "uploaded": False,
    }, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/imu_pilot_20261015/replay.yaml")
    parser.add_argument("--trajectory", default="data/experimental/imu_20261015/trajectory.json")
    args = parser.parse_args()
    result = build(ROOT / args.config, ROOT / args.trajectory)
    print(json.dumps({"episodes": result["episodes"], "frames": result["frames"]}), flush=True)
