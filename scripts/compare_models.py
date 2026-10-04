"""Compare saved main and expanded policies on identical simulator starts.

The defaults reproduce 10 narrow starts (seeds 3000-3009) and 10 wider starts
(3010-3019). These seeds have already been evaluated in the saved experiment.
Use new seed ranges for a new test. No dataset generation or training is run.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from phone_to_panda.common import load_config, project_path
from phone_to_panda.learning import run_policy


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--main-checkpoint", default="checkpoints/main")
    parser.add_argument("--expanded-checkpoint", default="checkpoints/expanded")
    parser.add_argument("--output", default="runs/evaluation_compare")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per start range and per model")
    parser.add_argument("--narrow-seed", type=int, default=3000)
    parser.add_argument("--wide-seed", type=int, default=3010)
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if set(range(args.narrow_seed, args.narrow_seed + args.episodes)) & set(range(args.wide_seed, args.wide_seed + args.episodes)):
        parser.error("Narrow and wide seed ranges must be disjoint")
    try:
        config = load_config(ROOT / args.config)
        checkpoints = {"main": project_path(config, args.main_checkpoint), "expanded": project_path(config, args.expanded_checkpoint)}
        output = project_path(config, args.output)
        groups = [{"name": "narrow", "seed": args.narrow_seed, "radius_m": .035},
                  {"name": "wide", "seed": args.wide_seed, "radius_m": .060}]
        plan = {"config": {k: v for k, v in config.items() if k != "_root"}, "groups": groups, "episodes_per_group": args.episodes,
                "checkpoints": {label: {file.relative_to(path).as_posix(): digest(file) for file in sorted(path.rglob("*")) if file.is_file()}
                                for label, path in checkpoints.items()}}
        for label, files in plan["checkpoints"].items():
            if not {"model.safetensors", "config.json", "train_config.json", "phone_to_panda_provenance.json"}.issubset(files):
                raise ValueError(f"Missing checkpoint files for {label}: {checkpoints[label]}")
        frozen = output / "plan.json"
        if frozen.exists():
            if json.loads(frozen.read_text()) != plan:
                raise ValueError(f"Comparison inputs changed. Preserve {output} and choose a new --output.")
        elif output.exists():
            raise ValueError(f"Existing results have no matching plan: {output}. Choose a new --output.")
        else:
            write_json(frozen, plan)
        reports = {}
        for label, checkpoint in checkpoints.items():
            reports[label] = {}
            for group in groups:
                destination = output / label / group["name"]
                if (destination / "report.json").is_file():
                    report = json.loads((destination / "report.json").read_text())
                else:
                    if destination.exists():
                        raise ValueError(f"Incomplete evaluation retained at {destination}; inspect before rerunning.")
                    evaluation = copy.deepcopy(config)
                    evaluation["simulation"]["randomize_start_m"] = group["radius_m"]
                    evaluation["evaluation"].update(save_video=False, save_trace=True)
                    print(f"Evaluating {label}: {group['name']} starts", flush=True)
                    report = run_policy(evaluation, checkpoint, destination, episodes=args.episodes, seed=group["seed"])
                expected_seeds = list(range(group["seed"], group["seed"] + args.episodes))
                if (Path(report.get("checkpoint", "")).resolve() != checkpoint.resolve()
                        or report.get("seed_start") != group["seed"]
                        or [ep["seed"] for ep in report.get("episodes", [])] != expected_seeds
                        or report.get("randomize_start_m") != group["radius_m"]):
                    raise ValueError(f"Cached evaluation identity does not match the comparison plan: {destination}")
                reports[label][group["name"]] = report
        # Every trace starts before the first policy action. Paired policies must
        # therefore see the same cube and robot state for each group and seed.
        for group in groups:
            for index in range(args.episodes):
                traces = [json.loads((output / label / group["name"] / f"episode_{index:03d}_trace.json").read_text()) for label in checkpoints]
                if not all(traces) or any(traces[0][0][key] != traces[1][0][key] for key in ("cube_xyz", "observation_state")):
                    raise ValueError(f"Paired initial cube or robot states differ: {group['name']}, episode {index}")
        totals = {label: sum(report["successes"] for report in groups.values()) for label, groups in reports.items()}
        summary = {"status": "completed", "successes": totals, "episodes_per_model": args.episodes * 2, "paired_initial_cube_and_robot_states_verified": True, "reports": reports,
                   "interpretation": "Paired simulation starts in the same task. Expanded training changes both replay data and update count; this is not a fixed-budget data ablation or real-world test."}
        write_json(output / "summary.json", summary)
        print(json.dumps({k: v for k, v in summary.items() if k != "reports"}, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Model comparison: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
