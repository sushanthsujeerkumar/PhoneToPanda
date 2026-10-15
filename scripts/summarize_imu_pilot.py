"""Compare all three fixed pilot models on the same ten simulator starts.

This reports the planned trials, including failures. It does not select a model,
estimate sensor accuracy, or turn one human recording into independent sessions.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_candidate import validate_report

STUDY = ROOT / "results/imu_pilot_20261015"
MODELS = ("original", "continuation", "augmented")
LABELS = ("Original model", "Continued training", "IMU-assisted augmentation")
GROUPS = ("narrow", "wide")


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def collect():
    plan = read(STUDY / "plan.json")
    models = {}
    reference_starts = {}
    reference_implementation = None
    reference_simulation = None
    for model in MODELS:
        model_folder = STUDY / "evaluation" / model
        identity = read(model_folder / "identity.json")
        completed = read(model_folder / "summary.json")
        if (identity["model"] != model or completed["model"] != model
                or identity["plan_sha256"] != digest(STUDY / "plan.json")
                or completed["identity_sha256"] != digest(model_folder / "identity.json")):
            raise ValueError(f"Evaluation identity changed: {model}")
        implementation = (identity["source_sha256"], identity["scene_assets_sha256"])
        if reference_implementation is None:
            reference_implementation = implementation
        elif implementation != reference_implementation:
            raise ValueError("Models were evaluated with different code or scene assets")
        groups = {}
        for group in GROUPS:
            folder = STUDY / "evaluation" / model / group
            manifest = read(folder / "manifest.json")
            expected = plan["evaluation"][group]
            if (manifest["identity_sha256"] != digest(model_folder / "identity.json")
                    or manifest["group"] != group or manifest["episode_plan"] != expected):
                raise ValueError(f"Group identity differs from the fixed plan: {model}/{group}")
            simulation = {k: v for k, v in manifest["config"]["simulation"].items()
                          if k != "randomize_start_m"}
            if reference_simulation is None:
                reference_simulation = simulation
            elif simulation != reference_simulation:
                raise ValueError("Models were evaluated with different simulation settings")
            reports = list(folder.rglob("report.json"))
            if len(reports) != 1:
                raise ValueError(f"Expected one completed report for {model}/{group}")
            actual_hashes = {str(p.relative_to(reports[0].parent)): digest(p)
                             for p in reports[0].parent.rglob("*") if p.is_file()}
            if read(folder / "outputs.json") != actual_hashes:
                raise ValueError(f"Retained evaluation outputs changed: {model}/{group}")
            report = read(reports[0])
            if (completed["groups"][group]["report_sha256"] != digest(reports[0])
                    or completed["groups"][group]["report"] != str(reports[0].relative_to(ROOT))):
                raise ValueError(f"Report differs from the completed model receipt: {model}/{group}")
            validate_report(report, checkpoint=Path(identity["checkpoint"]),
                            action_steps=plan["evaluation"]["n_action_steps"],
                            settings=expected, config=manifest["config"])
            episodes = report["episodes"]
            seeds = list(range(expected["seed"], expected["seed"] + expected["episodes"]))
            if ([e["seed"] for e in episodes] != seeds
                    or report["randomize_start_m"] != expected["radius_m"]
                    or report["policy_settings"]["n_action_steps"] != plan["evaluation"]["n_action_steps"]
                    or report["orientation_control"] != "fixed_downward"
                    or any(type(e["success"]) is not bool for e in episodes)):
                raise ValueError(f"Report differs from the fixed pilot plan: {model}/{group}")
            rows = []
            for index, episode in enumerate(episodes):
                trace = read(reports[0].parent / f"episode_{index:03d}_trace.json")
                first = trace[0]
                start = (first["cube_xyz"], first["observation_state"])
                key = (group, episode["seed"])
                if model == "original":
                    reference_starts[key] = start
                elif start != reference_starts[key]:
                    raise ValueError(f"Initial cube/robot state differs: {model}/{key}")
                events = episode["first_events"]
                lifted = events["lift_step"] is not None
                rows.append({"seed": episode["seed"], "success": episode["success"],
                             "lifted": lifted, "steps": episode["steps"],
                             "failure": None if episode["success"] else
                             "after_lift" if lifted else "before_lift"})
            successes = sum(row["success"] for row in rows)
            if successes != report["successes"]:
                raise ValueError("Report success count differs from its episode results")
            groups[group] = {"successes": successes, "trials": len(rows), "episodes": rows,
                             "report": str(reports[0].relative_to(ROOT))}
        models[model] = {"groups": groups,
                         "successes": sum(g["successes"] for g in groups.values()),
                         "trials": sum(g["trials"] for g in groups.values())}
    paired = {}
    for model in ("original", "continuation"):
        pairs = [(a, b) for group in GROUPS
                 for a, b in zip(models[model]["groups"][group]["episodes"],
                                 models["augmented"]["groups"][group]["episodes"], strict=True)]
        paired[f"augmented_vs_{model}"] = {
            "improved": sum(not a["success"] and b["success"] for a, b in pairs),
            "regressed": sum(a["success"] and not b["success"] for a, b in pairs),
            "both_successful": sum(a["success"] and b["success"] for a, b in pairs),
            "both_failed": sum(not a["success"] and not b["success"] for a, b in pairs),
            "identical_initial_cube_and_robot_states": True,
        }
    return {"kind": "imu_assisted_demonstration_pilot", "models": models, "paired": paired,
            "default_checkpoint_changed": False,
            "limits": ["One new human recording; four robot replays reuse its path.",
                       "Ten new simulator starts per model, one scene and one training seed.",
                       "Timing is provisional; shoulder is fixed; physical tracking accuracy is unmeasured.",
                       "The policy uses simulator images and robot state, not IMUs.",
                       "This tests demonstration augmentation, not vision/IMU sensor fusion.",
                       "Every planned result is reported; no model is selected using these starts."]}


def plot(summary):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), gridspec_kw={"height_ratios": [1, 1.2]})
    colors = ("#64748b", "#2563eb", "#0d9488")
    x = np.arange(2)
    for index, model in enumerate(MODELS):
        values = [summary["models"][model]["groups"][g]["successes"] for g in GROUPS]
        bars = axes[0].bar(x + (index - 1) * .24, values, .22, color=colors[index], label=LABELS[index])
        axes[0].bar_label(bars, labels=[f"{v}/5" for v in values], padding=3)
    axes[0].set_xticks(x, ["Narrow starts (±3.5 cm each axis)", "Wide starts (±7 cm each axis)"])
    axes[0].set_ylim(0, 6.2)
    axes[0].set_yticks(range(6))
    axes[0].set_ylabel("Successful placements")
    axes[0].legend(loc="upper center", ncols=3, fontsize=8, frameon=False)
    axes[0].spines[["top", "right"]].set_visible(False)
    outcomes = np.array([[row["success"] for group in GROUPS
                          for row in summary["models"][model]["groups"][group]["episodes"]]
                         for model in MODELS], dtype=int)
    from matplotlib.colors import ListedColormap
    axes[1].imshow(outcomes, cmap=ListedColormap(["#fce7e7", "#c5eee5"]), vmin=0, vmax=1, aspect="auto")
    for row in range(3):
        for col in range(10):
            axes[1].text(col, row, "Placed" if outcomes[row, col] else "Failed", ha="center", va="center", fontsize=9)
    axes[1].set_yticks(range(3), LABELS)
    axes[1].set_xticks(range(10), [f"N{i+1}" for i in range(5)] + [f"W{i+1}" for i in range(5)])
    axes[1].axvline(4.5, color="white", linewidth=3)
    axes[1].set_xlabel("Same initial cube and robot states in each column")
    fig.suptitle("One-recording IMU-assisted training pilot", fontsize=16, fontweight="bold")
    fig.text(.5, .025, "Small same-scene test • one training seed • demonstration augmentation, not sensor fusion",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .95))
    fig.savefig(STUDY / "comparison.png", dpi=160)
    plt.close(fig)


def write(summary):
    lines = ["# IMU-assisted demonstration pilot", "",
             "Four controller replays succeeded. The learned policies were then tested separately on ten matching new simulator starts.", "",
             "| Model | Narrow | Wide | Total |", "| --- | ---: | ---: | ---: |"]
    for model, label in zip(MODELS, LABELS, strict=True):
        result = summary["models"][model]
        values = [f"{result['groups'][g]['successes']}/5" for g in GROUPS]
        lines.append(f"| {label} | {' | '.join(values)} | {result['successes']}/10 |")
    lines += ["", "![All planned outcomes](comparison.png)", "",
              "The original checkpoint remains the default. These small pilot counts do not establish general improvement.", ""]
    for key, result in summary["paired"].items():
        reference = key.removeprefix("augmented_vs_")
        lines.append(f"Compared with {reference}, augmentation fixes {result['improved']} failed starts and regresses on {result['regressed']} successful starts.")
    lines += ["", *[f"- {limit}" for limit in summary["limits"]], "",
              "[Method and reproduction](../../docs/imu_pilot_20261015.md) · [Raw comparison](summary.json)", ""]
    (STUDY / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (STUDY / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    summary = collect()
    plot(summary)
    write(summary)
    print(json.dumps({name: value["successes"] for name, value in summary["models"].items()}))
