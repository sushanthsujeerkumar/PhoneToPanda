"""Summarize explicitly selected completed study runs without mixing phases.

The first --names entry is the reference model. Development is used for diagnosis
and candidate selection; only final_test is the frozen test of locked candidates.
No results are scanned outside the requested phase, candidates and groups.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIFT_HEIGHT_M = .97  # This scene's table Z (.90 m) plus the .07 m lift criterion.


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wilson_interval(successes, trials, z=1.959963984540054):
    """Two-sided 95% Wilson binomial interval, including zero/all-success cases."""
    if (isinstance(successes, bool) or isinstance(trials, bool)
            or not isinstance(successes, int) or not isinstance(trials, int)
            or trials <= 0 or not 0 <= successes <= trials):
        raise ValueError("Wilson interval requires integer 0 <= successes <= positive trials")
    rate = successes / trials
    denominator = 1 + z * z / trials
    centre = (rate + z * z / (2 * trials)) / denominator
    width = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / denominator
    return [max(0., centre - width), min(1., centre + width)]


def exact_mcnemar(improved, regressed):
    """Exact two-sided binomial test of discordant paired placement outcomes."""
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in (improved, regressed)):
        raise ValueError("Discordant counts must be nonnegative integers")
    total = improved + regressed
    if not total:
        return 1.
    return min(1., 2 * sum(math.comb(total, k) for k in range(min(improved, regressed) + 1)) / (2 ** total))


def summarize_trials(trials, include_intervals=True):
    if not trials:
        raise ValueError("Cannot summarize an empty or incomplete group")
    count = len(trials)
    placed = sum(trial["success"] for trial in trials)
    lifted = sum(trial["lifted"] for trial in trials)
    result = {
        "trials": count, "placements": placed, "lifts": lifted,
        "failed_lift": count - lifted, "failed_after_lift": lifted - placed,
        "placement_rate": placed / count, "lift_rate": lifted / count,
    }
    if include_intervals:
        result.update(placement_wilson95=wilson_interval(placed, count),
                      lift_wilson95=wilson_interval(lifted, count))
    return result


def paired_counts(reference, candidate):
    if not reference or len(reference) != len(candidate):
        raise ValueError("Paired groups must be complete and have the same number of trials")
    improved = regressed = both_success = both_failure = 0
    for before, after in zip(reference, candidate):
        if before["seed"] != after["seed"] or before["group"] != after["group"]:
            raise ValueError("Paired seed/group mismatch")
        for key in ("initial_cube_xyz", "initial_observation_state"):
            if before[key] != after[key]:
                raise ValueError(f"Paired initial state mismatch for seed {before['seed']}: {key}")
        improved += not before["success"] and after["success"]
        regressed += before["success"] and not after["success"]
        both_success += before["success"] and after["success"]
        both_failure += not before["success"] and not after["success"]
    return {
        "trials": len(reference), "improved": improved, "regressed": regressed,
        "both_success": both_success, "both_failure": both_failure,
        "placement_rate_difference": (improved - regressed) / len(reference),
        "exact_mcnemar_two_sided_p": exact_mcnemar(improved, regressed),
        "identical_initial_cube_and_robot_states": True,
    }


def finite_vector(value, size, label):
    if (not isinstance(value, list) or len(value) != size
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value)):
        raise ValueError(f"Invalid {label}: expected {size} finite numbers")
    return value


def load_group(study, phase, name, group, settings, plan_hash, locked_identity=None, locked_context=None,
               lock_hash=None):
    folder = study / phase / name / group
    manifest = read_json(folder / "manifest.json")
    if any(manifest.get(key) != value for key, value in {
        "candidate": name, "phase": phase, "group": group,
        "episode_plan": settings, "study_plan_sha256": plan_hash,
    }.items()):
        raise ValueError(f"Manifest differs from frozen plan: {name}/{group}")
    identity = manifest["identity"]
    if locked_identity is not None and identity != locked_identity:
        raise ValueError(f"Final candidate differs from selection lock: {name}/{group}")
    if locked_context is not None:
        if not lock_hash or manifest.get("candidate_lock_sha256") != lock_hash:
            raise ValueError(f"Final candidate selection lock changed or is missing: {name}/{group}")
        if (not locked_context.get("config_sha256")
                or manifest.get("base_config_sha256") != locked_context["config_sha256"]):
            raise ValueError(f"Final base configuration differs from selection lock: {name}/{group}")
        if (not locked_context.get("source_sha256")
                or manifest.get("source_sha256") != locked_context["source_sha256"]):
            raise ValueError(f"Final source code differs from selection lock: {name}/{group}")
    report_path = folder / "evaluation/report.json"
    report = read_json(report_path)
    expected_seeds = list(range(settings["seed"], settings["seed"] + settings["episodes"]))
    episodes = report.get("episodes", [])
    if ([episode.get("seed") for episode in episodes] != expected_seeds
            or report.get("seed_start") != settings["seed"]):
        raise ValueError(f"Incomplete or mismatched seed group: {name}/{group}")
    if report.get("kind") != "trained_policy_physics_evaluation" or report.get("randomize_start_m") != settings["radius_m"]:
        raise ValueError(f"Wrong evaluation kind or start range: {name}/{group}")
    policy_settings = report.get("policy_settings", {})
    if policy_settings.get("n_action_steps") != identity["n_action_steps"]:
        raise ValueError(f"Reported action horizon differs from manifest: {name}/{group}")
    simulation = manifest["config"]["simulation"]
    if report.get("orientation_control") != "fixed_downward" or simulation.get("orientation_control") != "fixed_downward":
        raise ValueError("This study summary requires the recorded fixed-downward hybrid controller")
    trials = []
    for index, episode in enumerate(episodes):
        if not isinstance(episode.get("success"), bool):
            raise ValueError(f"Invalid placement outcome: {name}/{group}/{index}")
        height = episode.get("max_cube_height_m")
        if isinstance(height, bool) or not isinstance(height, (int, float)) or not math.isfinite(height):
            raise ValueError(f"Missing finite cube height: {name}/{group}/{index}")
        lifted = height > LIFT_HEIGHT_M
        if episode["success"] and not lifted:
            raise ValueError(f"Placement reported without required lift: {name}/{group}/{index}")
        trace = read_json(folder / "evaluation" / f"episode_{index:03d}_trace.json")
        if not trace or len(trace) != episode.get("steps") or [entry.get("step") for entry in trace] != list(range(len(trace))):
            raise ValueError(f"Incomplete action trace: {name}/{group}/{index}")
        if any(entry.get("episode_seed") != episode["seed"] for entry in trace):
            raise ValueError(f"Trace seed differs from report: {name}/{group}/{index}")
        if trace[-1].get("success") != episode["success"]:
            raise ValueError(f"Trace final outcome differs from report: {name}/{group}/{index}")
        trials.append({
            "seed": episode["seed"], "group": group, "success": episode["success"], "lifted": lifted,
            "steps": episode["steps"], "max_cube_height_m": height,
            "initial_cube_xyz": finite_vector(trace[0].get("cube_xyz"), 3, "initial cube pose"),
            "initial_observation_state": finite_vector(trace[0].get("observation_state"), 8, "initial robot state"),
        })
    counts = summarize_trials(trials)
    if (report.get("successes") != counts["placements"]
            or not math.isclose(report.get("success_rate", -1), counts["placement_rate"], abs_tol=1e-12)):
        raise ValueError(f"Reported success aggregate differs from episodes: {name}/{group}")
    return {
        **counts, "episode_results": trials, "report": str(report_path.relative_to(study)),
        "report_sha256": digest(report_path), "identity": identity,
        "policy_settings": policy_settings, "wall_seconds": report.get("wall_seconds"),
        "comparison_contract": {key: simulation.get(key) for key in ("control_hz", "horizon", "image_size", "orientation_control")},
    }


def build_summary(study, phase, names, groups=None):
    study = Path(study).resolve()
    if phase not in ("development", "final_test"):
        raise ValueError("Choose development or final_test explicitly")
    if not names or len(set(names)) != len(names) or any(not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name) for name in names):
        raise ValueError("Candidate names must be unique simple names")
    plan_file = study / "study_plan.json"
    plan, plan_hash = read_json(plan_file), digest(plan_file)
    phase_plan = plan[phase]
    groups = list(phase_plan) if groups is None else list(groups)
    if not groups or len(set(groups)) != len(groups) or any(group not in phase_plan for group in groups):
        raise ValueError("Groups must be unique names from the selected phase")
    lock = lock_hash = None
    locked_selection = {}
    if phase == "final_test":
        if len(names) < 2 or set(groups) != set(phase_plan):
            raise ValueError("Final summary requires baseline and candidate plus every frozen final group")
        lock_path = study / "candidate_lock.json"
        lock_bytes = lock_path.read_bytes()
        lock = json.loads(lock_bytes)
        lock_hash = hashlib.sha256(lock_bytes).hexdigest()
        if lock.get("study_plan_sha256") != plan_hash or any(name not in lock.get("candidates", {}) for name in names):
            raise ValueError("Missing candidate lock or changed study plan")
        reference, selected = lock.get("reference_candidate"), lock.get("selected_candidate")
        candidates, roles = lock["candidates"], lock.get("contrast_roles")
        if (reference not in candidates or selected not in candidates or not isinstance(roles, dict)
                or not (set(candidates) - {reference}) <= set(roles)
                or not set(roles) <= set(candidates)
                or any(not isinstance(role, str) or not role.strip() for role in roles.values())):
            raise ValueError("Invalid reference, selection or contrast roles in final selection lock")
        if names[0] != reference:
            raise ValueError("Final summary must put the locked reference_candidate first")
        if not isinstance(lock.get("selection_sidecars_sha256"), dict) or not lock["selection_sidecars_sha256"]:
            raise ValueError("Missing selection sidecars in final selection lock")
        omitted = sorted(set(candidates) - set(names))
        locked_selection = {
            "candidate_lock_sha256": lock_hash,
            "reference_candidate": reference, "selected_candidate": selected,
            "contrast_roles": roles, "selection_sidecars_sha256": lock["selection_sidecars_sha256"],
            "report_scope": "secondary_subset" if omitted else "all_locked_candidates",
            "omitted_locked_candidates": omitted,
            "selected_candidate_included": selected in names,
        }
    models = {}
    for name in names:
        model_groups = {group: load_group(study, phase, name, group, phase_plan[group], plan_hash,
                                        lock["candidates"][name] if lock else None, lock, lock_hash) for group in groups}
        identities = [value["identity"] for value in model_groups.values()]
        if any(identity != identities[0] for identity in identities[1:]):
            raise ValueError(f"Model/settings changed across groups for {name}")
        combined = [trial for group in groups for trial in model_groups[group]["episode_results"]]
        if len({trial["seed"] for trial in combined}) != len(combined):
            raise ValueError("Selected groups overlap seeds; refusing a misleading combined count")
        combined_counts = summarize_trials(combined, include_intervals=False)
        combined_counts["interval_note"] = "No pooled binomial interval: the combined result uses fixed per-group start quotas."
        models[name] = {"groups": model_groups, "combined": combined_counts}
    paired = {}
    reference = models[names[0]]["groups"]
    for name in names[1:]:
        candidate = models[name]["groups"]
        for group in groups:
            if reference[group]["comparison_contract"] != candidate[group]["comparison_contract"]:
                raise ValueError(f"Controller/control rate/horizon/image settings differ: {name}/{group}")
        comparisons = {group: paired_counts(reference[group]["episode_results"], candidate[group]["episode_results"]) for group in groups}
        paired[name] = {
            "reference": names[0], "groups": comparisons,
            "combined": paired_counts([t for group in groups for t in reference[group]["episode_results"]],
                                      [t for group in groups for t in candidate[group]["episode_results"]]),
        }
    total_trials = sum(phase_plan[group]["episodes"] for group in groups)
    group_weights = {group: phase_plan[group]["episodes"] / total_trials for group in groups}
    mixture = ", ".join(f"{group}: {weight:.1%}" for group, weight in group_weights.items())
    return {
        "phase": phase, "names": names, "groups": groups, "reference": names[0],
        "combined_group_weights": group_weights,
        "study_plan_sha256": plan_hash, "source_recording_count": len(plan.get("source_videos", {})),
        "selection_rule": plan.get("selection_rule"), "models": models, "paired": paired,
        **({"locked_selection": locked_selection} if lock else {}),
        "limits": [
            "Development results are diagnostics and selection evidence, not a held-out final test." if phase == "development"
            else "Final results use the frozen seed groups and model/settings identities locked before final testing.",
            "One custom simulation scene, five source recordings, fixed cube and target; no new human recordings or real-robot evidence.",
            "Hybrid control: learned XYZ/gripper with fixed downward wrist feedback.",
            "Lift means cube centre exceeded 0.97 m. Failed after lift combines dropped-object and unsuccessful-placement outcomes; it does not diagnose which occurred.",
            "Wilson intervals describe each start group's marginal binomial proportion; they are not confidence intervals on the paired difference.",
            f"Combined rates use fixed per-group trial quotas ({mixture}). No pooled binomial interval is reported for this fixed stratified design.",
            "Intervals and exact paired p values compare these fixed checkpoints over simulator starts. One training seed provides no estimate of optimizer-seed or training-run variability.",
            "Development comparisons used for candidate selection are exploratory. Final candidate identities must be locked before final outcomes are observed; final outcomes must not select a new winner.",
            "Repeated starts in this scene do not establish broad task generalization. A selected rollout video illustrates one seed; aggregate counts provide the performance comparison.",
            "Recorded wall times may include concurrent workloads and are not a latency benchmark.",
        ],
    }


def write_markdown(summary, path):
    phase_label = "Development diagnostics" if summary["phase"] == "development" else "Frozen final test"
    selection = summary.get("locked_selection")
    if selection and selection["report_scope"] == "secondary_subset":
        phase_label += " — secondary subset"
    lines = [f"# {phase_label}", "", f"Reference: **{summary['reference']}**. Five source recordings; one custom simulated task.", ""]
    if selection:
        lines += [f"Selected before final testing: **{selection['selected_candidate']}**. This selection is unchanged by final outcomes.", ""]
        if selection["omitted_locked_candidates"]:
            lines += ["Secondary subset; omitted locked candidates: " + ", ".join(selection["omitted_locked_candidates"]) + ".", ""]
        lines += ["| Locked candidate | Comparison role |", "| --- | --- |"]
        lines += [f"| {name} | {role} |" for name, role in selection["contrast_roles"].items()]
        lines += [""]
    lines += ["| Model | Group | Placements | 95% Wilson interval | Lifts | Failed to lift | Lifted, not placed |",
             "| --- | --- | ---: | --- | ---: | ---: | ---: |"]
    for name in summary["names"]:
        model = summary["models"][name]
        for group in [*summary["groups"], "combined"]:
            result = model["combined"] if group == "combined" else model["groups"][group]
            interval = "Not pooled"
            if group != "combined":
                low, high = result["placement_wilson95"]
                interval = f"{low:.1%}–{high:.1%}"
            lines.append(f"| {name} | {group} | {result['placements']}/{result['trials']} | {interval} | {result['lifts']}/{result['trials']} | {result['failed_lift']} | {result['failed_after_lift']} |")
    if summary["paired"]:
        lines += ["", "All compared starts have exactly matching initial cube and robot states.", "",
                  "| Candidate vs reference | Group | Improved | Regressed | Both placed | Both failed | Exact paired p |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
        for name, paired in summary["paired"].items():
            for group in [*summary["groups"], "combined"]:
                result = paired["combined"] if group == "combined" else paired["groups"][group]
                lines.append(f"| {name} | {group} | {result['improved']} | {result['regressed']} | {result['both_success']} | {result['both_failure']} | {result['exact_mcnemar_two_sided_p']:.4g} |")
        lines += ["", "Exact p values use a two-sided McNemar/binomial test of discordant placements; no multiple-comparison adjustment is applied."]
    lines += ["", "![Placement counts and uncertainty](placement_results.png)", "", "![Initial positions and outcomes](start_positions.png)", ""]
    lines += [f"- {limit}" for limit in summary["limits"]]
    path.write_text("\n".join(lines) + "\n")


def plot_summary(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    names, groups = summary["names"], summary["groups"]
    phase_label = "Development diagnostics" if summary["phase"] == "development" else "Frozen final test"
    if summary.get("locked_selection", {}).get("report_scope") == "secondary_subset":
        phase_label += " — secondary subset"
    labels = [*groups, "combined"]
    figure, axes = plt.subplots(figsize=(max(8, 2.1 * len(labels)), 5.4), layout="constrained")
    width = .8 / len(names)
    for index, name in enumerate(names):
        model = summary["models"][name]
        results = [model["groups"][group] for group in groups] + [model["combined"]]
        values = np.array([result["placement_rate"] for result in results]) * 100
        intervals = np.array([result["placement_wilson95"] for result in results[:-1]]) * 100
        x = np.arange(len(labels)) + (index - (len(names) - 1) / 2) * width
        bars = axes.bar(x, values, width, label=name, alpha=.88)
        axes.errorbar(x[:-1], values[:-1], yerr=np.maximum(0., np.stack((values[:-1] - intervals[:, 0], intervals[:, 1] - values[:-1]))), fmt="none", ecolor="#263238", capsize=3, linewidth=1)
        label_heights = [*intervals[:, 1], values[-1]]
        for bar, result, height in zip(bars, results, label_heights):
            axes.annotate(f"{result['placements']}/{result['trials']}", (bar.get_x() + bar.get_width() / 2, height + 1.5), ha="center", va="bottom", fontsize=9)
    axes.set_xticks(range(len(labels)), [label.replace("_", " ") for label in labels])
    axes.set_ylim(0, 116)
    axes.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    axes.set_ylabel("Placement success")
    axes.set_title(phase_label + " · placements / trials\nWilson 95% intervals by group; combined count has no pooled interval", pad=14)
    axes.legend(loc="upper left", bbox_to_anchor=(0, -.12), ncol=min(2, len(names)), frameon=False)
    axes.grid(axis="y", alpha=.2)
    axes.set_axisbelow(True)
    figure.savefig(output / "placement_results.png", dpi=180, facecolor="white")
    plt.close(figure)
    figure, axes = plt.subplots(len(groups), len(names), squeeze=False,
                                figsize=(max(5, 4.0 * len(names)), max(4, 3.8 * len(groups))), layout="constrained")
    statuses = [("Placed", "#087e72", "o"), ("Failed to lift", "#c73538", "x"), ("Lifted, not placed", "#bb7900", "^")]
    for row, group in enumerate(groups):
        starts = summary["models"][names[0]]["groups"][group]["episode_results"]
        xy = np.array([trial["initial_cube_xyz"][:2] for trial in starts]) * 100
        span = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 2.) + 2.
        centre = (xy.min(axis=0) + xy.max(axis=0)) / 2
        for col, name in enumerate(names):
            axis = axes[row, col]
            result = summary["models"][name]["groups"][group]
            trials = result["episode_results"]
            for category, (label, colour, marker) in enumerate(statuses):
                subset = [trial for trial in trials if (0 if trial["success"] else 2 if trial["lifted"] else 1) == category]
                axis.scatter([trial["initial_cube_xyz"][0] * 100 for trial in subset],
                             [trial["initial_cube_xyz"][1] * 100 for trial in subset],
                             color=colour, marker=marker, s=48, label=label, linewidths=1.6, alpha=.9)
            axis.set(xlim=(centre[0] - span / 2, centre[0] + span / 2), ylim=(centre[1] - span / 2, centre[1] + span / 2),
                     xlabel="Initial cube X (cm)", ylabel="Initial cube Y (cm)", aspect="equal")
            axis.set_title(f"{name}\n{group.replace('_', ' ')} · {result['placements']}/{result['trials']} placed")
            axis.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    figure.suptitle(phase_label + "\nSame initial positions, paired policy outcomes", fontsize=14)
    figure.savefig(output / "start_positions.png", dpi=180, facecolor="white")
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="runs/improvement_20261011")
    parser.add_argument("--phase", choices=("development", "final_test"), required=True)
    parser.add_argument("--names", nargs="+", required=True, help="Reference first, followed by candidates")
    parser.add_argument("--groups", nargs="+", help="Development groups only; final requires all frozen groups")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.phase == "final_test" and args.groups is not None:
            raise ValueError("Do not select final groups; summarize every frozen final group")
        summary = build_summary(ROOT / args.study, args.phase, args.names, args.groups)
        output = (ROOT / args.output).resolve()
        output.mkdir(parents=True, exist_ok=False)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        write_markdown(summary, output / "summary.md")
        plot_summary(summary, output)
        print(json.dumps({"phase": args.phase, "output": str(output), "models": {name: data["combined"] for name, data in summary["models"].items()}}, indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"Study summary: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
