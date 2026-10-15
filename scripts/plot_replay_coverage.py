"""Plot only completed selected replays from the 11 October study.

Cube XY is recovered from saved, unsaturated approach state/action pairs using
_Recorder.action_to: target = state + action * 0.05 / 0.65. During approach the
target XY is the observed cube XY immediately after reset. This avoids plotting
unused candidate coordinates or treating end-effector state as cube position.
Only small state/action arrays are loaded; no simulator or model is started.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLIPS = {"IMG_2092", "IMG_2093", "IMG_2095", "IMG_2096", "IMG_2098"}


def read_json(path):
    return json.loads(path.read_text())


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def replay_position(path):
    metadata_path = path.with_suffix(".json")
    metadata = read_json(metadata_path)
    if metadata.get("success") is not True or metadata.get("source", {}).get("kind") != "personal_video":
        raise ValueError(f"Only completed personal-video replays may be plotted: {path}")
    with np.load(path, allow_pickle=False) as arrays:
        states, actions = arrays["state"], arrays["action"]
    phases = metadata.get("phases", [])
    if len(phases) != len(actions) or states.shape != (len(actions), 8) or actions.shape[1:] != (7,):
        raise ValueError(f"Unaligned state/action/phase arrays: {path}")
    use = np.array([phase == "approach" for phase in phases]) & (np.max(np.abs(actions[:, :2]), axis=1) < .649)
    if not use.any():
        raise ValueError(f"No unsaturated approach frames available: {path}")
    targets = states[use, :2].astype(float) + actions[use, :2].astype(float) * (.05 / .65)
    xy = np.median(targets, axis=0)
    if not np.isfinite(targets).all() or np.max(np.ptp(targets, axis=0)) > 1e-6:
        raise ValueError(f"Approach target is not constant within one micrometre: {path}")
    settings = metadata["simulation_settings"]
    jitter = settings.get("randomize_start_m", .035)
    declared = np.asarray(settings.get("cube_start_xy", [0., -.17]), float)
    declared = declared + np.random.default_rng(metadata["seed"]).uniform(-jitter, jitter, 2)
    if np.linalg.norm(xy - declared) > 1e-5:
        raise ValueError(f"Recovered position disagrees with recorded reset settings: {path}")
    return {"name": path.name, "seed": metadata["seed"], "clip": Path(metadata["source"]["video"]).stem,
            "xy": xy.tolist(), "frames": len(actions), "source": metadata["source"],
            "state_action_sha256": hashlib.sha256(states.tobytes() + actions.tobytes()).hexdigest(),
            "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest()}, metadata


def selected_arm(folder, originals):
    manifest = read_json(folder / "build/plan.json")
    summary = read_json(folder / "build/summary.json")
    plan = manifest["selection_plan"]
    if canonical_hash(manifest) != summary["plan_sha256"] or not summary.get("complete"):
        raise ValueError(f"Incomplete or changed selected build: {folder}")
    selected = set(summary["selected_episodes"])
    if {p.name for p in folder.glob("*.npz")} != set(originals) | selected:
        raise ValueError(f"Export collection differs from completed selection: {folder}")
    points = []
    for name in sorted(selected):
        point, metadata = replay_position(folder / name)
        selection = metadata["selection"]
        job = plan["clips"][point["clip"]][selection["candidate_index"]]
        if (selection["plan_sha256"] != summary["plan_sha256"] or job["seed"] != point["seed"]
                or metadata["simulation_settings"].get("randomize_start_m") != 0
                or metadata["simulation_settings"].get("cube_start_xy") != job["cube_start_xy"]):
            raise ValueError(f"Recorded selected replay differs from its declared job: {name}")
        points.append(point)
    quotas = Counter(point["clip"] for point in points)
    planned = {clip: len(jobs) for clip, jobs in plan["clips"].items()}
    attempted = read_json(folder / "build/replays.json")
    if (set(quotas) != CLIPS or any(count != 4 for count in quotas.values())
            or set(planned) != CLIPS or any(count != 10 for count in planned.values())
            or len(attempted) != 20 or any(job.get("success") is not True for job in attempted)):
        raise ValueError("This figure requires the verified five-clips/four-selected-of-ten study design.")
    return {"points": points, "selected_per_clip": dict(quotas), "planned_per_clip": planned,
            "attempted": len(attempted), "selected": len(points), "unused_candidates": sum(planned.values()) - len(attempted),
            "plan_sha256": summary["plan_sha256"]}


def make_plot(original_dir, selection_root, notes_path, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle, Patch

    originals = {path.name: replay_position(path)[0] for path in sorted(original_dir.glob("*.npz"))}
    quotas = Counter(point["clip"] for point in originals.values())
    if set(quotas) != CLIPS or any(count != 4 for count in quotas.values()):
        raise ValueError("Expected original20, four successful replays per source clip.")
    arms = {name: selected_arm(selection_root / name, originals) for name in ("uniform", "targeted")}
    notes = read_json(notes_path)
    anchors = notes["failure_anchor_verification"]
    if len(anchors) != 5 or {a["seed"] for a in anchors} != {3010, 3012, 3013, 3014, 3015}:
        raise ValueError("Unexpected historical failure anchors.")
    old_xy = np.array([point["xy"] for point in originals.values()])
    anchor_xy = np.array([point["cube_start_xy"] for point in anchors])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "normal"})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.1), sharex=True, sharey=True)
    for ax, name in zip(axes, arms):
        xy = np.array([point["xy"] for point in arms[name]["points"]])
        ax.add_patch(Rectangle((-.06, -.23), .12, .12, fill=False, edgecolor="#aab0b5", linewidth=.9, linestyle=":"))
        ax.add_patch(Rectangle((-.035, -.205), .07, .07, fill=False, edgecolor="#7d858b", linewidth=.9, linestyle="--"))
        ax.scatter(old_xy[:, 0], old_xy[:, 1], s=24, facecolors="none", edgecolors="#555b60", linewidths=.9, zorder=3)
        ax.scatter(xy[:, 0], xy[:, 1], s=23, facecolors="#0072b2", edgecolors="white", linewidths=.35, zorder=4)
        ax.scatter(anchor_xy[:, 0], anchor_xy[:, 1], s=38, marker="x", c="#c44e20", linewidths=1.0, zorder=5)
        ax.set(title=f"Original 20 + {name} 20", xlabel="Cube start x (m)", xlim=(-.07, .07), ylim=(-.24, -.10))
        ax.set_xticks([-.06, -.03, 0, .03, .06]); ax.set_yticks([-.23, -.20, -.17, -.14, -.11])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=.12, linewidth=.6)
    axes[0].set_ylabel("Cube start y (m)")
    handles = [Line2D([], [], color="#555b60", marker="o", markerfacecolor="none", linestyle="none", markersize=5, label="Original replay"),
               Line2D([], [], color="#0072b2", marker="o", linestyle="none", markersize=5, label="Selected addition"),
               Line2D([], [], color="#c44e20", marker="x", linestyle="none", markersize=6, label="Old failure anchor"),
               Patch(facecolor="none", edgecolor="#7d858b", linestyle="--", label="Narrow range (±0.035 m)"),
               Patch(facecolor="none", edgecolor="#aab0b5", linestyle=":", label="Wide range (±0.060 m)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .09), ncol=3, frameon=False, columnspacing=1.4, handlelength=1.5, fontsize=8)
    fig.text(.5, .035, "Saved replay starts only · 5 source clips × 4 additions per arm · 30 unused candidates per arm omitted",
             ha="center", fontsize=7.7, color="#4b5359")
    fig.subplots_adjust(left=.10, right=.985, top=.93, bottom=.30, wspace=.20)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)
    report = {"kind": "replay_start_coverage_not_policy_evaluation", "source_recordings": 5,
              "coordinate_method": "Median unsaturated approach target: state_xy + action_xy * 0.05 / 0.65; verified against recorded reset metadata.",
              "original": list(originals.values()), "arms": arms, "historical_failure_anchors": anchors,
              "narrow_bounds_xy": [[-.035, .035], [-.205, -.135]], "wide_bounds_xy": [[-.06, .06], [-.23, -.11]],
              "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "selection_notes_sha256": hashlib.sha256(notes_path.read_bytes()).hexdigest()}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "original": len(originals), "arms": {
        name: {key: value[key] for key in ("selected_per_clip", "planned_per_clip", "attempted", "selected", "unused_candidates")}
        for name, value in arms.items()}}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dir", default="data/robot/main")
    parser.add_argument("--selection-root", default="data/robot/improvement_20261011")
    parser.add_argument("--notes", default="runs/improvement_20261011/data_selection_notes.json")
    parser.add_argument("--output", default="runs/improvement_20261011/inspection/replay_coverage.png")
    args = parser.parse_args()
    make_plot(ROOT / args.original_dir, ROOT / args.selection_root, ROOT / args.notes, ROOT / args.output)


if __name__ == "__main__":
    main()
