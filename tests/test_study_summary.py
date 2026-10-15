"""Study statistics and strict pairing tests using synthetic reports only."""
import importlib.util
import hashlib
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("summarize_study", Path(__file__).resolve().parents[1] / "scripts/summarize_study.py")
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_study(root):
    plan = {
        "source_videos": {f"source_{index}.MOV": "fixture" for index in range(5)},
        "development": {"known_wide": {"seed": 3010, "episodes": 3, "radius_m": .06}},
        "final_test": {"narrow": {"seed": 6100, "episodes": 3, "radius_m": .035},
                       "wide": {"seed": 7100, "episodes": 3, "radius_m": .06}},
        "selection_rule": "Select on development only.",
    }
    write_json(root / "study_plan.json", plan)
    plan_hash = summary.digest(root / "study_plan.json")
    identities = {name: {"checkpoint_files": {"model.safetensors": name}, "n_action_steps": 5} for name in ("baseline", "candidate")}
    context = {"config_sha256": hashlib.sha256(b"synthetic base configuration").hexdigest(),
               "source_sha256": {"phone_to_panda/learning.py": hashlib.sha256(b"synthetic policy code").hexdigest(),
                                 "phone_to_panda/simulation.py": hashlib.sha256(b"synthetic simulator code").hexdigest()}}
    write_json(root / "candidate_lock.json", {"study_plan_sha256": plan_hash, "candidates": identities,
        "reference_candidate": "baseline", "selected_candidate": "candidate",
        "contrast_roles": {"candidate": "Primary selected adaptation"},
        "selection_sidecars_sha256": {"selection_rule.json": hashlib.sha256(b"synthetic selection rule").hexdigest()}, **context})
    for phase in ("development", "final_test"):
        for name in identities:
            for group, settings in plan[phase].items():
                folder = root / phase / name / group
                write_json(folder / "manifest.json", {
                    "candidate": name, "phase": phase, "group": group, "episode_plan": settings,
                    "study_plan_sha256": plan_hash, "identity": identities[name],
                    "source_sha256": context["source_sha256"],
                    **({"base_config_sha256": context["config_sha256"],
                        "candidate_lock_sha256": summary.digest(root / "candidate_lock.json")} if phase == "final_test" else {}),
                    "config": {"simulation": {"control_hz": 20, "horizon": 600, "image_size": 256,
                                                "orientation_control": "fixed_downward"}},
                })
                episodes = []
                for index in range(3):
                    success = index == 0 or (name == "candidate" and index == 1)
                    seed = settings["seed"] + index
                    episodes.append({"seed": seed, "success": success, "max_cube_height_m": 1.02 if success or index == 2 else .93, "steps": 1})
                    write_json(folder / "evaluation" / f"episode_{index:03d}_trace.json", [{
                        "step": 0, "episode_seed": seed, "cube_xyz": [index * .01, -.17, .92],
                        "observation_state": [0., -.15, 1.1, 0., 0., 0., .04, -.04], "success": success,
                    }])
                write_json(folder / "evaluation/report.json", {
                    "kind": "trained_policy_physics_evaluation", "episodes": episodes,
                    "seed_start": settings["seed"], "randomize_start_m": settings["radius_m"],
                    "orientation_control": "fixed_downward", "policy_settings": {"n_action_steps": 5},
                    "successes": sum(episode["success"] for episode in episodes),
                    "success_rate": sum(episode["success"] for episode in episodes) / 3, "wall_seconds": 4.,
                })
    return root


def test_wilson_zero_all_and_single_trial_boundaries():
    assert summary.wilson_interval(0, 10) == pytest.approx([0., .2775327998628892])
    assert summary.wilson_interval(10, 10) == pytest.approx([.7224672001371107, 1.])
    assert summary.wilson_interval(1, 1) == pytest.approx([.20654931437723745, 1.])
    for successes, trials in [(0, 0), (-1, 10), (11, 10), (True, 10), (1, 1.5)]:
        with pytest.raises(ValueError):
            summary.wilson_interval(successes, trials)


def test_stage_counts_and_exact_discordant_test():
    counts = summary.summarize_trials([{"success": True, "lifted": True},
                                      {"success": False, "lifted": True},
                                      {"success": False, "lifted": False}])
    assert (counts["placements"], counts["lifts"], counts["failed_lift"], counts["failed_after_lift"]) == (1, 2, 1, 1)
    assert summary.exact_mcnemar(0, 0) == 1.
    assert summary.exact_mcnemar(1, 1) == 1.
    assert summary.exact_mcnemar(5, 0) == pytest.approx(.0625)
    assert summary.exact_mcnemar(0, 5) == pytest.approx(.0625)
    with pytest.raises(ValueError):
        summary.summarize_trials([])


def test_complete_final_pairs_count_improvements_and_keep_phase_separate(tmp_path):
    study = make_study(tmp_path)
    result = summary.build_summary(study, "final_test", ["baseline", "candidate"])
    assert result["source_recording_count"] == 5
    assert result["locked_selection"]["selected_candidate"] == "candidate"
    assert result["locked_selection"]["reference_candidate"] == "baseline"
    assert result["locked_selection"]["report_scope"] == "all_locked_candidates"
    assert result["models"]["baseline"]["combined"]["placements"] == 2
    assert result["models"]["candidate"]["combined"]["placements"] == 4
    assert result["models"]["baseline"]["combined"]["trials"] == 6
    pair = result["paired"]["candidate"]["combined"]
    assert pair["improved"] == 2 and pair["regressed"] == 0
    assert pair["both_success"] == 2 and pair["both_failure"] == 2
    assert pair["identical_initial_cube_and_robot_states"]
    assert "placement_wilson95" not in result["models"]["candidate"]["combined"]
    assert "lift_wilson95" not in result["models"]["candidate"]["combined"]
    assert "placement_wilson95" in result["models"]["candidate"]["groups"]["wide"]
    assert all(trial["seed"] >= 6100 for group in result["models"]["candidate"]["groups"].values() for trial in group["episode_results"])


@pytest.mark.parametrize("key,index", [("cube_xyz", 0), ("observation_state", 3)])
def test_exact_initial_state_mismatch_is_rejected(tmp_path, key, index):
    study = make_study(tmp_path)
    path = study / "final_test/candidate/wide/evaluation/episode_000_trace.json"
    trace = summary.read_json(path)
    trace[0][key][index] += 1e-12
    write_json(path, trace)
    with pytest.raises(ValueError, match="initial state mismatch"):
        summary.build_summary(study, "final_test", ["baseline", "candidate"])


def test_missing_or_partial_final_group_fails(tmp_path):
    study = make_study(tmp_path)
    with pytest.raises(ValueError, match="every frozen final group"):
        summary.build_summary(study, "final_test", ["baseline", "candidate"], ["narrow"])
    path = study / "final_test/candidate/wide/evaluation/report.json"
    report = summary.read_json(path)
    report["episodes"].pop()
    write_json(path, report)
    with pytest.raises(ValueError, match="Incomplete or mismatched seed"):
        summary.build_summary(study, "final_test", ["baseline", "candidate"])
    path.unlink()
    with pytest.raises(FileNotFoundError):
        summary.build_summary(study, "final_test", ["baseline", "candidate"])


def test_wrong_seed_and_changed_candidate_lock_are_rejected(tmp_path):
    study = make_study(tmp_path)
    path = study / "final_test/candidate/wide/evaluation/report.json"
    report = summary.read_json(path)
    report["episodes"][0]["seed"] = 1
    write_json(path, report)
    with pytest.raises(ValueError, match="seed group"):
        summary.build_summary(study, "final_test", ["baseline", "candidate"])
    report["episodes"][0]["seed"] = 7100
    write_json(path, report)
    lock_path = study / "candidate_lock.json"
    lock = summary.read_json(lock_path)
    lock["candidates"]["candidate"]["n_action_steps"] = 1
    write_json(lock_path, lock)
    with pytest.raises(ValueError, match="selection lock"):
        summary.build_summary(study, "final_test", ["baseline", "candidate"])


def test_development_does_not_read_final_reports_or_lock(tmp_path, monkeypatch):
    study = make_study(tmp_path)
    original_reader = summary.read_json
    def development_only(path):
        assert "final_test" not in Path(path).parts
        assert Path(path).name != "candidate_lock.json"
        return original_reader(path)
    monkeypatch.setattr(summary, "read_json", development_only)
    result = summary.build_summary(study, "development", ["baseline", "candidate"], ["known_wide"])
    assert result["models"]["baseline"]["combined"]["trials"] == 3
    assert result["limits"][0].startswith("Development results are diagnostics")


def test_paired_regression_is_not_reported_as_improvement():
    reference = [{"seed": i, "group": "wide", "initial_cube_xyz": [0, 0, .92],
                  "initial_observation_state": [0] * 8, "success": success} for i, success in enumerate([True, False, True, False])]
    candidate = [{**trial, "success": success} for trial, success in zip(reference, [False, True, True, False])]
    pair = summary.paired_counts(reference, candidate)
    assert (pair["improved"], pair["regressed"], pair["both_success"], pair["both_failure"]) == (1, 1, 1, 1)
    assert pair["placement_rate_difference"] == 0.
    with pytest.raises(ValueError, match="seed/group"):
        summary.paired_counts(reference, list(reversed(candidate)))


@pytest.mark.parametrize("corruption,message", [
    ("aggregate", "success aggregate"),
    ("success_without_lift", "without required lift"),
    ("horizon", "action horizon"),
    ("empty_trace", "Incomplete action trace"),
    ("trace_outcome", "outcome differs"),
    ("physics_horizon", "settings differ"),
])
def test_inconsistent_report_or_evaluation_contract_is_rejected(tmp_path, corruption, message):
    study = make_study(tmp_path)
    folder = study / "final_test/candidate/wide"
    path = folder / "evaluation/report.json"
    report = summary.read_json(path)
    if corruption == "aggregate":
        report["successes"] = 0
    elif corruption == "success_without_lift":
        report["episodes"][0]["max_cube_height_m"] = .94
    elif corruption == "horizon":
        report["policy_settings"]["n_action_steps"] = 1
    elif corruption == "physics_horizon":
        manifest_path = folder / "manifest.json"
        manifest = summary.read_json(manifest_path)
        manifest["config"]["simulation"]["horizon"] = 100
        write_json(manifest_path, manifest)
    else:
        trace_path = folder / "evaluation/episode_000_trace.json"
        trace = summary.read_json(trace_path)
        if corruption == "empty_trace":
            trace = []
        else:
            trace[-1]["success"] = False
        write_json(trace_path, trace)
    write_json(path, report)
    with pytest.raises(ValueError, match=message):
        summary.build_summary(study, "final_test", ["baseline", "candidate"])


def test_changed_frozen_plan_is_rejected_even_when_seed_groups_unchanged(tmp_path):
    study = make_study(tmp_path)
    path = study / "study_plan.json"
    plan = summary.read_json(path)
    plan["selection_rule"] = "Changed after candidate lock"
    write_json(path, plan)
    with pytest.raises(ValueError, match="changed study plan"):
        summary.build_summary(study, "final_test", ["baseline", "candidate"])


@pytest.mark.parametrize("field,message", [("source_sha256", "source code"), ("base_config_sha256", "base configuration")])
@pytest.mark.parametrize("remove", [True, False])
def test_final_manifest_context_must_match_locked_source_and_config(tmp_path, field, message, remove):
    study = make_study(tmp_path)
    path = study / "final_test/candidate/wide/manifest.json"
    manifest = summary.read_json(path)
    if remove:
        del manifest[field]
    else:
        manifest[field] = "changed"
    write_json(path, manifest)
    with pytest.raises(ValueError, match=message):
        summary.build_summary(study, "final_test", ["baseline", "candidate"])


@pytest.mark.parametrize("field", ["source_sha256", "config_sha256"])
def test_old_final_lock_without_full_context_is_rejected(tmp_path, field):
    study = make_study(tmp_path)
    path = study / "candidate_lock.json"
    lock = summary.read_json(path)
    del lock[field]
    write_json(path, lock)
    with pytest.raises(ValueError, match="selection lock"):
        summary.build_summary(study, "final_test", ["baseline", "candidate"])


def test_combined_intervals_are_absent_from_markdown_and_plot(tmp_path, monkeypatch):
    from matplotlib.axes import Axes
    study = make_study(tmp_path / "study")
    result = summary.build_summary(study, "final_test", ["baseline", "candidate"])
    document = tmp_path / "summary.md"
    summary.write_markdown(result, document)
    first_table = document.read_text().split("All compared starts", 1)[0]
    combined_rows = [line for line in first_table.splitlines() if "| combined |" in line]
    assert len(combined_rows) == 2
    assert all("Not pooled" in line for line in combined_rows)
    assert any("One training seed" in limit for limit in result["limits"])
    original = Axes.errorbar
    calls = []
    def checked_errorbar(self, x, y, *args, **kwargs):
        calls.append((len(x), len(y)))
        return original(self, x, y, *args, **kwargs)
    monkeypatch.setattr(Axes, "errorbar", checked_errorbar)
    summary.plot_summary(result, tmp_path)
    assert calls == [(2, 2), (2, 2)]  # Two groups; the combined bar has no errorbar.


def test_final_reference_cannot_be_swapped(tmp_path):
    study = make_study(tmp_path)
    with pytest.raises(ValueError, match="locked reference_candidate first"):
        summary.build_summary(study, "final_test", ["candidate", "baseline"])


def test_changed_selection_decision_is_rejected_even_when_models_are_unchanged(tmp_path):
    study = make_study(tmp_path)
    path = study / "candidate_lock.json"
    lock = summary.read_json(path)
    lock["selected_candidate"] = "baseline"
    write_json(path, lock)
    with pytest.raises(ValueError, match="selection lock changed"):
        summary.build_summary(study, "final_test", ["baseline", "candidate"])


def test_final_subset_is_explicit_and_retains_locked_selection_and_roles(tmp_path):
    study = make_study(tmp_path)
    path = study / "candidate_lock.json"
    lock = summary.read_json(path)
    lock["candidates"]["other"] = lock["candidates"]["candidate"]
    lock["selected_candidate"] = "other"
    lock["contrast_roles"]["other"] = "Primary selected extension"
    lock["contrast_roles"]["candidate"] = "Secondary matched-data ablation"
    write_json(path, lock)
    # A synthetic pre-run lock with a third model; no reports for that omitted
    # model are needed to create an explicitly secondary subset report.
    for manifest_path in (study / "final_test").glob("*/*/manifest.json"):
        manifest = summary.read_json(manifest_path)
        manifest["candidate_lock_sha256"] = summary.digest(path)
        write_json(manifest_path, manifest)
    result = summary.build_summary(study, "final_test", ["baseline", "candidate"])
    selection = result["locked_selection"]
    assert selection["report_scope"] == "secondary_subset"
    assert selection["selected_candidate"] == "other"
    assert selection["selected_candidate_included"] is False
    assert selection["omitted_locked_candidates"] == ["other"]
    assert selection["contrast_roles"] == lock["contrast_roles"]
    output = tmp_path / "subset.md"
    summary.write_markdown(result, output)
    assert "secondary subset" in output.read_text()
    assert "Selected before final testing: **other**" in output.read_text()


@pytest.mark.parametrize("remove", [True, False])
def test_missing_or_wrong_final_lock_digest_is_rejected(tmp_path, remove):
    study = make_study(tmp_path)
    path = study / "final_test/candidate/wide/manifest.json"
    manifest = summary.read_json(path)
    if remove:
        del manifest["candidate_lock_sha256"]
    else:
        manifest["candidate_lock_sha256"] = "changed"
    write_json(path, manifest)
    with pytest.raises(ValueError, match="selection lock changed"):
        summary.build_summary(study, "final_test", ["baseline", "candidate"])
