"""A comparison must pair actual starts and keep each result tied to its model."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from scripts import summarize_imu_pilot as summarizer


class PilotSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.study = self.root / "results/imu_pilot_20261015"
        self.plan = {"evaluation": {
            "models": list(summarizer.MODELS), "n_action_steps": 5,
            "narrow": {"radius_m": .035, "seed": 84000, "episodes": 5},
            "wide": {"radius_m": .07, "seed": 85000, "episodes": 5}}}
        self.write(self.study / "plan.json", self.plan)
        outcomes = {"original": [False, True, False, False, True, False, False, True, False, False],
                    "continuation": [True, True, False, False, True, False, False, True, False, False],
                    "augmented": [True, False, False, True, True, False, True, True, False, False]}
        for model in summarizer.MODELS:
            folder = self.study / "evaluation" / model
            identity = {"model": model, "plan_sha256": summarizer.digest(self.study / "plan.json"),
                        "checkpoint": str(self.root / "checkpoints" / model),
                        "checkpoint_files_sha256": {"model.safetensors": f"synthetic-{model}"},
                        "source_sha256": {"learning.py": "same-code"},
                        "scene_assets_sha256": {"cube.xml": "same-scene"}}
            self.write(folder / "identity.json", identity)
            completed = {"model": model, "kind": "single_recording_imu_assisted_policy_evaluation",
                         "identity_sha256": summarizer.digest(folder / "identity.json"), "groups": {},
                         "episodes": 10, "successes": sum(outcomes[model])}
            for group_index, group in enumerate(summarizer.GROUPS):
                settings = self.plan["evaluation"][group]
                config = {"simulation": {"randomize_start_m": settings["radius_m"],
                                         "orientation_control": "fixed_downward", "horizon": 600,
                                         "control_hz": 20, "image_size": 256},
                          "evaluation": {"save_video": True, "save_trace": True, "n_action_steps": 5,
                                         "episodes": 5, "seed": settings["seed"]}}
                self.write(folder / group / "manifest.json", {
                    "identity_sha256": completed["identity_sha256"], "group": group,
                    "episode_plan": settings, "config": config})
                output = folder / group / "evaluation"
                episodes = []
                for index in range(5):
                    success = outcomes[model][group_index * 5 + index]
                    events = {"contact_step": 0, "bilateral_contact_step": 0, "firm_grasp_step": 0,
                              "lift_step": 1 if success else None, "release_after_lift_step": None}
                    episode = {"seed": settings["seed"] + index, "success": success, "steps": 2,
                               "first_events": events, "max_cube_height_m": 1. if success else .93,
                               "video": f"episode_{index:03d}.mp4", "wall_seconds": 1.}
                    episodes.append(episode)
                    self.write(output / episode["video"], "synthetic fixture, not a physics video")
                    self.write(output / f"episode_{index:03d}_trace.json", [
                        {"step": step, "episode_seed": episode["seed"], "cube_xyz": [float(index), 0., .92],
                         "observation_state": [0.] * 8, "success": success if step == 1 else False}
                        for step in range(2)])
                report = {"kind": "trained_policy_physics_evaluation", "checkpoint": identity["checkpoint"],
                          "randomize_start_m": settings["radius_m"], "seed_start": settings["seed"],
                          "policy_settings": {"n_action_steps": 5}, "orientation_control": "fixed_downward",
                          "held_out_from_training_seeds": True, "video_recorded": True, "episodes": episodes,
                          "successes": sum(row["success"] for row in episodes), "wall_seconds": 5.}
                report["success_rate"] = report["successes"] / 5
                self.write(output / "report.json", report)
                self.write(output / "episodes.json", episodes)
                self.refresh_output_receipt(model, group)
                completed["groups"][group] = {"report": str((output / "report.json").relative_to(self.root)),
                                               "report_sha256": summarizer.digest(output / "report.json"),
                                               "successes": report["successes"], "episodes": 5,
                                               "success_rate": report["success_rate"], "episode_results": episodes}
            self.write(folder / "summary.json", completed)
        for name, value in (("ROOT", self.root), ("STUDY", self.study)):
            mocked = patch.object(summarizer, name, value)
            mocked.start()
            self.addCleanup(mocked.stop)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value))

    def refresh_output_receipt(self, model, group):
        folder = self.study / "evaluation" / model / group
        output = folder / "evaluation"
        self.write(folder / "outputs.json", {str(path.relative_to(output)): summarizer.digest(path)
                                             for path in output.rglob("*") if path.is_file()})

    def test_counts_and_paired_outcomes_include_every_planned_start(self):
        result = summarizer.collect()
        self.assertEqual({model: row["successes"] for model, row in result["models"].items()},
                         {"original": 3, "continuation": 4, "augmented": 5})
        self.assertEqual(result["paired"]["augmented_vs_original"], {
            "improved": 3, "regressed": 1, "both_successful": 2, "both_failed": 4,
            "identical_initial_cube_and_robot_states": True})
        self.assertEqual(result["paired"]["augmented_vs_continuation"]["improved"], 2)

    def test_mismatched_initial_states_cannot_be_called_paired(self):
        path = self.study / "evaluation/augmented/wide/evaluation/episode_000_trace.json"
        trace = summarizer.read(path)
        trace[0]["cube_xyz"][0] += .01
        self.write(path, trace)
        self.refresh_output_receipt("augmented", "wide")
        with self.assertRaisesRegex(ValueError, "Initial cube/robot state differs"):
            summarizer.collect()

    def test_group_copied_from_other_model_is_rejected(self):
        destination = self.study / "evaluation/augmented/narrow"
        shutil.rmtree(destination)
        shutil.copytree(self.study / "evaluation/original/narrow", destination)
        with self.assertRaises(ValueError):
            summarizer.collect()

    def test_completed_summary_cannot_point_to_a_different_report_hash(self):
        path = self.study / "evaluation/continuation/summary.json"
        completed = summarizer.read(path)
        completed["groups"]["wide"]["report_sha256"] = "different-report"
        self.write(path, completed)
        with self.assertRaises(ValueError):
            summarizer.collect()

    def test_horizon_change_across_models_invalidates_comparison(self):
        path = self.study / "evaluation/augmented/wide/manifest.json"
        manifest = summarizer.read(path)
        manifest["config"]["simulation"]["horizon"] = 1200
        self.write(path, manifest)
        with self.assertRaises(ValueError):
            summarizer.collect()

    def test_report_checkpoint_must_match_model_identity(self):
        path = self.study / "evaluation/augmented/wide/evaluation/report.json"
        report = summarizer.read(path)
        report["checkpoint"] = str(self.root / "checkpoints/original")
        self.write(path, report)
        self.refresh_output_receipt("augmented", "wide")
        # Keep the completion hash internally consistent to isolate checkpoint validation.
        completed_path = self.study / "evaluation/augmented/summary.json"
        completed = summarizer.read(completed_path)
        completed["groups"]["wide"]["report_sha256"] = summarizer.digest(path)
        self.write(completed_path, completed)
        with self.assertRaises(ValueError):
            summarizer.collect()


if __name__ == "__main__":
    unittest.main()
