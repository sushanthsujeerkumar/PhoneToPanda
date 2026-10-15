"""Pilot evaluation keeps the fixed design and never replaces existing evidence."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import evaluate_imu_pilot as evaluator


class PilotEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.plan = {
            "paired_video": "IMG_2276.MOV", "sensor_session_id": "recorded-session",
            "primary_offset_s": .54, "offset_status": "provisional", "source_human_recordings": 1,
            "training": {"initial_checkpoint": "checkpoints/main"},
            "evaluation": {"models": list(evaluator.MODELS), "n_action_steps": 5,
                           "narrow": {"radius_m": .035, "seed": 84000, "episodes": 5},
                           "wide": {"radius_m": .07, "seed": 85000, "episodes": 5},
                           "selection": "No model selection", "interpretation": "Same scene"},
            "scope": "Extra demonstration pilot, not sensor fusion",
        }
        self.write(evaluator.STUDY / "plan.json", self.plan)
        self.config = {"simulation": {"orientation_control": "fixed_downward"},
                       "evaluation": {"save_video": False}, "paths": {"dataset": "data/datasets/main"}}
        for name in ("continuation", "augmented"):
            self.write(f"configs/imu_pilot_20261015/{name}.yaml", "synthetic fixture configuration")
        for name in evaluator.MODELS:
            checkpoint = ("checkpoints/main" if name == "original" else
                          f"runs/imu_pilot_20261015/train_{name}/checkpoints/last/pretrained_model")
            for file in ("model.safetensors", "config.json", "train_config.json", "phone_to_panda_provenance.json"):
                self.write(f"{checkpoint}/{file}", "synthetic fixture, never loaded")
        for name in ("phone_to_panda/common.py", "phone_to_panda/learning.py", "phone_to_panda/simulation.py",
                     "scripts/evaluate_candidate.py", "scripts/evaluate_imu_pilot.py", "assets/cube.xml"):
            self.write(name, "synthetic source fixture")
        self.write("data/datasets/main/meta/phone_to_panda_provenance.json", {})
        for name in ("dataset_inputs.json", "dataset_summary.json", "replays.json"):
            self.write(evaluator.STUDY / name, {})
        video = self.write("IMG_2276.MOV", "synthetic fixture, never decoded")
        session = self.write("session.json", {})
        events = self.write("events.json", {"imu_time_offset_s": .54, "synchronization": "provisional"})
        self.write("data/experimental/imu_20261015/trajectory.json", {"source": {
            "kind": "personal_video_imu_pilot", "session_id": "recorded-session",
            "video": str(video), "sha256": evaluator.digest(video), "imu_session": str(session),
            "imu_sha256": evaluator.digest(session), "events": str(events), "events_sha256": evaluator.digest(events),
        }})
        self.root_patch = patch.object(evaluator, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.config_patch = patch.object(evaluator, "load_config", side_effect=lambda _: deepcopy(self.config))
        self.config_patch.start()
        self.addCleanup(self.config_patch.stop)

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value))
        return path

    def fake_run(self, config, checkpoint, output, *, episodes, seed):
        self.assertEqual(episodes, 5)
        self.assertIn(seed, (84000, 85000))
        self.assertEqual(config["evaluation"], {"save_video": True, "save_trace": True,
                                              "episodes": 5, "seed": seed, "n_action_steps": 5})
        output.mkdir()
        rows = []
        for index in range(episodes):
            events = dict.fromkeys(evaluator.EVENTS)
            if index in (0, 2, 3, 4):
                events["contact_step"] = 0
            if index in (0, 3, 4):
                events["bilateral_contact_step"] = events["firm_grasp_step"] = 0
            if index in (0, 4):
                events["lift_step"] = 1
            row = {"seed": seed + index, "success": index == 0, "steps": 2,
                   "first_events": events, "max_cube_height_m": 1. if index in (0, 4) else .92,
                   "video": f"episode_{index:03d}.mp4"}
            rows.append(row)
            (output / row["video"]).write_bytes(b"mock video, not a physics result")
            (output / f"episode_{index:03d}_trace.json").write_text(json.dumps([
                {"step": step, "episode_seed": row["seed"], "success": row["success"] if step == 1 else False}
                for step in range(2)]))
        report = {"kind": "trained_policy_physics_evaluation", "checkpoint": str(checkpoint),
                  "randomize_start_m": config["simulation"]["randomize_start_m"], "seed_start": seed,
                  "policy_settings": {"n_action_steps": 5}, "orientation_control": "fixed_downward",
                  "held_out_from_training_seeds": True, "video_recorded": True, "episodes": rows,
                  "successes": 1, "success_rate": .2, "wall_seconds": 1.}
        (output / "episodes.json").write_text(json.dumps(rows))
        (output / "report.json").write_text(json.dumps(report))
        return report

    def run_original(self):
        return evaluator.main(["--model", "original"])

    def test_both_groups_saved_with_raw_outcomes_and_exact_resume(self):
        with patch.object(evaluator, "run_policy", side_effect=self.fake_run) as run:
            self.assertEqual(self.run_original(), 0)
            self.assertEqual(self.run_original(), 0)
            self.assertEqual(run.call_count, 2)
        summary = evaluator.read_json(self.root / evaluator.STUDY / "evaluation/original/summary.json")
        self.assertEqual((summary["successes"], summary["episodes"]), (2, 10))
        self.assertEqual(summary["groups"]["wide"]["failure_categories"], {
            "no_gripper_contact": 1, "contact_without_firm_grasp": 1,
            "firm_grasp_without_lift": 1, "lifted_without_successful_placement": 1})
        self.assertEqual([row["seed"] for row in summary["groups"]["wide"]["episode_results"]], list(range(85000, 85005)))

    def test_changed_model_config_plan_source_or_data_cannot_reuse_results(self):
        with patch.object(evaluator, "run_policy", side_effect=self.fake_run):
            self.run_original()
        for name in ("checkpoints/main/model.safetensors", "configs/imu_pilot_20261015/continuation.yaml",
                     "phone_to_panda/learning.py", "assets/cube.xml", "results/imu_pilot_20261015/plan.json",
                     "results/imu_pilot_20261015/dataset_inputs.json"):
            path = self.root / name
            original = path.read_text()
            path.write_text(original + "\n")  # Same parsed JSON, different bytes must still be rejected.
            with self.subTest(name=name), patch.object(evaluator, "run_policy") as run:
                with self.assertRaisesRegex(ValueError, "Existing record differs"):
                    self.run_original()
                run.assert_not_called()
            path.write_text(original)

    def test_modified_saved_trace_is_rejected_without_rerunning(self):
        with patch.object(evaluator, "run_policy", side_effect=self.fake_run):
            self.run_original()
        trace = self.root / evaluator.STUDY / "evaluation/original/wide/evaluation/episode_000_trace.json"
        trace.write_text("[]")
        with patch.object(evaluator, "run_policy") as run, self.assertRaisesRegex(ValueError, "outputs changed"):
            self.run_original()
        run.assert_not_called()
        self.assertEqual(trace.read_text(), "[]")

    def test_interrupted_run_is_retained_and_not_repeated(self):
        def interrupt(config, checkpoint, output, **kwargs):
            output.mkdir()
            (output / "partial.mp4").write_bytes(b"retained partial evidence")
            raise RuntimeError("test interruption")
        with patch.object(evaluator, "run_policy", side_effect=interrupt), self.assertRaises(RuntimeError):
            self.run_original()
        with patch.object(evaluator, "run_policy") as run, self.assertRaisesRegex(ValueError, "Incomplete evaluation retained"):
            self.run_original()
        run.assert_not_called()
        self.assertTrue((self.root / evaluator.STUDY / "evaluation/original/narrow/evaluation/partial.mp4").is_file())

    def test_model_choices_resolve_expected_checkpoint_and_config(self):
        for model in ("continuation", "augmented"):
            with self.subTest(model=model), patch.object(evaluator, "run_policy", side_effect=self.fake_run) as run:
                evaluator.main(["--model", model])
                self.assertEqual(run.call_args.args[1], self.root / f"runs/imu_pilot_20261015/train_{model}/checkpoints/last/pretrained_model")
                identity = evaluator.read_json(self.root / evaluator.STUDY / f"evaluation/{model}/identity.json")
                self.assertTrue(identity["base_config"].endswith(f"/{model}.yaml"))

    def test_source_change_mid_run_blocks_receipt_and_second_group(self):
        def change_source(*args, **kwargs):
            result = self.fake_run(*args, **kwargs)
            self.write("phone_to_panda/simulation.py", "changed during evaluation")
            return result
        with patch.object(evaluator, "run_policy", side_effect=change_source) as run:
            with self.assertRaisesRegex(ValueError, "inputs changed during"):
                self.run_original()
            self.assertEqual(run.call_count, 1)
        self.assertFalse((self.root / evaluator.STUDY / "evaluation/original/narrow/outputs.json").exists())

    def test_missing_required_video_prevents_completion_receipt(self):
        def missing_video(config, checkpoint, output, **kwargs):
            report = self.fake_run(config, checkpoint, output, **kwargs)
            (output / "episode_003.mp4").unlink()
            return report
        with patch.object(evaluator, "run_policy", side_effect=missing_video) as run:
            with self.assertRaisesRegex(ValueError, "video is missing"):
                self.run_original()
            self.assertEqual(run.call_count, 1)
        self.assertFalse((self.root / evaluator.STUDY / "evaluation/original/narrow/outputs.json").exists())


if __name__ == "__main__":
    unittest.main()
