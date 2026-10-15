"""Cached evaluations and final-test locks must identify the actual experiment."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import evaluate_candidate as evaluator
from scripts.evaluate_candidate import validate_final_lock, validate_report


class EvaluationReportTests(unittest.TestCase):
    def setUp(self):
        self.checkpoint = Path("/tmp/phone-to-panda-test-checkpoint")
        self.settings = {"radius_m": .06, "seed": 7100, "episodes": 2}
        self.config = {"simulation": {"orientation_control": "fixed_downward"},
                       "evaluation": {"save_video": False}}
        self.report = {"kind": "trained_policy_physics_evaluation", "checkpoint": str(self.checkpoint),
                       "randomize_start_m": .06, "seed_start": 7100,
                       "policy_settings": {"n_action_steps": 5},
                       "orientation_control": "fixed_downward", "held_out_from_training_seeds": True,
                       "video_recorded": False, "episodes": [{"seed": 7100, "success": True},
                                                               {"seed": 7101, "success": False}],
                       "successes": 1, "success_rate": .5, "wall_seconds": 25.}

    def validate(self, report):
        validate_report(report, checkpoint=self.checkpoint, action_steps=5,
                        settings=self.settings, config=self.config)

    def test_matching_report_is_reusable_without_mutation(self):
        original = deepcopy(self.report)
        self.validate(self.report)
        self.assertEqual(self.report, original)

    def test_wrong_checkpoint_radius_horizon_or_controller_is_rejected(self):
        changes = ({"checkpoint": "/tmp/other-checkpoint"}, {"randomize_start_m": .035},
                   {"seed_start": 7101}, {"policy_settings": {"n_action_steps": 1}},
                   {"orientation_control": "policy"}, {"video_recorded": True},
                   {"held_out_from_training_seeds": False})
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate({**self.report, **change})

    def test_missing_repeated_or_reordered_seeds_are_rejected(self):
        for episodes in (self.report["episodes"][:1], self.report["episodes"] * 2,
                         list(reversed(self.report["episodes"]))):
            with self.subTest(episodes=episodes), self.assertRaisesRegex(ValueError, "every planned seed"):
                self.validate({**self.report, "episodes": episodes})

    def test_success_count_and_rate_must_match_boolean_episode_results(self):
        changes = ({"successes": 2}, {"successes": True}, {"success_rate": 1.},
                   {"success_rate": float("nan")}, {"success_rate": True},
                   {"episodes": [{"seed": 7100, "success": "false"}, {"seed": 7101, "success": False}]})
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate({**self.report, **change})


class FinalLockTests(unittest.TestCase):
    def setUp(self):
        self.context = {"study_plan_sha256": "plan-hash", "config_sha256": "config-hash",
                        "source_sha256": {"phone_to_panda/learning.py": "learning-hash",
                                          "phone_to_panda/simulation.py": "simulation-hash"},
                        "candidate": "targeted_h5", "identity": {"checkpoint_files": {"model.safetensors": "model-hash"},
                                                                    "n_action_steps": 5}}
        self.lock = {key: deepcopy(value) for key, value in self.context.items() if key not in {"candidate", "identity"}}
        self.lock["candidates"] = {self.context["candidate"]: deepcopy(self.context["identity"])}
        self.lock.update(reference_candidate=self.context["candidate"], selected_candidate=self.context["candidate"],
                         contrast_roles={}, selection_sidecars_sha256={"selection_rule.json": "selection-hash"})

    def test_exact_context_and_candidate_are_accepted(self):
        validate_final_lock(self.lock, **self.context)

    def test_source_config_or_plan_cannot_change_between_final_groups(self):
        for field in ("study_plan_sha256", "config_sha256", "source_sha256"):
            for replace in (True, False):
                lock = deepcopy(self.lock)
                if replace:
                    lock[field] = "different"
                else:
                    del lock[field]
                with self.subTest(field=field, replace=replace), self.assertRaisesRegex(ValueError, field):
                    validate_final_lock(lock, **self.context)

    def test_weight_or_action_horizon_changes_are_rejected(self):
        for identity in ({"checkpoint_files": {"model.safetensors": "other"}, "n_action_steps": 5},
                         {"checkpoint_files": {"model.safetensors": "model-hash"}, "n_action_steps": 1}):
            lock = deepcopy(self.lock)
            lock["candidates"]["targeted_h5"] = identity
            with self.assertRaisesRegex(ValueError, "Candidate differs"):
                validate_final_lock(lock, **self.context)

    def test_selection_and_contrast_roles_are_required(self):
        for field in ("reference_candidate", "selected_candidate", "contrast_roles", "selection_sidecars_sha256"):
            lock = deepcopy(self.lock)
            del lock[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_final_lock(lock, **self.context)

    def test_selection_sidecars_are_hash_checked_and_cannot_escape_study(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            sidecar = study / "selection_rule.json"
            sidecar.write_text('{"rule": "development only"}')
            lock = {"selection_sidecars_sha256": {sidecar.name: evaluator.digest(sidecar)}}
            evaluator.validate_selection_sidecars(study, lock)
            sidecar.write_text('{"rule": "changed"}')
            with self.assertRaisesRegex(ValueError, "sidecar differs"):
                evaluator.validate_selection_sidecars(study, lock)
            lock["selection_sidecars_sha256"] = {"../outside.json": "hash"}
            with self.assertRaisesRegex(ValueError, "inside the study"):
                evaluator.validate_selection_sidecars(study, lock)


class EvaluationManifestTests(unittest.TestCase):
    def test_base_config_fingerprint_is_final_only_and_cache_remains_reusable(self):
        for phase in ("development", "final_test"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                study, checkpoint = root / "study", root / "checkpoint"
                study.mkdir()
                checkpoint.mkdir()
                (checkpoint / "model.safetensors").write_bytes(b"synthetic fixture, never loaded")
                config_path = root / "base.yaml"
                config_path.write_text("simulation: {}\nevaluation: {}\n")
                code_hashes = {}
                for name in ("learning.py", "simulation.py"):
                    path = root / "phone_to_panda" / name
                    path.parent.mkdir(exist_ok=True)
                    path.write_text("# synthetic code identity\n")
                    code_hashes[str(path.relative_to(root))] = evaluator.digest(path)
                settings = {"radius_m": .06, "seed": 7100, "episodes": 1}
                (study / "study_plan.json").write_text(json.dumps({phase: {"wide": settings}}))
                identity = {"checkpoint_files": {"model.safetensors": evaluator.digest(checkpoint / "model.safetensors")},
                            "n_action_steps": 5}
                if phase == "final_test":
                    sidecar = study / "selection_rule.json"
                    sidecar.write_text('{"rule": "Select before final testing"}')
                    (study / "candidate_lock.json").write_text(json.dumps({
                        "study_plan_sha256": evaluator.digest(study / "study_plan.json"),
                        "config_sha256": evaluator.digest(config_path), "source_sha256": code_hashes,
                        "reference_candidate": "candidate", "selected_candidate": "candidate", "contrast_roles": {},
                        "selection_sidecars_sha256": {sidecar.name: evaluator.digest(sidecar)},
                        "candidates": {"candidate": identity}}))
                config = {"simulation": {"orientation_control": "fixed_downward"}, "evaluation": {}}
                def record(config, checkpoint, output, *, episodes, seed):
                    output.mkdir()
                    report = {"kind": "trained_policy_physics_evaluation", "checkpoint": str(checkpoint),
                              "randomize_start_m": .06, "seed_start": seed,
                              "policy_settings": {"n_action_steps": 5}, "orientation_control": "fixed_downward",
                              "held_out_from_training_seeds": True, "video_recorded": False,
                              "episodes": [{"seed": seed, "success": True}], "successes": 1,
                              "success_rate": 1., "wall_seconds": 1.}
                    (output / "report.json").write_text(json.dumps(report))
                    return report
                args = ["--study", str(study), "--config", str(config_path), "--name", "candidate",
                        "--checkpoint", str(checkpoint), "--action-steps", "5", "--phase", phase]
                with patch.object(evaluator, "ROOT", root), \
                        patch.object(evaluator, "load_config", side_effect=lambda _: deepcopy(config)), \
                        patch.object(evaluator, "run_policy", side_effect=record) as run:
                    self.assertEqual(evaluator.main(args), 0)
                    self.assertEqual(evaluator.main(args), 0)
                    self.assertEqual(run.call_count, 1)
                manifest = json.loads((study / phase / "candidate/wide/manifest.json").read_text())
                if phase == "final_test":
                    self.assertEqual(manifest["base_config_sha256"], evaluator.digest(config_path))
                    self.assertEqual(manifest["candidate_lock_sha256"], evaluator.digest(study / "candidate_lock.json"))
                    lock_path = study / "candidate_lock.json"
                    lock = json.loads(lock_path.read_text())
                    lock["note"] = "Selection metadata changed after a final run"
                    lock_path.write_text(json.dumps(lock))
                    with patch.object(evaluator, "ROOT", root), \
                            patch.object(evaluator, "load_config", side_effect=lambda _: deepcopy(config)), \
                            patch.object(evaluator, "run_policy") as forbidden:
                        with self.assertRaisesRegex(ValueError, "Inputs changed"):
                            evaluator.main(args)
                        forbidden.assert_not_called()
                else:
                    self.assertNotIn("base_config_sha256", manifest)
                    self.assertNotIn("candidate_lock_sha256", manifest)


if __name__ == "__main__":
    unittest.main()
