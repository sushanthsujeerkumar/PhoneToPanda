"""Checks the provenance gate and real training/evaluation interfaces."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from phone_to_panda.learning import _validate_episode, export_dataset, train, training_command


class DataLineageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.video = self.root / "my_recording.mp4"
        self.video.write_bytes(b"fixture, not a genuine video; only hashes are under test")
        self.path = self.root / "episode.npz"
        self.arrays = {"state": np.zeros((2, 8), np.float32), "action": np.zeros((2, 7), np.float32),
                       "image": np.zeros((2, 256, 256, 3), np.uint8), "image2": np.zeros((2, 256, 256, 3), np.uint8)}
        np.savez(self.path, **self.arrays)
        self.metadata = {"source": {"kind": "personal_video", "video": self.video.name,
                         "sha256": hashlib.sha256(self.video.read_bytes()).hexdigest()},
                         "success": True, "seed": 1, "task": "Put the cube in the box."}
        self.save_metadata()

    def tearDown(self):
        self.tmp.cleanup()

    def save_metadata(self):
        self.path.with_suffix(".json").write_text(json.dumps(self.metadata))

    def test_modified_recording_breaks_lineage(self):
        _validate_episode(self.path, self.root)
        self.video.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            _validate_episode(self.path, self.root)

    def test_setup_demo_is_rejected_before_export_writes(self):
        self.metadata["source"]["kind"] = "setup_test"
        self.save_metadata()
        destination = self.root / "dataset"
        with patch("phone_to_panda.learning._create_dataset") as writer:
            with self.assertRaisesRegex(ValueError, "setup/synthetic"):
                export_dataset({"_root": str(self.root)}, self.root, destination)
            writer.assert_not_called()
        self.assertFalse(destination.exists())

    def test_unsuccessful_and_out_of_bounds_actions_rejected(self):
        self.metadata["success"] = False
        self.save_metadata()
        with self.assertRaisesRegex(ValueError, "unsuccessful"):
            _validate_episode(self.path, self.root)
        self.metadata["success"] = True
        self.save_metadata()
        self.arrays["action"][0, 0] = 1.5
        np.savez(self.path, **self.arrays)
        with self.assertRaisesRegex(ValueError, "normalized"):
            _validate_episode(self.path, self.root)

    def test_train_rejects_exported_setup_fixture(self):
        destination = self.root / "dataset" / "meta"
        destination.mkdir(parents=True)
        (destination / "phone_to_panda_provenance.json").write_text(json.dumps({
            "provenance": [{"source": {"kind": "setup_test"}, "success": True}]}))
        with self.assertRaisesRegex(ValueError, "forbidden"):
            train({"_root": str(self.root), "paths": {"dataset": "dataset"}})


class TrainingInterfaceTests(unittest.TestCase):
    def test_feature_overrides_clear_pretrained_camera_names(self):
        # Exercise the real configuration parser: empty dicts merge with the
        # original features instead of removing the pretrained robot's cameras.
        try:
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.configs.types import FeatureType, PolicyFeature
            from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        except ModuleNotFoundError:
            self.skipTest("Requires the optional LeRobot training environment")
        config = SmolVLAConfig(device="cpu", input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        }, output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))})
        command = training_command({})
        overrides = [arg.replace("--policy.", "--", 1) for arg in command
                     if arg.startswith(("--policy.input_features=", "--policy.output_features="))]
        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            loaded = PreTrainedConfig.from_pretrained(directory, cli_overrides=overrides)
        self.assertIsNone(loaded.input_features)
        self.assertIsNone(loaded.output_features)

    def test_dry_run_needs_no_data_and_preserves_paths_with_spaces(self):
        config = {"_root": "/tmp/project with spaces", "training": {"output_dir": "runs/my run"}}
        with patch("phone_to_panda.learning.subprocess.run") as execute:
            result = train(config, dry_run=True)
            execute.assert_not_called()
        command = result["command"]
        self.assertIn("--output_dir=/tmp/project with spaces/runs/my run", command)
        self.assertIn("--policy.input_features=null", command)
        self.assertIn("--policy.output_features=null", command)
        self.assertIn("--policy.push_to_hub=false", command)
        self.assertIn("--policy.path=lerobot/smolvla_base", command)
        self.assertFalse(result["training_started"])

    def test_resume_loads_full_training_state(self):
        command = training_command({"_root": "/tmp/project"}, resume=True)
        self.assertIn("--config_path=/tmp/project/runs/train/checkpoints/last/pretrained_model/train_config.json", command)
        self.assertIn("--resume=true", command)
        self.assertFalse(any(arg.startswith("--policy.path") for arg in command))

    def test_preserved_normalization_uses_wrapper_for_training_and_resume(self):
        config = {"_root": "/tmp/project with spaces", "training": {"preserve_normalization": True}}
        for resume in (False, True):
            command = training_command(config, resume=resume)
            self.assertEqual(command[1], "/tmp/project with spaces/scripts/train_with_frozen_stats.py")
            self.assertNotIn("lerobot.scripts.lerobot_train", command)
        self.assertEqual(training_command({})[1:3], ["-m", "lerobot.scripts.lerobot_train"])


class EvaluationSettingsTests(unittest.TestCase):
    def test_action_horizon_bounds_and_default(self):
        from phone_to_panda.learning import _evaluation_action_steps
        self.assertEqual(_evaluation_action_steps(5, 20), 5)
        self.assertEqual(_evaluation_action_steps(5, 20, 1), 1)
        self.assertEqual(_evaluation_action_steps(5, 20, 20), 20)
        for invalid in (0, -1, 21, 1.5, True, "1"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "n_action_steps"):
                _evaluation_action_steps(5, 20, invalid)

    def test_override_is_applied_before_model_construction_without_writing_checkpoint(self):
        from types import SimpleNamespace
        from phone_to_panda.learning import ACTION, IMAGES, STATE, _load_policy
        try:
            from lerobot.configs.types import FeatureType, PolicyFeature
            from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        except ModuleNotFoundError:
            self.skipTest("Requires LeRobot configuration classes")
        saved = SmolVLAConfig(device="cpu", n_action_steps=5, chunk_size=20,
            input_features={STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
                            **{key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)) for key in IMAGES}},
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))})
        captured = []
        def construct(path, *, config, strict):
            captured.append(config.n_action_steps)
            instance = SimpleNamespace(config=config)
            instance.to = lambda device: instance
            instance.eval = lambda: instance
            return instance
        with tempfile.TemporaryDirectory() as directory:
            saved.save_pretrained(directory)
            path = Path(directory)
            before = (path / "config.json").read_bytes()
            with patch("lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy.from_pretrained", side_effect=construct), \
                 patch("lerobot.policies.factory.make_pre_post_processors", return_value=(None, None)):
                policy, _, _ = _load_policy(path, "cpu", n_action_steps=1)
                self.assertEqual(policy.config.n_action_steps, 1)
                _load_policy(path, "cpu")
            self.assertEqual(captured, [1, 5])
            self.assertEqual((path / "config.json").read_bytes(), before)

    def test_evaluation_repeats_rng_actions_and_keeps_old_trace_fields(self):
        import random
        from types import SimpleNamespace
        import torch
        from phone_to_panda.learning import CONVENTION, IMAGES, STATE, run_policy
        class FakePolicy:
            def __init__(self, steps):
                self.config = SimpleNamespace(n_action_steps=steps, chunk_size=20, num_steps=10)
                self.reset()
            def reset(self):
                self.remaining = 0
            def _check_get_actions_condition(self):
                return self.remaining == 0
            def select_action(self, obs):
                if self.remaining == 0:
                    self.remaining = self.config.n_action_steps
                self.remaining -= 1
                return torch.rand(7)
        initial_rng = []
        class FakeEnv:
            def __init__(self, config, seed):
                initial_rng.append((random.random(), float(np.random.rand()), float(torch.rand(1))))
                self.count, self._max_cube_height = 0, .92
                self.cube_position = np.array([0., 0., .92])
            def reset(self):
                return {STATE: np.zeros(8, np.float32), **{key: np.zeros((2, 2, 3), np.uint8) for key in IMAGES}}
            def step(self, action):
                self.count += 1
                if self.count >= 2:
                    self._max_cube_height = 1.02
                return self.reset(), 0., self.count == 4, {"applied_action": action.tolist()}
            def success(self):
                return self.count == 4
            def grasp_diagnostics(self):
                return {"gripper_contact": 1 <= self.count <= 2,
                        "bilateral_contact": self.count == 2, "firm_bilateral_grasp": self.count == 2,
                        "lifted": self.count == 2, "ever_lifted": self.count >= 2}
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "model.safetensors").write_bytes(b"Test fixture only; never loaded")
            (checkpoint / "train_config.json").write_text("{}")
            (checkpoint / "config.json").write_text(json.dumps({"n_action_steps": 5, "chunk_size": 20}))
            (checkpoint / "phone_to_panda_provenance.json").write_text(json.dumps({
                "orientation_control": "fixed_downward", "action_convention": CONVENTION, "fps": 20,
                "provenance": [{"seed": 0, "simulation_settings": {"orientation_control": "fixed_downward"}}]}))
            config = {"task": "Place cube", "training": {"device": "cpu"},
                      "simulation": {"orientation_control": "fixed_downward", "control_hz": 20, "image_size": 256, "horizon": 4},
                      "evaluation": {"n_action_steps": 1, "save_trace": True, "save_video": False}}
            def load(path, device, n_action_steps=None):
                return FakePolicy(n_action_steps or 5), lambda x: x, lambda x: x
            with patch("phone_to_panda.learning._load_policy", side_effect=load), \
                 patch("phone_to_panda.simulation.make_env", side_effect=FakeEnv):
                first = run_policy(config, checkpoint, root / "a", episodes=1, seed=3010)
                run_policy(config, checkpoint, root / "b", episodes=1, seed=3010)
                config["evaluation"].pop("n_action_steps")
                default = run_policy(config, checkpoint, root / "c", episodes=1, seed=3010)
            a = json.loads((root / "a/episode_000_trace.json").read_text())
            b = json.loads((root / "b/episode_000_trace.json").read_text())
            self.assertEqual(initial_rng[0], initial_rng[1])
            self.assertEqual([t["policy_action"] for t in a], [t["policy_action"] for t in b])
            self.assertTrue(all(t["prediction_refresh"] for t in a))
            self.assertEqual(first["policy_settings"]["n_action_steps"], 1)
            self.assertEqual(default["policy_settings"]["n_action_steps"], 5)
            self.assertEqual(first["inference_timing"]["prediction_calls"], 4)
            self.assertEqual(default["inference_timing"]["prediction_calls"], 1)
            self.assertEqual(first["episodes"][0]["first_events"], {"contact_step": 0, "bilateral_contact_step": 1,
                             "firm_grasp_step": 1, "lift_step": 1, "release_after_lift_step": 2})
            old_keys = {"step", "observation_state", "cube_xyz", "policy_action", "applied_action", "next_cube_xyz", "success"}
            self.assertTrue(old_keys.issubset(a[0]))
            self.assertEqual(a[0]["episode_seed"], 3010)
            self.assertFalse(a[0]["diagnostics"]["gripper_contact"])
            self.assertTrue(a[0]["next_diagnostics"]["gripper_contact"])


if __name__ == "__main__":
    unittest.main()
