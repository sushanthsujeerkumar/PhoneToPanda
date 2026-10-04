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


if __name__ == "__main__":
    unittest.main()
