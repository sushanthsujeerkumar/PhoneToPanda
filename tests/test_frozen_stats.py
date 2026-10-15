"""The adaptation wrapper must retain source scaling in both directions."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

try:
    import torch
    from safetensors.torch import save_file
except ModuleNotFoundError:
    torch = None

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_with_frozen_stats.py"
SPEC = importlib.util.spec_from_file_location("train_with_frozen_stats", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@unittest.skipIf(torch is None, "Requires optional LeRobot training dependencies")
class FrozenStatsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.states = []
        self.pipelines = []
        for config_name, registry_name, required in MODULE.PROCESSORS:
            state = {}
            for feature, size in required.items():
                state[f"{feature}.mean"] = torch.arange(size, dtype=torch.float32) / 10
                state[f"{feature}.std"] = torch.ones(size)
                state[f"{feature}.count"] = torch.tensor([4471.])
            state_file = registry_name + ".safetensors"
            save_file(state, self.source / state_file)
            config = {"steps": [{"registry_name": registry_name, "state_file": state_file,
                                  "config": {"norm_map": {"STATE": "MEAN_STD", "ACTION": "MEAN_STD"}}}]}
            (self.source / config_name).write_text(json.dumps(config))
            self.states.append(state)
            self.pipelines.append(SimpleNamespace(steps=[SimpleNamespace(state_dict=lambda s=state: s)]))
        self.factory = Mock(return_value=tuple(self.pipelines))
        self.saver = Mock()
        self.guard = MODULE.FrozenStatsGuard(self.factory, self.saver)

    def tearDown(self):
        self.temporary.cleanup()

    def test_removes_stats_only_and_does_not_mutate_arguments(self):
        overrides = {"dataset_stats": {"action": {"mean": [999]}},
                     "preprocessor_overrides": {"device_processor": {"device": "cpu"},
                                                "normalizer_processor": {"stats": {"new": 42}, "eps": 1e-8}},
                     "postprocessor_overrides": {"unnormalizer_processor": {"stats": {"new": 42}, "eps": 1e-8}}}
        original = copy.deepcopy(overrides)
        self.guard.make_processors(policy_cfg=None, pretrained_path=self.source, **overrides)
        passed = self.factory.call_args.kwargs
        self.assertNotIn("dataset_stats", passed)
        self.assertEqual(passed["preprocessor_overrides"]["device_processor"], {"device": "cpu"})
        self.assertEqual(passed["preprocessor_overrides"]["normalizer_processor"], {"eps": 1e-8})
        self.assertEqual(passed["postprocessor_overrides"]["unnormalizer_processor"], {"eps": 1e-8})
        self.assertEqual(overrides, original)

    def test_missing_source_stats_fails_before_factory(self):
        (self.source / "normalizer_processor.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "Missing valid source"):
            self.guard.make_processors(None, self.source)
        self.factory.assert_not_called()

    def test_mismatched_action_scaling_fails(self):
        self.states[1]["action.mean"][0] = 9
        save_file(self.states[1], self.source / "unnormalizer_processor.safetensors")
        with self.assertRaisesRegex(ValueError, "disagree"):
            self.guard.make_processors(None, self.source)

    def test_factory_cannot_silently_use_new_statistics(self):
        self.states[0]["action.mean"][0] = 9
        with self.assertRaisesRegex(ValueError, "not preserved"):
            self.guard.make_processors(None, self.source)

    def test_save_verifies_serialization_and_records_provenance(self):
        self.guard.make_processors(None, self.source)
        checkpoint = self.root / "checkpoints/001000"
        self.saver.side_effect = lambda **kwargs: shutil.copytree(self.source, checkpoint / "pretrained_model")
        self.guard.save(checkpoint_dir=checkpoint, step=1000,
                        preprocessor=self.pipelines[0], postprocessor=self.pipelines[1])
        metadata = json.loads((checkpoint / "pretrained_model/frozen_normalization.json").read_text())
        self.assertTrue(metadata["saved_statistics_verified_equal"])
        self.assertFalse(metadata["dataset_statistics_modified"])
        self.assertEqual(metadata["checkpoint_step"], 1000)
        self.assertEqual(metadata["source_checkpoint"], str(self.source))
        self.assertEqual(len(metadata["source_files_sha256"]), 4)

    def test_real_normalizer_reloads_source_scaling_on_cpu(self):
        try:
            from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
            from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep
        except ModuleNotFoundError:
            self.skipTest("Requires optional LeRobot processor dependencies")
        feature = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
        norm_map = {FeatureType.ACTION: NormalizationMode.MEAN_STD}
        normalizer = NormalizerProcessorStep(features=feature, norm_map=norm_map, device="cpu")
        unnormalizer = UnnormalizerProcessorStep(features=feature, norm_map=norm_map, device="cpu")
        normalizer.load_state_dict(self.states[0])
        unnormalizer.load_state_dict(self.states[1])
        for key in ("action.mean", "action.std"):
            torch.testing.assert_close(normalizer.state_dict()[key], unnormalizer.state_dict()[key], rtol=0, atol=0)
        # Explicit new stats reproduce upstream's scale replacement, confirming
        # why the wrapper must remove the overrides before state loading.
        replacement = NormalizerProcessorStep(features=feature, norm_map=norm_map,
                                              stats={"action": {"mean": [9.] * 7, "std": [2.] * 7}}, device="cpu")
        replacement.load_state_dict(self.states[0])
        self.assertFalse(torch.equal(replacement.state_dict()["action.mean"], normalizer.state_dict()["action.mean"]))

    def test_count_scalarization_only_allows_identical_single_metadata_value(self):
        self.assertTrue(MODULE.same_statistic("action.count", torch.tensor(4471.), torch.tensor([4471.])))
        self.assertFalse(MODULE.same_statistic("action.count", torch.tensor(4472.), torch.tensor([4471.])))
        self.assertFalse(MODULE.same_statistic("action.count", torch.tensor([[4471.]]), torch.tensor([4471.])))
        self.assertFalse(MODULE.same_statistic("action.mean", torch.tensor(.1), torch.tensor([.1])))
        self.assertFalse(MODULE.same_statistic("action.std", torch.tensor([1.001]), torch.tensor([1.])))
        self.assertTrue(MODULE.same_statistic("episode_index.max", torch.tensor(19.), torch.tensor([19.])))
        self.assertFalse(MODULE.same_statistic("episode_index.max", torch.tensor(20.), torch.tensor([19.])))
        self.assertFalse(MODULE.same_statistic("observation.images.image.mean", torch.tensor(.1), torch.tensor([.1])))

    def test_real_pipeline_transfer_call_save_and_reload_preserve_transform(self):
        from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
        from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep

        state_feature = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,))}
        action_feature = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
        norm_map = {FeatureType.STATE: NormalizationMode.MEAN_STD, FeatureType.ACTION: NormalizationMode.MEAN_STD}
        norm = NormalizerProcessorStep(features={**state_feature, **action_feature}, norm_map=norm_map, device="cpu")
        unnorm = UnnormalizerProcessorStep(features=action_feature, norm_map=norm_map, device="cpu")
        norm.load_state_dict(self.states[0])
        unnorm.load_state_dict(self.states[1])
        original_pre = PolicyProcessorPipeline(steps=[norm], name="policy_preprocessor")
        original_post = PolicyProcessorPipeline(steps=[unnorm], name="policy_postprocessor",
                                                to_transition=policy_action_to_transition,
                                                to_output=transition_to_policy_action)
        source = self.root / "real_source"
        original_pre.save_pretrained(source)
        original_post.save_pretrained(source)
        config = SmolVLAConfig(device="cpu", input_features=state_feature, output_features=action_feature)
        def save_processors(*, checkpoint_dir, preprocessor, postprocessor, **kwargs):
            preprocessor.save_pretrained(checkpoint_dir / "pretrained_model")
            postprocessor.save_pretrained(checkpoint_dir / "pretrained_model")
        guard = MODULE.FrozenStatsGuard(make_pre_post_processors, save_processors)
        pre, post = guard.make_processors(config, source)
        self.assertEqual(tuple(pre.steps[0].state_dict()["action.count"].shape), (1,))
        # The first real GPU batch calls the same upstream .to() when it moves
        # CPU statistics. Reproduce that reconstruction on CPU, without a GPU.
        pre.steps[0].to(device="cpu")
        post.steps[0].to(device="cpu")
        self.assertEqual(tuple(pre.steps[0].state_dict()["action.count"].shape), ())
        batch = {"observation.state": torch.ones(2, 8), "action": torch.ones(2, 20, 7)}
        transformed = pre(batch)
        action = post(transformed["action"])
        torch.testing.assert_close(action, batch["action"], rtol=0, atol=0)
        guard.verify((pre, post))
        destination = self.root / "saved/000002"
        guard.save(checkpoint_dir=destination, step=2, preprocessor=pre, postprocessor=post)
        reloaded_pre, reloaded_post = make_pre_post_processors(config, destination / "pretrained_model")
        guard.verify((reloaded_pre, reloaded_post))
        reloaded = reloaded_pre(batch)
        torch.testing.assert_close(transformed["action"], reloaded["action"], rtol=0, atol=0)
        torch.testing.assert_close(action, reloaded_post(reloaded["action"]), rtol=0, atol=0)
        reloaded_pre.steps[0]._tensor_stats["action"]["mean"][0] += .01
        with self.assertRaisesRegex(ValueError, "not preserved"):
            guard.verify((reloaded_pre, reloaded_post))

    def test_actual_source_checkpoint_all_statistics_survive_cpu_transfer(self):
        from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep

        source = SCRIPT.parents[1] / "checkpoints/main"
        if not source.is_dir():
            self.skipTest("Requires the existing local PhoneToPanda source checkpoint")
        _, expected = MODULE.read_source(source)
        pipelines = []
        changed_shapes = []
        for (filename, _, _), specification, cls in zip(
                MODULE.PROCESSORS, expected, (NormalizerProcessorStep, UnnormalizerProcessorStep), strict=True):
            config = json.loads((source / filename).read_text())["steps"][specification["index"]]["config"]
            processor = cls(**config, device="cpu")
            processor.load_state_dict(specification["state"])
            processor.to(device="cpu")
            actual = processor.state_dict()
            for key, saved in specification["state"].items():
                self.assertTrue(MODULE.same_statistic(key, actual[key], saved), key)
                if actual[key].shape != saved.shape:
                    changed_shapes.append(key)
            pipelines.append(processor)
        self.assertIn("action.count", changed_shapes)
        self.assertIn("episode_index.max", changed_shapes)
        self.assertNotIn("action.mean", changed_shapes)
        self.assertNotIn("observation.state.std", changed_shapes)


if __name__ == "__main__":
    unittest.main()
