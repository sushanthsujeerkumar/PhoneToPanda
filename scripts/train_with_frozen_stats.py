"""Run LeRobot training with a local checkpoint's normalization kept unchanged.

Accepts the same arguments as ``python -m lerobot.scripts.lerobot_train``.
The installed trainer otherwise replaces the checkpoint's observation and action
statistics with those of the new dataset. This wrapper keeps the saved statistics
without changing dataset metadata, training batches, or the upstream optimizer.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


PROCESSORS = (
    ("policy_preprocessor.json", "normalizer_processor", {"observation.state": 8, "action": 7}),
    ("policy_postprocessor.json", "unnormalizer_processor", {"action": 7}),
)
ANCILLARY_FEATURES = frozenset({"episode_index", "frame_index", "index", "task_index", "timestamp"})


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def same_statistic(key, actual, expected) -> bool:
    """Allow upstream's harmless scalarization of single ancillary statistics.

    LeRobot reconstructs ``self.stats`` through ``from_tensor_to_numpy``, which
    turns one-element tensors into scalars. Its first device transfer therefore
    changes one-element metadata from shape (1,) to (). Counts and episode/frame/
    task/timestamp indexes are not inputs to this policy's normalization transform.
    Action/state/image transform tensors retain exact source shapes and values.
    """
    import torch

    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    if torch.equal(actual, expected):
        return True
    ancillary = key.endswith(".count") or key.rsplit(".", 1)[0] in ANCILLARY_FEATURES
    return (ancillary and tuple(actual.shape) in ((), (1,))
            and tuple(expected.shape) in ((), (1,))
            and torch.equal(actual.reshape(1), expected.reshape(1)))


def read_source(checkpoint: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate the saved PhoneToPanda processors without loading model weights."""
    import torch
    from safetensors.torch import load_file

    source = Path(checkpoint).resolve()
    if not source.is_dir():
        raise ValueError("Frozen normalization requires a local completed checkpoint directory.")
    files = {}
    processors = []
    for config_name, registry_name, required in PROCESSORS:
        config_path = source / config_name
        if not config_path.is_file():
            raise ValueError(f"Missing source processor configuration: {config_path}")
        config = json.loads(config_path.read_text())
        matches = [(index, entry) for index, entry in enumerate(config.get("steps", []))
                   if entry.get("registry_name") == registry_name]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one saved {registry_name} in {config_path}")
        index, entry = matches[0]
        filename = entry.get("state_file")
        if not filename or Path(filename).name != filename or not (source / filename).is_file():
            raise ValueError(f"Missing valid source normalization state for {registry_name}")
        state = load_file(source / filename, device="cpu")
        for feature, size in required.items():
            feature_type = "STATE" if feature == "observation.state" else "ACTION"
            if entry.get("config", {}).get("norm_map", {}).get(feature_type) != "MEAN_STD":
                raise ValueError(f"Expected saved MEAN_STD normalization for {feature}")
            for statistic in ("mean", "std"):
                key = f"{feature}.{statistic}"
                value = state.get(key)
                if value is None or tuple(value.shape) != (size,) or not torch.isfinite(value).all():
                    raise ValueError(f"Missing or invalid saved normalization tensor: {key}")
                if statistic == "std" and (value < 0).any():
                    raise ValueError(f"Saved normalization std cannot be negative: {key}")
        files[config_name] = _digest(config_path)
        files[filename] = _digest(source / filename)
        processors.append({"index": index, "state": state, "registry_name": registry_name})
    for statistic in ("mean", "std"):
        key = f"action.{statistic}"
        if not torch.equal(processors[0]["state"][key], processors[1]["state"][key]):
            raise ValueError(f"Source action normalizer and unnormalizer disagree: {key}")
    for name in ("config.json", "frozen_normalization.json"):
        if (source / name).is_file():
            files[name] = _digest(source / name)
    manifest = {"kind": "preserved_checkpoint_normalization", "source_checkpoint": str(source),
                "source_files_sha256": files, "dataset_statistics_modified": False,
                "description": "Training and inference use the source checkpoint's observation/action statistics."}
    return manifest, processors


class FrozenStatsGuard:
    """Remove only statistics overrides and verify the values actually loaded/saved."""

    def __init__(self, processor_factory, save_checkpoint):
        self.processor_factory = processor_factory
        self.save_checkpoint = save_checkpoint
        self.manifest = None
        self.processors = None

    def verify(self, pipelines) -> None:
        import torch

        if self.processors is None:
            raise RuntimeError("Frozen processors have not been initialized.")
        for pipeline, expected in zip(pipelines, self.processors, strict=True):
            state = pipeline.steps[expected["index"]].state_dict()
            if set(state) != set(expected["state"]):
                raise ValueError(f"Loaded {expected['registry_name']} statistics keys changed.")
            for key, value in expected["state"].items():
                if not same_statistic(key, state[key], value):
                    raise ValueError(f"Source normalization was not preserved: {key}")

    def make_processors(self, policy_cfg, pretrained_path=None, **kwargs):
        if pretrained_path is None:
            raise ValueError("Frozen normalization requires --policy.path pointing to a local checkpoint.")
        self.manifest, self.processors = read_source(pretrained_path)
        # Remove both routes for new dataset statistics so input normalization
        # and output action unnormalization use the checkpoint's saved scales.
        # Copy options to preserve caller settings and dataset metadata;
        # model weights still train normally.
        options = copy.deepcopy(kwargs)
        options.pop("dataset_stats", None)
        for group, registry_name in (("preprocessor_overrides", "normalizer_processor"),
                                     ("postprocessor_overrides", "unnormalizer_processor")):
            options.setdefault(group, {}).setdefault(registry_name, {}).pop("stats", None)
        pipelines = self.processor_factory(policy_cfg=policy_cfg, pretrained_path=pretrained_path, **options)
        self.verify(pipelines)
        print(f"FROZEN NORMALIZATION: preserving source observation/action statistics from {pretrained_path}", flush=True)
        return pipelines

    def save(self, *, checkpoint_dir, preprocessor, postprocessor, **kwargs):
        self.verify((preprocessor, postprocessor))
        result = self.save_checkpoint(checkpoint_dir=checkpoint_dir, preprocessor=preprocessor,
                                      postprocessor=postprocessor, **kwargs)
        destination = Path(checkpoint_dir) / "pretrained_model"
        # Reopen the serialized values, so the assertion covers checkpoint reload.
        saved_manifest, saved_processors = read_source(destination)
        import torch
        for saved, expected in zip(saved_processors, self.processors, strict=True):
            if set(saved["state"]) != set(expected["state"]) or any(
                    not same_statistic(key, saved["state"][key], value) for key, value in expected["state"].items()):
                raise ValueError("Serialized checkpoint normalization differs from the source.")
        manifest = {**self.manifest, "checkpoint_step": kwargs.get("step"),
                    "saved_statistics_verified_equal": True,
                    "normalization_transform_tensors_verified_exact": True,
                    "ancillary_metadata_policy": "Single count or episode/frame/task/index/timestamp metadata may have shape () or (1,) with identical value; action/state/image normalization transform tensors retain exact source shape and values.",
                    "saved_processor_files_sha256": {key: value for key, value in saved_manifest["source_files_sha256"].items()
                                                      if key.startswith("policy_")}}
        (destination / "frozen_normalization.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return result


def main() -> None:
    from lerobot.scripts import lerobot_train

    guard = FrozenStatsGuard(lerobot_train.make_pre_post_processors, lerobot_train.save_checkpoint)
    lerobot_train.make_pre_post_processors = guard.make_processors
    lerobot_train.save_checkpoint = guard.save
    try:
        lerobot_train.main()
    finally:
        lerobot_train.make_pre_post_processors = guard.processor_factory
        lerobot_train.save_checkpoint = guard.save_checkpoint


if __name__ == "__main__":
    main()
