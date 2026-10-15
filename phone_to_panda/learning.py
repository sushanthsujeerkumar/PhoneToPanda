"""Local LeRobot export, SmolVLA fine-tuning, and simulation evaluation.

Imports of the ML stack are lazy so annotation tools remain lightweight.
The smoke test uses a tiny random model to check the API, not task performance.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np

from .simulation import IMU_PILOT_KIND, validate_imu_pilot_source

STATE = "observation.state"
IMAGES = ("observation.images.image", "observation.images.image2")
ACTION = "action"
CONVENTION = "robosuite_OSC_POSE_delta_world_xyz_rotvec_gripper"
STATE_NAMES = ["eef_x", "eef_y", "eef_z", "eef_rx", "eef_ry", "eef_rz", "finger_left", "finger_right"]
ACTION_NAMES = ["delta_x", "delta_y", "delta_z", "delta_rx", "delta_ry", "delta_rz", "gripper"]


def controller_mode(provenance: dict) -> str:
    """A checkpoint must run with the same rotation controller as its data."""
    modes = {ep.get("simulation_settings", {}).get("orientation_control", "policy")
             for ep in provenance.get("provenance", [])}
    if len(modes) > 1:
        raise ValueError("Do not mix learned-rotation and fixed-downward demonstrations in one dataset.")
    mode = next(iter(modes), provenance.get("orientation_control", "policy"))
    if mode not in ("policy", "fixed_downward"):
        raise ValueError(f"Unsupported orientation controller: {mode}")
    return mode


def validate_controller_contract(config: dict, provenance: dict) -> str:
    mode = controller_mode(provenance)
    if config.get("simulation", {}).get("orientation_control", "policy") != mode:
        raise ValueError("The configured wrist controller differs from the training demonstrations. Use the saved run configuration.")
    return mode


def validate_checkpoint(config: dict, checkpoint: Path, seed: int):
    if not (checkpoint / "model.safetensors").is_file():
        raise ValueError("Choose the pretrained_model directory of a completed training checkpoint.")
    if (checkpoint / "SETUP_TEST_ONLY.json").exists():
        raise ValueError("A random setup-test model cannot be shown as a trained policy.")
    saved = json.loads((checkpoint / "train_config.json").read_text())
    source = checkpoint / "phone_to_panda_provenance.json"
    if not source.is_file():
        source = Path(saved["dataset"]["root"]) / "meta/phone_to_panda_provenance.json"
    provenance = json.loads(source.read_text())
    validate_controller_contract(config, provenance)
    if provenance.get("action_convention") != CONVENTION:
        raise ValueError("The training action convention does not match the simulator.")
    sim = config.get("simulation", {})
    if int(sim.get("control_hz", 20)) != provenance.get("fps"):
        raise ValueError("Simulation control rate must match the training dataset.")
    if int(sim.get("image_size", 256)) != 256 or int(sim.get("horizon", 400)) < 1:
        raise ValueError("Use image_size 256 and a positive simulation horizon.")
    if seed in {ep.get("seed") for ep in provenance.get("provenance", [])}:
        raise ValueError("Choose a test seed different from the training demonstration seeds.")


def _root(config: dict) -> Path:
    return Path(config.get("_root", Path(__file__).resolve().parents[1])).resolve()


def _path(config: dict, name: str, default: str) -> Path:
    value = Path(config.get("paths", {}).get(name, default))
    return value if value.is_absolute() else _root(config) / value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_episode(path: Path, root: Path) -> tuple[dict[str, np.ndarray], dict]:
    metadata_path = path.with_suffix(".json")
    if not metadata_path.is_file():
        raise ValueError(f"Missing provenance sidecar: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    source = metadata.get("source", {})
    if source.get("kind") not in {"personal_video", IMU_PILOT_KIND}:
        raise ValueError(f"{path.name}: only personal_video or explicitly verified IMU-pilot demonstrations may enter training; setup/synthetic data rejected")
    if metadata.get("success") is not True:
        raise ValueError(f"{path.name}: unsuccessful or unverified demonstration rejected")
    if source.get("kind") == IMU_PILOT_KIND:
        validate_imu_pilot_source(source, root, metadata.get("trajectory_quality", {}))
    sha256 = source.get("sha256", "")
    if not isinstance(source.get("video"), str) or not isinstance(sha256, str):
        raise ValueError(f"{path.name}: missing source video or SHA256 provenance")
    video = Path(source["video"])
    if not video.is_absolute():
        video = root / video
    if not video.is_file() or len(sha256) != 64 or _hash(video) != sha256:
        raise ValueError(f"{path.name}: source video missing or SHA256 does not match; preserve the original recording")
    with np.load(path, allow_pickle=False) as raw:
        if not {"state", "image", "image2", "action"}.issubset(raw.files):
            raise ValueError(f"{path.name}: missing aligned state/image/image2/action arrays")
        arrays = {key: raw[key].copy() for key in ("state", "image", "image2", "action")}
    frames = len(arrays["action"])
    if frames < 2:
        raise ValueError(f"{path.name}: need at least two aligned observation/action frames")
    expected = {"state": (frames, 8), "action": (frames, 7), "image": (frames, 256, 256, 3), "image2": (frames, 256, 256, 3)}
    for key, shape in expected.items():
        arr = arrays[key]
        dtype = np.uint8 if key.startswith("image") else np.float32
        if arr.shape != shape or arr.dtype != dtype:
            raise ValueError(f"{path.name}: {key} must be {dtype.__name__} with shape {shape}, got {arr.dtype} {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{path.name}: non-finite {key}")
    if np.max(np.abs(arrays["action"])) > 1.00001:
        raise ValueError(f"{path.name}: OSC actions must be normalized to [-1, 1]")
    if not metadata.get("task"):
        raise ValueError(f"{path.name}: missing language task")
    return arrays, metadata


def _create_dataset(episodes: Iterable[tuple[dict[str, np.ndarray], dict]], destination: Path, repo_id: str, fps: int) -> dict:
    """Internal writer; caller is responsible for provenance. Images are lossless PNG."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if destination.exists():
        raise FileExistsError(f"Dataset already exists: {destination}. Choose a new output path; no data is overwritten.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    features = {
        STATE: {"dtype": "float32", "shape": (8,), "names": STATE_NAMES},
        ACTION: {"dtype": "float32", "shape": (7,), "names": ACTION_NAMES},
        **{key: {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channels"]} for key in IMAGES},
    }
    dataset = LeRobotDataset.create(repo_id=repo_id, root=destination, fps=fps,
                                   robot_type="panda", features=features, use_videos=False,
                                   image_writer_threads=0, video_backend="pyav")
    provenance = []
    try:
        for arrays, metadata in episodes:
            for index in range(len(arrays["action"])):
                dataset.add_frame({STATE: arrays["state"][index], ACTION: arrays["action"][index],
                                   IMAGES[0]: arrays["image"][index], IMAGES[1]: arrays["image2"][index],
                                   "task": metadata["task"]})
            dataset.save_episode(parallel_encoding=False)
            provenance.append({"episode_index": len(provenance), "frames": len(arrays["action"]), **metadata})
    finally:
        dataset.finalize()
    summary = {"repo_id": repo_id, "root": str(destination), "episodes": len(provenance),
               "frames": sum(episode["frames"] for episode in provenance), "fps": fps,
               "action_convention": CONVENTION, "uploaded": False, "provenance": provenance}
    summary["orientation_control"] = controller_mode(summary)
    _write_json(destination / "meta" / "phone_to_panda_provenance.json", summary)
    return summary


def export_dataset(config: dict, demos_dir: str | Path | None = None, output_dir: str | Path | None = None) -> dict:
    """Validate every source before writing; never upload and never overwrite."""
    source = Path(demos_dir) if demos_dir else _path(config, "robot", "data/robot/main")
    destination = Path(output_dir) if output_dir else _path(config, "dataset", "data/datasets/main")
    files = sorted(source.glob("*.npz"))
    if not files:
        raise ValueError(f"No robot demonstrations in {source}. Record, annotate and successfully retarget your own videos first.")
    # Validate the complete collection first, then export one episode at a time.
    # Holding all camera arrays for a full dataset would exhaust laptop memory.
    fps = int(config.get("simulation", {}).get("control_hz", 20))
    episode_metadata = []
    for path in files:
        arrays, metadata = _validate_episode(path, _root(config))
        del arrays
        if int(metadata.get("fps", fps)) != fps:
            raise ValueError(f"{path.name}: recorded control rate differs from simulation.control_hz; use the recording rate")
        episode_metadata.append(metadata)
    validate_controller_contract(config, {"provenance": episode_metadata})
    episodes = (_validate_episode(path, _root(config)) for path in files)
    summary = _create_dataset(episodes, destination, config.get("training", {}).get("repo_id", "local/phone_to_panda"), fps)
    return summary


def training_command(config: dict, dataset_dir: str | Path | None = None,
                     output_dir: str | Path | None = None, resume: bool = False) -> list[str]:
    """Construct argv (no shell). Dry runs need neither data nor network access."""
    settings = config.get("training", {})
    dataset = Path(dataset_dir) if dataset_dir else _path(config, "dataset", "data/datasets/main")
    output = Path(output_dir or settings.get("output_dir", "runs/train"))
    if not output.is_absolute():
        output = _root(config) / output
    executable = ([sys.executable, str(_root(config) / "scripts/train_with_frozen_stats.py")]
                  if settings.get("preserve_normalization", False)
                  else [sys.executable, "-m", "lerobot.scripts.lerobot_train"])
    if resume:
        checkpoint_config = output / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
        return [*executable, f"--config_path={checkpoint_config}", "--resume=true"]
    steps = int(settings.get("steps", 10000))
    if steps < 1:
        raise ValueError("training.steps must be positive")
    warmup = min(int(settings.get("warmup_steps", 100)), max(1, steps // 10))
    args = [*executable,
            f"--policy.path={settings.get('policy_path', 'lerobot/smolvla_base')}",
            f"--dataset.repo_id={settings.get('repo_id', 'local/phone_to_panda')}",
            f"--dataset.root={dataset}", "--dataset.video_backend=pyav",
            f"--output_dir={output}", "--job_name=phone_to_panda",
            f"--batch_size={int(settings.get('batch_size', 1))}", f"--steps={steps}",
            f"--save_freq={int(settings.get('save_freq', 1000))}",
            f"--num_workers={int(settings.get('num_workers', 0))}",
            f"--seed={int(settings.get('seed', 42))}", "--eval_freq=0", "--log_freq=10",
            f"--policy.device={settings.get('device', 'cuda')}",
            f"--policy.use_amp={str(bool(settings.get('use_amp', True))).lower()}",
            "--policy.push_to_hub=false", "--wandb.enable=false",
            # draccus merges dictionary overrides: {} retains the base model's
            # camera names. null clears them so make_policy infers this dataset.
            "--policy.input_features=null", "--policy.output_features=null", "--policy.load_vlm_weights=false",
            "--policy.freeze_vision_encoder=true", "--policy.train_expert_only=true", "--policy.train_state_proj=true",
            f"--policy.n_action_steps={int(settings.get('n_action_steps', 10))}",
            f"--policy.scheduler_warmup_steps={warmup}",
            f"--policy.scheduler_decay_steps={max(steps, warmup + 1)}"]
    for key in ("chunk_size", "num_steps", "optimizer_lr", "resize_imgs_with_padding"):
        if key in settings:
            args.append(f"--policy.{key}={json.dumps(settings[key])}")
    return args


def train(config: dict, dry_run: bool = False, resume: bool = False) -> dict:
    command = training_command(config, resume=resume)
    if dry_run:
        return {"dry_run": True, "command": command, "training_started": False}
    dataset = _path(config, "dataset", "data/datasets/main")
    if resume:
        saved_path = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--config_path=")))
        saved = json.loads(saved_path.read_text())
        dataset = Path(saved["dataset"]["root"])
    provenance_path = dataset / "meta" / "phone_to_panda_provenance.json"
    if not provenance_path.is_file():
        raise ValueError("Export validated personal-video demonstrations before training.")
    provenance = json.loads(provenance_path.read_text())
    validate_controller_contract(config, provenance)
    episodes = provenance.get("provenance", [])
    if not episodes or any(ep.get("source", {}).get("kind") not in {"personal_video", IMU_PILOT_KIND}
                           or ep.get("success") is not True for ep in episodes):
        raise ValueError("Training requires successful personal-video-derived or verified IMU-pilot episodes; setup-test datasets are forbidden.")
    for episode in episodes:
        if episode.get("source", {}).get("kind") == IMU_PILOT_KIND:
            validate_imu_pilot_source(episode["source"], _root(config), episode.get("trajectory_quality", {}))
    base = Path(config.get("training", {}).get("policy_path", "lerobot/smolvla_base"))
    if (base / "SETUP_TEST_ONLY.json").exists():
        raise ValueError("The tiny random smoke-test model cannot be used as the pretrained training policy.")
    settings = config.get("training", {})
    if str(settings.get("device", "cuda")).startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Repair GPU support or explicitly choose training.device: cpu (slow).")
    subprocess.run(command, cwd=_root(config), check=True)
    output = Path(settings.get("output_dir", "runs/train"))
    if not output.is_absolute():
        output = _root(config) / output
    for checkpoint in (output / "checkpoints").glob("*/pretrained_model"):
        _write_json(checkpoint / "phone_to_panda_provenance.json", provenance)
    return {"training_completed": True, "command": command,
            "output_dir": str(settings.get("output_dir", "runs/train")),
            "note": "Training completion is not evidence of task success. Evaluate the saved checkpoint in physics."}


def _evaluation_action_steps(saved_steps: int, chunk_size: int, override=None) -> int:
    """Validate an inference-only queue length without coercing invalid values."""
    steps = saved_steps if override is None else override
    if (isinstance(steps, bool) or not isinstance(steps, int)
            or isinstance(chunk_size, bool) or not isinstance(chunk_size, int)
            or not 1 <= steps <= chunk_size):
        raise ValueError("evaluation.n_action_steps must be an integer from 1 through the saved chunk_size.")
    return steps


def _load_policy(checkpoint: Path, device: str, n_action_steps: int | None = None):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors

    # Saved policy configs include the polymorphic ``type`` discriminator.
    # The base loader consumes it before decoding the concrete dataclass.
    config = PreTrainedConfig.from_pretrained(str(checkpoint))
    if not isinstance(config, SmolVLAConfig):
        raise ValueError("Checkpoint must contain a SmolVLA policy.")
    # Each prediction still produces chunk_size actions. Execute n_action_steps
    # before predicting again. Set this before constructing the action queue;
    # checkpoint files and the trained chunk size remain unchanged.
    config.n_action_steps = _evaluation_action_steps(config.n_action_steps, config.chunk_size, n_action_steps)
    config.device = device
    config.load_vlm_weights = False
    if config.output_features[ACTION].shape != (7,) or config.input_features[STATE].shape != (8,):
        raise ValueError("Checkpoint is incompatible: expected 8-state, 7-action PhoneToPanda convention.")
    if set(config.image_features) != set(IMAGES):
        raise ValueError("Checkpoint image keys do not match the front and wrist cameras.")
    policy = SmolVLAPolicy.from_pretrained(str(checkpoint), config=config, strict=True).to(device).eval()
    pre, post = make_pre_post_processors(config, pretrained_path=str(checkpoint),
                                       preprocessor_overrides={"device_processor": {"device": device}})
    return policy, pre, post


def _timing_summary(action_seconds: list[float], prediction_seconds: list[float]) -> dict:
    """CPU wall time includes preprocessing, policy selection and action transfer."""
    return {
        "action_calls": len(action_seconds),
        "prediction_calls": len(prediction_seconds),
        "total_seconds": float(sum(action_seconds)),
        "mean_action_seconds": float(np.mean(action_seconds)) if action_seconds else 0.,
        "prediction_total_seconds": float(sum(prediction_seconds)),
        "mean_prediction_seconds": float(np.mean(prediction_seconds)) if prediction_seconds else 0.,
        "p95_prediction_seconds": float(np.percentile(prediction_seconds, 95)) if prediction_seconds else 0.,
        "scope": "Observation preprocessing, policy.select_action, postprocessing and blocking CPU action transfer; excludes rendering, diagnostics and physics.",
    }


def _observation_batch(obs: dict, task: str) -> dict:
    import torch
    batch = {STATE: torch.as_tensor(obs[STATE], dtype=torch.float32), "task": task}
    for key in IMAGES:
        array = np.ascontiguousarray(obs[key])
        if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
            raise ValueError(f"Expected HWC RGB uint8 camera frame for {key}, got {array.shape} {array.dtype}")
        batch[key] = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
    return batch


def run_policy(config: dict, checkpoint: str | Path, output_dir: str | Path,
               episodes: int = 10, seed: int = 1000) -> dict:
    """Closed-loop evaluation: only the simulator's success predicate counts."""
    import torch
    import imageio.v2 as imageio
    from phone_to_panda.simulation import make_env

    if episodes < 1:
        raise ValueError("episodes must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed + episodes > 2**32:
        raise ValueError("Evaluation seeds must be integers in the NumPy seed range [0, 2**32).")
    checkpoint = Path(checkpoint).resolve()
    if not (checkpoint / "model.safetensors").is_file():
        raise ValueError("Choose the pretrained_model directory of a completed local checkpoint.")
    if (checkpoint / "SETUP_TEST_ONLY.json").exists():
        raise ValueError("Setup-test model is random and cannot be presented as a trained-policy demo.")
    training_config_path = checkpoint / "train_config.json"
    if not training_config_path.is_file():
        raise ValueError("Missing training configuration; use a checkpoint produced by the training command.")
    training_config = json.loads(training_config_path.read_text())
    saved_policy = json.loads((checkpoint / "config.json").read_text())
    action_override = config.get("evaluation", {}).get("n_action_steps")
    effective_steps = _evaluation_action_steps(saved_policy["n_action_steps"], saved_policy["chunk_size"], action_override)
    provenance_path = checkpoint / "phone_to_panda_provenance.json"
    if not provenance_path.is_file():
        provenance_path = Path(training_config["dataset"]["root"]) / "meta" / "phone_to_panda_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    orientation_control = validate_controller_contract(config, provenance)
    if provenance.get("action_convention") != CONVENTION:
        raise ValueError("Dataset action convention does not match this simulator.")
    if int(config.get("simulation", {}).get("control_hz", 20)) != provenance.get("fps"):
        raise ValueError("Evaluation control rate must match the training dataset fps.")
    if int(config.get("simulation", {}).get("image_size", 256)) != 256:
        raise ValueError("This trained-policy interface requires simulation.image_size: 256.")
    training_seeds = {ep.get("seed") for ep in provenance.get("provenance", [])}
    if any(seed + index in training_seeds for index in range(episodes)):
        raise ValueError("Evaluation seeds overlap training demonstrations. Use held-out seeds (default 4000).")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    device = config.get("training", {}).get("device", "cuda")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; explicitly select CPU in configuration if intended.")
    policy, pre, post = _load_policy(checkpoint, device, n_action_steps=action_override)
    policy_settings = {
        "checkpoint_n_action_steps": saved_policy["n_action_steps"],
        "n_action_steps": int(policy.config.n_action_steps),
        "chunk_size": int(policy.config.chunk_size),
        "denoising_steps": int(policy.config.num_steps),
        "evaluation_override": action_override,
    }
    if policy_settings["n_action_steps"] != effective_steps:
        raise ValueError("Loaded policy did not apply the requested evaluation action horizon.")
    max_steps = int(config.get("simulation", {}).get("horizon", 400))
    if max_steps < 1:
        raise ValueError("simulation.horizon must be positive")
    fps = int(config.get("simulation", {}).get("control_hz", 20))
    results = []
    save_video = bool(config.get("evaluation", {}).get("save_video", True))
    save_trace = bool(config.get("evaluation", {}).get("save_trace", False))
    all_action_seconds, all_prediction_seconds = [], []
    started = time.perf_counter()
    for index in range(episodes):
        # Reset simulation and policy random streams per episode. Replanning
        # intervals change random-sample consumption, so equal seeds need not
        # yield equal predictions. Seed exclusion covers simulator seeds only,
        # not separate recording sessions or scenes.
        episode_seed = seed + index
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        env = make_env(config, seed=episode_seed)
        policy.reset()
        writer = None
        trace = []
        action_seconds, prediction_seconds = [], []
        first_events = {key: None for key in ("contact_step", "bilateral_contact_step", "firm_grasp_step", "lift_step", "release_after_lift_step")}
        previous_diagnostics = None
        episode_started = time.perf_counter()
        try:
            if save_video:
                writer = imageio.get_writer(destination / f"episode_{index:03d}.mp4", fps=fps, codec="libx264", macro_block_size=1)
            obs = env.reset()
            if save_trace:
                previous_diagnostics = env.grasp_diagnostics()
            success = False
            for step in range(max_steps):
                if writer is not None:
                    writer.append_data(np.concatenate([obs[IMAGES[0]], obs[IMAGES[1]]], axis=1))
                # The pinned SmolVLA implementation predicts only when its action
                # queue is empty. Record actual refreshes, not assumed timing.
                prediction_refresh = bool(policy._check_get_actions_condition())
                inference_started = time.perf_counter()
                with torch.inference_mode():
                    action = post(policy.select_action(pre(_observation_batch(obs, config["task"]))))
                array = action.detach().cpu().numpy().reshape(-1)
                inference_seconds = time.perf_counter() - inference_started
                action_seconds.append(inference_seconds)
                if prediction_refresh:
                    prediction_seconds.append(inference_seconds)
                if array.shape != (7,) or not np.isfinite(array).all():
                    raise ValueError("Policy produced an invalid action")
                if save_trace:
                    trace.append({"step": step, "observation_state": obs[STATE].tolist(),
                                  "cube_xyz": env.cube_position.tolist(),
                                  "policy_action": array.tolist(),
                                  "episode_seed": episode_seed,
                                  "prediction_refresh": prediction_refresh,
                                  "inference_seconds": inference_seconds,
                                  "diagnostics": previous_diagnostics})
                obs, reward, done, info = env.step(np.clip(array, -1.0, 1.0))
                if save_trace:
                    diagnostics = env.grasp_diagnostics()
                    trace[-1].update(applied_action=info.get("applied_action"),
                                     next_cube_xyz=env.cube_position.tolist(), success=bool(env.success()),
                                     next_diagnostics=diagnostics)
                    events = {
                        "contact_step": diagnostics["gripper_contact"],
                        "bilateral_contact_step": diagnostics["bilateral_contact"],
                        "firm_grasp_step": diagnostics["firm_bilateral_grasp"],
                        "lift_step": diagnostics["lifted"],
                        "release_after_lift_step": diagnostics["ever_lifted"] and not diagnostics["gripper_contact"],
                    }
                    for event, happened in events.items():
                        if happened and first_events[event] is None:
                            first_events[event] = step
                    previous_diagnostics = diagnostics
                success = bool(env.success())
                if success or done:
                    break
            if writer is not None:
                writer.append_data(np.concatenate([obs[IMAGES[0]], obs[IMAGES[1]]], axis=1))
            results.append({"seed": episode_seed, "success": success, "steps": step + 1,
                            "wall_seconds": time.perf_counter() - episode_started,
                            "max_cube_height_m": env._max_cube_height,
                            "final_cube_xyz": env.cube_position.tolist(),
                            "first_events": first_events if save_trace else None,
                            "event_step_convention": "Zero-based action index; event observed immediately after that action.",
                            "inference_timing": _timing_summary(action_seconds, prediction_seconds),
                            "video": f"episode_{index:03d}.mp4" if save_video else None})
        finally:
            if writer is not None:
                writer.close()
            env.close()
        all_action_seconds.extend(action_seconds)
        all_prediction_seconds.extend(prediction_seconds)
        _write_json(destination / "episodes.json", results)
        if save_trace:
            _write_json(destination / f"episode_{index:03d}_trace.json", trace)
    report = {"kind": "trained_policy_physics_evaluation", "checkpoint": str(checkpoint),
              "episodes": results, "successes": sum(ep["success"] for ep in results),
              "success_rate": sum(ep["success"] for ep in results) / len(results),
              "wall_seconds": time.perf_counter() - started, "seed_start": seed,
              "randomize_start_m": float(config.get("simulation", {}).get("randomize_start_m", .035)),
              "held_out_from_training_seeds": True,
              "orientation_control": orientation_control,
              "controller": "SmolVLA XYZ and gripper plus fixed downward wrist feedback" if orientation_control == "fixed_downward" else "SmolVLA XYZ, rotation and gripper",
              "video_recorded": save_video,
              "policy_settings": policy_settings,
              "inference_timing": _timing_summary(all_action_seconds, all_prediction_seconds),
              "seed_scheme": {
                  "episode_seed": "seed_start + episode_index",
                  "seeded_rngs": ["python.random", "numpy.random", "torch", "simulator"],
                  "note": "Seeded per episode before simulator construction. Backend/version changes may still change numerical results; changing replanning frequency changes RNG consumption.",
              },
              "interpretation": "Held-out random seeds test this simulation distribution; they do not prove real-world transfer."}
    _write_json(destination / "report.json", report)
    return report


def smoke_test(config: dict, output_dir: str | Path) -> dict:
    """Check LeRobot training and checkpoint reload without network or personal data.

    Write synthetic setup data and a tiny random SmolVLM/SmolVLA under the
    supplied output directory. This checks the API, not task performance.
    """
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (LlamaConfig, PreTrainedTokenizerFast, SmolVLMConfig,
                              SmolVLMImageProcessor, SmolVLMProcessor, SmolVLMVideoProcessor)
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    destination = Path(output_dir).resolve()
    if destination.exists():
        destination = destination.with_name(destination.name + "_" + str(time.time_ns()))
    destination.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(7)
    backbone = destination / "tiny_backbone"
    vocab = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3,
             "<image>": 4, "<fake_token_around_image>": 5, "<global-img>": 6,
             "Pick": 7, "up": 8, "the": 9, "cube": 10, "and": 11, "place": 12, "it": 13, "inside": 14, "box": 15, ".": 16}
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", bos_token="[BOS]", eos_token="[EOS]",
                                        additional_special_tokens=["<image>", "<fake_token_around_image>", "<global-img>"],
                                        extra_special_tokens={"image_token": "<image>", "fake_image_token": "<fake_token_around_image>", "global_image_token": "<global-img>"})
    processor = SmolVLMProcessor(image_processor=SmolVLMImageProcessor(size={"longest_edge": 32}, max_image_size={"longest_edge": 32}),
                                 tokenizer=tokenizer, video_processor=SmolVLMVideoProcessor(size={"longest_edge": 32}), image_seq_len=4)
    processor.save_pretrained(backbone)
    text_config = LlamaConfig(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128,
                              num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                              head_dim=16, pad_token_id=0, bos_token_id=2, eos_token_id=3,
                              attention_bias=False, max_position_embeddings=512)
    SmolVLMConfig(text_config=text_config, vision_config={"hidden_size": 32, "intermediate_size": 64,
                    "num_hidden_layers": 1, "num_attention_heads": 4, "image_size": 32, "patch_size": 8},
                  image_token_id=4, pad_token_id=0, scale_factor=2).save_pretrained(backbone)
    arrays = {"state": np.zeros((4, 8), dtype=np.float32), "action": np.full((4, 7), 0.1, dtype=np.float32),
              "image": np.zeros((4, 256, 256, 3), dtype=np.uint8), "image2": np.full((4, 256, 256, 3), 100, dtype=np.uint8)}
    arrays["state"][:, 0] = np.arange(4) * 0.01
    arrays["action"][:, 0] = np.linspace(-0.2, 0.2, 4)
    dataset_dir = destination / "dataset"
    _create_dataset([(arrays, {"source": {"kind": "setup_test"}, "success": False, "task": config["task"]})],
                    dataset_dir, "local/phone_to_panda_smoke", 20)
    model_config = SmolVLAConfig(vlm_model_name=str(backbone), load_vlm_weights=False, device="cpu", use_amp=False,
                    input_features={STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
                                    **{key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)) for key in IMAGES}},
                    output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
                    chunk_size=4, n_action_steps=2, num_steps=2, num_vlm_layers=2, num_expert_layers=2,
                    resize_imgs_with_padding=(32, 32), expert_width_multiplier=1.0, tokenizer_max_length=24,
                    self_attn_every_n_layers=2, train_expert_only=True, freeze_vision_encoder=True,
                    push_to_hub=False, prefix_length=0)
    initial = destination / "random_initial_policy"
    policy = SmolVLAPolicy(model_config)
    parameter = next(p for p in policy.model.action_out_proj.parameters() if p.requires_grad).detach().clone()
    policy.save_pretrained(initial)
    metadata = LeRobotDatasetMetadata(repo_id="local/phone_to_panda_smoke", root=dataset_dir)
    pre, post = make_pre_post_processors(model_config, dataset_stats=metadata.stats)
    pre.save_pretrained(initial)
    post.save_pretrained(initial)
    _write_json(initial / "SETUP_TEST_ONLY.json", {"trained_task_policy": False})
    del policy, pre, post
    smoke_config = {**config, "training": {"policy_path": str(initial), "repo_id": "local/phone_to_panda_smoke",
                   "device": "cpu", "use_amp": False, "steps": 2, "save_freq": 2, "batch_size": 1,
                   "num_workers": 0, "n_action_steps": 2, "chunk_size": 4, "num_steps": 2}}
    command = training_command(smoke_config, dataset_dir=dataset_dir, output_dir=destination / "training")
    environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                       OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    with (destination / "training.log").open("w") as log:
        subprocess.run(command, cwd=_root(config), env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    checkpoint = destination / "training" / "checkpoints" / "last" / "pretrained_model"
    _write_json(checkpoint / "SETUP_TEST_ONLY.json", {"trained_task_policy": False})
    policy, pre, post = _load_policy(checkpoint, "cpu")
    updated = next(p for p in policy.model.action_out_proj.parameters() if p.requires_grad).detach()
    changed = not torch.equal(parameter, updated)
    obs = {STATE: arrays["state"][0], IMAGES[0]: arrays["image"][0], IMAGES[1]: arrays["image2"][0]}
    with torch.inference_mode():
        action = post(policy.select_action(pre(_observation_batch(obs, config["task"]))))
    valid = action.shape == (1, 7) and bool(torch.isfinite(action).all())
    if not changed or not valid:
        raise RuntimeError("Tiny SmolVLA training/checkpoint/inference check failed")
    report = {"kind": "setup_test_only", "output_dir": str(destination), "trained_task_policy": False, "pretrained_weights_downloaded": False,
              "real_lerobot_training_steps": 2, "optimizer_changed_action_weights": changed,
              "checkpoint_reloaded": True, "finite_action_shape": list(action.shape),
              "note": "Tiny random initialization and synthetic frames only; no task-performance claim.", "command": command}
    _write_json(destination / "report.json", report)
    return report
