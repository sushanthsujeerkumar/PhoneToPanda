"""Friendly command-line entry point. Run bash run.sh --help."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

from .common import ROOT, load_config, project_path


def _print(value):
    print(json.dumps(value, indent=2, default=str))


def doctor(config):
    packages = {}
    for name in ("lerobot", "hf-libero", "robosuite", "mujoco", "torch", "torchvision", "transformers", "opencv-contrib-python", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "MISSING"
    import cv2
    import torch
    cuda = torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(0) if cuda else None
    if cuda:
        # Execute a real CUDA kernel, not just a driver availability query.
        torch.mm(torch.ones((16, 16), device="cuda"), torch.ones((16, 16), device="cuda")).cpu()
    raw = project_path(config, config["paths"]["raw"])
    videos = [p.name for p in raw.iterdir() if p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv")] if raw.exists() else []
    report = {"project": str(ROOT), "python": sys.version.split()[0], "packages": packages,
              "aruco_available": hasattr(cv2, "aruco"), "cuda_available": cuda, "gpu": gpu,
              "personal_videos": videos, "trained_policy": latest_checkpoint(config, required=False),
              "next_step": "See README.md for the measured dataset, training, and demo commands."}
    _print(report)
    return 0 if "MISSING" not in packages.values() and report["aruco_available"] else 1


def latest_checkpoint(config, required=True):
    base = project_path(config, config["training"]["output_dir"]) / "checkpoints"
    candidates = [ROOT / "checkpoints/main", base / "last" / "pretrained_model"] + sorted(base.glob("*/pretrained_model"), reverse=True)
    for path in candidates:
        if (path / "config.json").is_file() and (path / "model.safetensors").is_file():
            return str(path)
    if required:
        raise ValueError("No trained policy found. Prepare your recordings, export the dataset, and run bash run.sh train first. For an installation check use bash run.sh setup-demo.")
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Phone to Panda: record, prepare, retarget, train, evaluate.")
    parser.add_argument("--config", help="Alternate YAML configuration")
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("doctor", help="Check installed packages and graphics card")
    subs.add_parser("markers", help="Create printable marker and calibration assets")
    calibrate = subs.add_parser("calibrate", help="Calibrate camera from a chessboard recording")
    calibrate.add_argument("--video", required=True)
    events = subs.add_parser("events", help="Create a grasp/release annotation template")
    events.add_argument("--video", required=True)
    events.add_argument("--output")
    prepare = subs.add_parser("prepare", help="Track measured object movement from your video")
    prepare.add_argument("--video", required=True)
    prepare.add_argument("--events")
    prepare.add_argument("--output")
    demo = subs.add_parser("setup-demo", help="Physics installation test, using a scripted controller")
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--output")
    replay = subs.add_parser("replay", help="Retarget a tracked personal demonstration to Panda")
    replay.add_argument("--trajectory", required=True)
    replay.add_argument("--seed", type=int, default=0)
    replay.add_argument("--output")
    subs.add_parser("export", help="Convert successful personal demos to a LeRobot training dataset")
    train = subs.add_parser("train", help="Fine-tune pretrained SmolVLA on your local dataset")
    train.add_argument("--dry-run", action="store_true", help="Print the training command without training")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--steps", type=int)
    train.add_argument("--batch-size", type=int)
    for name, helptext in [("demo", "Run one trained-policy episode and save video"), ("evaluate", "Evaluate trained policy on held-out starting positions")]:
        p = subs.add_parser(name, help=helptext)
        p.add_argument("--checkpoint")
        p.add_argument("--episodes", type=int, default=1 if name == "demo" else None)
        p.add_argument("--seed", type=int)
        p.add_argument("--output")
    subs.add_parser("test-learning", help="Offline tiny-model integration test; not task training")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    path = lambda value: project_path(config, value)
    try:
        if args.command == "doctor":
            return doctor(config)
        if args.command in ("markers", "calibrate", "events", "prepare"):
            from . import video
            settings = config["video"]
            if args.command == "markers":
                result = video.generate_calibration_assets(ROOT / "assets/calibration", table_marker_mm=settings["table_marker_size_m"] * 1000, object_marker_mm=settings["object_marker_size_m"] * 1000, chess_square_mm=settings["chess_square_size_m"] * 1000)
            elif args.command == "calibrate":
                result = video.calibrate_camera(path(args.video), path(config["paths"]["calibration"]), square_size_m=settings["chess_square_size_m"])
            elif args.command == "events":
                output = path(args.output) if args.output else path(config["paths"]["annotations"]) / (Path(args.video).stem + ".json")
                result = video.write_events_template(path(args.video), output)
            else:
                event_path = path(args.events) if args.events else path(config["paths"]["annotations"]) / (Path(args.video).stem + ".json")
                output = path(args.output) if args.output else path(config["paths"]["processed"]) / (Path(args.video).stem + ".json")
                result = video.extract_video(path(args.video), path(config["paths"]["calibration"]), event_path, output,
                    table_marker_size_m=settings["table_marker_size_m"], object_marker_size_m=settings["object_marker_size_m"],
                    sample_hz=settings["sample_hz"], max_gap_s=settings["max_gap_s"],
                    tracking_mode=settings.get("tracking_mode", "single_marker"),
                    reference_markers=settings.get("reference_markers"), object_marker_id=settings.get("object_marker_id", 1),
                    write_preview=settings.get("write_previews", True),
                    resting_height_range_m=settings.get("resting_height_range_m", [-0.01, 0.15]),
                    min_valid_fraction=settings.get("min_valid_fraction", 0.90),
                    max_pose_reprojection_error_px=settings.get("max_pose_reprojection_error_px", 3.0),
                    max_table_drift_px=settings.get("max_table_drift_px", 10.0))
            _print(result)
        elif args.command in ("setup-demo", "replay"):
            from .simulation import record_demo
            if args.command == "setup-demo":
                output = path(args.output) if args.output else ROOT / "runs/setup_check" / f"seed_{args.seed}.npz"
                trajectory = None
            else:
                trajectory = path(args.trajectory)
                output = path(args.output) if args.output else path(config["paths"]["robot"]) / f"{trajectory.stem}_seed_{args.seed}.npz"
            result = record_demo(config, output, trajectory_path=trajectory, seed=args.seed)
            _print(result)
            return 0 if result.get("success", False) else 2
        else:
            from . import learning
            if args.command == "export":
                _print(learning.export_dataset(config))
            elif args.command == "train":
                if args.steps is not None:
                    config["training"]["steps"] = args.steps
                if args.batch_size is not None:
                    config["training"]["batch_size"] = args.batch_size
                _print(learning.train(config, dry_run=args.dry_run, resume=args.resume))
            elif args.command == "test-learning":
                _print(learning.smoke_test(config, ROOT / "work/learning_smoke"))
            else:
                checkpoint = str(path(args.checkpoint)) if args.checkpoint else latest_checkpoint(config)
                episodes = args.episodes if args.episodes is not None else config["evaluation"]["episodes"]
                seed = args.seed if args.seed is not None else config["evaluation"]["seed"]
                output = path(args.output) if args.output else ROOT / "runs" / args.command
                _print(learning.run_policy(config, checkpoint, output, episodes=episodes, seed=seed))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, ImportError) as error:
        print(f"Phone to Panda: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
