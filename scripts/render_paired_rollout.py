"""Stack two recorded policy rollouts at the same simulator seed.

Reads only the selected group's reports and the selected seed's trace/video for
each named policy. Playback follows simulation time, not inference wall time.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FPS = 20
WIDTH, VIEW_HEIGHT, TITLE_HEIGHT, ROW_HEADER, FOOTER = 512, 256, 44, 36, 48
HEIGHT = TITLE_HEIGHT + 2 * (ROW_HEADER + VIEW_HEIGHT) + FOOTER


def read_json(path):
    text = Path(path).read_text()
    return text, json.loads(text)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def simple_name(value):
    return bool(value) and all(c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value)


def finite_vector(value, length):
    return (isinstance(value, list) and len(value) == length
            and all(not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v) for v in value))


def load_selected(study, phase, name, group, seed, settings, plan_hash, lock=None, lock_hash=None):
    folder = study / phase / name / group
    manifest_path, report_path = folder / "manifest.json", folder / "evaluation/report.json"
    _, manifest = read_json(manifest_path)
    _, report = read_json(report_path)
    expected_manifest = {"candidate": name, "phase": phase, "group": group,
                         "episode_plan": settings, "study_plan_sha256": plan_hash}
    if any(manifest.get(key) != value for key, value in expected_manifest.items()):
        raise ValueError(f"Manifest does not match the frozen study: {name}/{group}")
    identity = manifest.get("identity", {})
    if "model.safetensors" not in identity.get("checkpoint_files", {}):
        raise ValueError(f"Missing checkpoint identity for {name}")
    if lock is not None and lock["candidates"].get(name) != identity:
        raise ValueError(f"Final candidate identity differs from lock: {name}")
    if lock is not None:
        if not lock_hash or manifest.get("candidate_lock_sha256") != lock_hash:
            raise ValueError(f"Final candidate selection differs from lock: {name}")
        if not lock.get("config_sha256") or manifest.get("base_config_sha256") != lock["config_sha256"]:
            raise ValueError(f"Final base configuration differs from lock: {name}")
        if not lock.get("source_sha256") or manifest.get("source_sha256") != lock["source_sha256"]:
            raise ValueError(f"Final source code differs from lock: {name}")
    simulation = manifest["config"]["simulation"]
    if simulation.get("control_hz") != FPS or simulation.get("image_size") != 256:
        raise ValueError("Paired renderer requires the recorded 20 Hz, 256-pixel camera contract")
    if (report.get("kind") != "trained_policy_physics_evaluation"
            or report.get("randomize_start_m") != settings["radius_m"]
            or report.get("orientation_control") != simulation.get("orientation_control")
            or report.get("policy_settings", {}).get("n_action_steps") != identity.get("n_action_steps")):
        raise ValueError(f"Report settings do not match manifest: {name}")
    expected = list(range(settings["seed"], settings["seed"] + settings["episodes"]))
    episodes = report.get("episodes", [])
    if report.get("seed_start") != settings["seed"] or [episode.get("seed") for episode in episodes] != expected:
        raise ValueError(f"Incomplete or mismatched seed group for {name}")
    index = expected.index(seed)
    episode = episodes[index]
    steps = episode.get("steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1 or not isinstance(episode.get("success"), bool):
        raise ValueError(f"Invalid selected episode report for {name}")
    if not report.get("video_recorded") or episode.get("video") != f"episode_{index:03d}.mp4":
        raise ValueError(f"Selected episode has no recorded canonical video: {name}, seed {seed}")
    trace_path = folder / "evaluation" / f"episode_{index:03d}_trace.json"
    _, trace = read_json(trace_path)
    if (len(trace) != steps or [entry.get("step") for entry in trace] != list(range(steps))
            or any(entry.get("episode_seed") != seed for entry in trace)
            or trace[-1].get("success") != episode["success"]):
        raise ValueError(f"Trace is incomplete or contradicts report: {name}, seed {seed}")
    initial_cube, initial_robot = trace[0].get("cube_xyz"), trace[0].get("observation_state")
    if not finite_vector(initial_cube, 3) or not finite_vector(initial_robot, 8):
        raise ValueError(f"Invalid initial simulator state in {name}")
    video_path = folder / "evaluation" / episode["video"]
    return {
        "name": name, "seed": seed, "success": episode["success"], "steps": steps,
        **({"candidate_lock_sha256": lock_hash,
            "reference_candidate": lock["reference_candidate"], "selected_candidate": lock["selected_candidate"],
            "contrast_role": lock["contrast_roles"].get(name, "Locked reference"),
            "selected_contrast": name in (lock["reference_candidate"], lock["selected_candidate"])} if lock else {}),
        "identity": identity, "initial_cube_xyz": initial_cube, "initial_observation_state": initial_robot,
        "comparison_contract": {key: simulation.get(key) for key in ("control_hz", "image_size", "horizon", "orientation_control")},
        "video_path": video_path,
        "sources": {key: {"path": str(path.relative_to(study)), "sha256": digest(path)} for key, path in
                    (("manifest", manifest_path), ("report", report_path), ("trace", trace_path), ("video", video_path))},
    }


def validate_pair(study, phase, baseline, candidate, group, seed):
    study = Path(study).resolve()
    if phase not in ("development", "final_test"):
        raise ValueError("Select development or final_test explicitly")
    if not all(simple_name(value) for value in (baseline, candidate, group)) or baseline == candidate:
        raise ValueError("Choose distinct simple model names and a simple group name")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Seed must be an integer")
    plan_path = study / "study_plan.json"
    _, plan = read_json(plan_path)
    plan_hash = digest(plan_path)
    settings = plan[phase].get(group)
    if settings is None or seed not in range(settings["seed"], settings["seed"] + settings["episodes"]):
        raise ValueError("Requested group/seed is absent from the selected frozen phase")
    lock = lock_hash = None
    if phase == "final_test":
        lock_bytes = (study / "candidate_lock.json").read_bytes()
        lock, lock_hash = json.loads(lock_bytes), hashlib.sha256(lock_bytes).hexdigest()
        if lock.get("study_plan_sha256") != plan_hash:
            raise ValueError("Study plan differs from final candidate lock")
        candidates, roles = lock.get("candidates", {}), lock.get("contrast_roles")
        if (lock.get("reference_candidate") not in candidates or lock.get("selected_candidate") not in candidates
                or not isinstance(roles, dict) or not (set(candidates) - {lock["reference_candidate"]}) <= set(roles)
                or not set(roles) <= set(candidates)
                or any(not isinstance(role, str) or not role.strip() for role in roles.values())):
            raise ValueError("Invalid reference, selection or contrast roles in final lock")
        if baseline != lock["reference_candidate"]:
            raise ValueError("Final video baseline must be the locked reference_candidate")
    selected = [load_selected(study, phase, name, group, seed, settings, plan_hash, lock, lock_hash) for name in (baseline, candidate)]
    for key in ("initial_cube_xyz", "initial_observation_state", "comparison_contract"):
        if selected[0][key] != selected[1][key]:
            raise ValueError(f"Paired {key} differs; refusing an unmatched comparison video")
    return selected, plan_hash


@lru_cache(maxsize=4)
def font(size):
    from PIL import ImageFont
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def compose_frame(frames, selected, frame_index, phase, group):
    import numpy as np
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (WIDTH, HEIGHT), "#f3f5f7")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, TITLE_HEIGHT), fill="#152b3b")
    title_font, text_font = font(15), font(12)
    draw.text((10, 4), f"Paired simulation · {group} · seed {selected[0]['seed']}", fill="white", font=title_font)
    phase_label = "Development example" if phase == "development" else "Frozen final-test example"
    if phase == "final_test" and not selected[1]["selected_contrast"]:
        phase_label = "Secondary final-test example"
    draw.text((10, 24), f"{phase_label} | simulation time {frame_index / FPS:.2f} s", fill="#d5e4ed", font=text_font)
    for index, (frame, result) in enumerate(zip(frames, selected)):
        y = TITLE_HEIGHT + index * (ROW_HEADER + VIEW_HEIGHT)
        role = "Baseline" if index == 0 else "Candidate"
        name = result["name"] if len(result["name"]) <= 40 else result["name"][:37] + "..."
        draw.text((10, y + 1), f"{role}: {name}", fill="#152b3b", font=title_font)
        outcome = "PLACED" if result["success"] else "NOT PLACED"
        finished = frame_index >= result["steps"]
        status = "FINISHED · final frame held" if frame_index > result["steps"] else "FINISHED" if finished else "RUNNING"
        draw.text((10, y + 19), f"Recorded outcome: {outcome} | {status}", fill="#087e72" if result["success"] else "#a3292c", font=text_font)
        image.paste(Image.fromarray(frame), (0, y + ROW_HEADER))
    footer_y = HEIGHT - FOOTER
    draw.text((10, footer_y + 6), "20 Hz simulation playback; not inference wall time.", fill="#253b4b", font=text_font)
    draw.text((10, footer_y + 24), "Each row: front camera (left), wrist camera (right).", fill="#253b4b", font=text_font)
    return np.asarray(image)


def render_pair(study, phase, baseline, candidate, group, seed, output):
    import imageio.v2 as imageio
    import numpy as np
    output = Path(output).resolve()
    sidecar = output.with_suffix(".json")
    temporary = output.with_name(output.stem + ".partial.mp4")
    if output.suffix.lower() != ".mp4" or any(path.exists() for path in (output, sidecar, temporary)):
        raise ValueError("Choose a new .mp4 output path; completed or partial files are never overwritten")
    selected, plan_hash = validate_pair(study, phase, baseline, candidate, group, seed)
    readers = []
    try:
        for result in selected:
            reader = imageio.get_reader(result["video_path"], format="ffmpeg")
            readers.append(reader)
            metadata = reader.get_meta_data()
            if tuple(metadata.get("size", ())) != (WIDTH, VIEW_HEIGHT) or not math.isclose(metadata.get("fps", 0), FPS, abs_tol=.001):
                raise ValueError(f"Wrong source video resolution or simulation frame rate: {result['name']}")
            frame_count = reader.count_frames()
            if frame_count != result["steps"] + 1:
                raise ValueError(f"Video frame count differs from action trace: {result['name']}")
            result["video_frames"] = frame_count
        output.parent.mkdir(parents=True, exist_ok=True)
        count = max(result["video_frames"] for result in selected)
        last_frames = [None, None]
        with imageio.get_writer(temporary, fps=FPS, codec="libx264", quality=8, macro_block_size=1,
                                ffmpeg_params=["-threads", "2"]) as writer:
            for index in range(count):
                for row, (reader, result) in enumerate(zip(readers, selected)):
                    if index < result["video_frames"]:
                        frame = reader.get_next_data()
                        if frame.shape != (VIEW_HEIGHT, WIDTH, 3) or frame.dtype != np.uint8:
                            raise ValueError("Source video decoded with an unexpected camera layout")
                        last_frames[row] = frame
                writer.append_data(compose_frame(last_frames, selected, index, phase, group))
        evidence = {
            "phase": phase, "group": group, "seed": seed, "study_plan_sha256": plan_hash,
            "paired_initial_cube_and_robot_states_verified": True,
            "playback_fps": FPS, "output_frames": count, "size": [WIDTH, HEIGHT],
            "timing": "One video frame per 20 Hz simulator step plus the final state frame. Shorter completed rollouts hold their final frame. Playback is not inference wall time.",
            "interpretation": "Illustration of one selected seed; aggregate study counts provide the performance comparison.",
            "models": [{key: value for key, value in result.items() if key != "video_path"} for result in selected],
            "output_sha256": digest(temporary),
        }
        temporary.rename(output)
        sidecar.write_text(json.dumps(evidence, indent=2) + "\n")
        return evidence
    finally:
        for reader in readers:
            reader.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--phase", required=True, choices=("development", "final_test"))
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        result = render_pair(ROOT / args.study, args.phase, args.baseline, args.candidate, args.group, args.seed, ROOT / args.output)
        print(json.dumps({"output": args.output, "seed": result["seed"], "frames": result["output_frames"], "fps": FPS}))
        return 0
    except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
        parser.exit(1, f"Paired rollout: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
