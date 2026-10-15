"""Present two disclosed cases from the completed three-policy IMU pilot.

N1 is the first planned narrow start. W4 is the only start where augmentation
regressed against both references. Full attempts play at simulation speed; a
shorter episode's final source frame remains visibly labelled as finished.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("original", "continuation", "augmented")
LABELS = ("Original policy", "Continued training", "IMU-assisted augmentation")
RECIPES = ("20 demonstrations · reference", "20 demonstrations · +500 updates",
           "20 + 4 demonstrations · +500 updates")
COLORS = ("#718899", "#4c7da8", "#087f7f")
CASES = (("N1", "narrow", 0, 84000), ("W4", "wide", 3, 85003))
WIDTH, HEIGHT, FPS = 1600, 900, 20
INK, MUTED, PAPER = "#203844", "#657d88", "#edf2f3"
GREEN, RED = "#08765f", "#a34f37"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@lru_cache(maxsize=20)
def font(size, bold=False):
    return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)


def verify_summary(study):
    summary = read(study / "summary.json")
    if (summary.get("kind") != "imu_assisted_demonstration_pilot"
            or summary.get("default_checkpoint_changed") is not False):
        raise ValueError("Use the completed pilot summary with its original reference unchanged")
    for model in MODELS:
        result = summary["models"][model]
        total = 0
        for group, seed in (("narrow", 84000), ("wide", 85000)):
            value = result["groups"][group]
            episodes = value["episodes"]
            if (value["trials"] != 5 or len(episodes) != 5
                    or [row["seed"] for row in episodes] != list(range(seed, seed + 5))
                    or any(type(row["success"]) is not bool for row in episodes)
                    or sum(row["success"] for row in episodes) != value["successes"]):
                raise ValueError("All ten planned outcomes must be present for each model")
            total += value["successes"]
        if result["trials"] != 10 or result["successes"] != total:
            raise ValueError("Aggregate counts do not agree with individual outcomes")
    for reference in ("original", "continuation"):
        regressions = [(group, index) for group in ("narrow", "wide") for index in range(5)
                       if summary["models"][reference]["groups"][group]["episodes"][index]["success"]
                       and not summary["models"]["augmented"]["groups"][group]["episodes"][index]["success"]]
        if regressions != [("wide", 3)]:
            raise ValueError("W4 is no longer the sole augmented regression; update the disclosed selection")
        if summary["paired"][f"augmented_vs_{reference}"]["identical_initial_cube_and_robot_states"] is not True:
            raise ValueError("Summary does not confirm matched simulator starts")
    if not all(summary["models"][model]["groups"]["narrow"]["episodes"][0]["success"] for model in MODELS):
        raise ValueError("N1 is no longer an all-success example")
    return summary


def load_case(study, summary, case):
    case_name, group, index, seed = case
    clips = []
    start = None
    for model in MODELS:
        folder = study / "evaluation" / model / group
        episode_name = f"episode_{index:03d}"
        paths = {"video": folder / "evaluation" / f"{episode_name}.mp4",
                 "trace": folder / "evaluation" / f"{episode_name}_trace.json",
                 "report": folder / "evaluation" / "report.json"}
        receipt = read(folder / "outputs.json")
        for path in paths.values():
            if digest(path) != receipt.get(path.name):
                raise ValueError(f"Evaluation evidence changed: {path}")
        report, trace = read(paths["report"]), read(paths["trace"])
        episode = report["episodes"][index]
        selected = summary["models"][model]["groups"][group]["episodes"][index]
        if (report["kind"] != "trained_policy_physics_evaluation"
                or report["orientation_control"] != "fixed_downward"
                or report["policy_settings"]["n_action_steps"] != 5
                or episode["seed"] != seed or episode["success"] != selected["success"]
                or episode["steps"] != selected["steps"] or len(trace) != episode["steps"]):
            raise ValueError("Report, trace and summary do not identify the same learned-policy episode")
        initial = [trace[0]["cube_xyz"], trace[0]["observation_state"]]
        if start is None:
            start = initial
        elif initial != start:
            raise ValueError("The three clips do not share the same initial cube and robot state")
        frames = []
        with av.open(str(paths["video"])) as container:
            stream = container.streams.video[0]
            if (stream.width, stream.height) != (512, 256) or stream.average_rate != FPS:
                raise ValueError("Expected the recorded 512x256, 20 fps two-camera contract")
            for number, frame in enumerate(container.decode(stream)):
                if frame.pts is None or abs(float(frame.pts * frame.time_base) - number / FPS) > 1e-6:
                    raise ValueError("Source frame timestamps do not match simulation playback time")
                frames.append(frame.to_image())
        if len(frames) != episode["steps"] + 1:
            raise ValueError("The complete source video must contain the initial frame and every action result")
        clips.append({"model": model, "case": case_name, "seed": seed,
                      "success": episode["success"], "steps": episode["steps"],
                      "source_frames": len(frames), "duration_s": len(frames) / FPS,
                      "initial_cube_and_robot_state": initial,
                      "sources": {key: {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}
                                  for key, path in paths.items()},
                      "frames": frames})
    return clips


def make_background(summary, case):
    name, group, _, _ = case
    image = Image.new("RGB", (WIDTH, HEIGHT), PAPER)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 124), fill=INK)
    draw.text((28, 21), "PHONE TO PANDA  /  ONE-RECORDING IMU PILOT", font=font(16, True), fill="#a6d1ce")
    draw.text((28, 52), "One recording. Three policy runs.", font=font(35, True), fill="white")
    draw.text((1110, 38), "SAVED LEARNED-POLICY REPLAYS", font=font(16, True), fill="#dfebed")
    draw.text((1110, 69), "Simulation playback · normal speed", font=font(17), fill="#afc4cd")
    selection = "First planned narrow start · all three placed" if name == "N1" else "Sole augmented regression · retained in full"
    draw.text((27, 145), f"{name}  /  {selection}", font=font(24, True), fill=INK)
    draw.text((28, 181), "Same initial cube and robot state in all three views", font=font(17), fill=MUTED)
    for index, model in enumerate(MODELS):
        left = 16 + index * 528
        draw.rounded_rectangle((left - 1, 218, left + 513, 661), radius=12, fill="white")
        draw.line((left + 12, 220, left + 500, 220), width=4, fill=COLORS[index])
        draw.text((left + 16, 238), LABELS[index], font=font(22, True), fill=INK)
        draw.text((left + 16, 273), RECIPES[index], font=font(15), fill=MUTED)
        result = summary["models"][model]
        draw.text((left + 16, 306), f"{result['successes']} / {result['trials']} placements overall", font=font(19, True), fill=COLORS[index])
        draw.text((left + 104, 610), "Front camera", font=font(13), fill=MUTED)
        draw.text((left + 361, 610), "Wrist camera", font=font(13), fill=MUTED)
    draw.text((28, 694), "What was tested", font=font(21, True), fill=INK)
    draw.text((28, 731), "One new human take -> four robot demonstrations.", font=font(18), fill=MUTED)
    draw.text((28, 765), "No placement gain here; original model retained.", font=font(18), fill=MUTED)
    draw.text((28, 799), "Demonstration augmentation, not sensor fusion.", font=font(18), fill=MUTED)
    draw.text((828, 694), "All 30 planned outcomes", font=font(20, True), fill=INK)
    draw.text((1125, 698), "Green: placed  ·  red: failed", font=font(15), fill=MUTED)
    labels = [f"N{i+1}" for i in range(5)] + [f"W{i+1}" for i in range(5)]
    for column, label in enumerate(labels):
        x = 1050 + 50 * column
        draw.text((x + 5, 731), label, font=font(13, label == name), fill=INK)
    for row, model in enumerate(MODELS):
        y = 755 + row * 25
        short = ("Original", "Continued", "IMU-assisted")[row]
        draw.text((829, y + 1), short, font=font(14), fill=MUTED)
        values = [episode["success"] for g in ("narrow", "wide")
                  for episode in summary["models"][model]["groups"][g]["episodes"]]
        for column, success in enumerate(values):
            x = 1050 + 50 * column
            draw.rounded_rectangle((x, y, x + 33, y + 17), radius=4, fill="#b7e4d4" if success else "#f0cbbd")
            draw.text((x + 11, y + 1), "+" if success else "-", font=font(12, True), fill=GREEN if success else RED)
            if labels[column] == name:
                draw.rounded_rectangle((x - 2, y - 2, x + 35, y + 19), radius=5, outline=INK, width=1)
    draw.line((24, 845, WIDTH - 24, 845), fill="#cbd8dd", width=1)
    draw.text((28, 861), "Fixed downward wrist feedback · cameras + robot state at runtime · no IMU inputs", font=font(16), fill=MUTED)
    draw.text((1130, 861), "One scene · one training seed", font=font(16), fill=MUTED)
    return image


def compose(background, clips, frame_index, frame_count):
    image = background.copy()
    draw = ImageDraw.Draw(image)
    draw.text((1190, 153), f"{frame_index / FPS:05.2f} / {frame_count / FPS:05.2f} s", font=font(22, True), fill=INK)
    for index, clip in enumerate(clips):
        left = 16 + index * 528
        source_index = min(frame_index, clip["source_frames"] - 1)
        image.paste(clip["frames"][source_index], (left, 346))
        finished = frame_index >= clip["source_frames"] - 1
        result = "PLACED" if clip["success"] else "FAILED"
        draw.text((left + 16, 635), f"Recorded result: {result}", font=font(15, True), fill=GREEN if clip["success"] else RED)
        label = "Finished · final frame held" if finished else "Running"
        draw.text((left + 278, 635), label, font=font(14), fill=MUTED)
    return image


def render(study, output):
    study, output = Path(study).resolve(), Path(output).resolve()
    poster, sidecar = output.with_suffix(".jpg"), output.with_suffix(".json")
    temporary = output.with_name(output.stem + ".partial.mp4")
    if output.suffix.lower() != ".mp4" or any(p.exists() for p in (output, poster, sidecar, temporary)):
        raise ValueError("Choose a fresh MP4 output; completed artifacts are never overwritten")
    summary = verify_summary(study)
    output.parent.mkdir(parents=True, exist_ok=True)
    total_frames, segments = 0, []
    with av.open(str(temporary), "w", options={"movflags": "+faststart"}) as destination:
        stream = destination.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        stream.options = {"crf": "20", "preset": "medium"}
        stream.time_base = Fraction(1, FPS)
        for case in CASES:
            clips = load_case(study, summary, case)
            count = max(clip["source_frames"] for clip in clips)
            background = make_background(summary, case)
            segment_start = total_frames
            for index in range(count):
                frame = compose(background, clips, index, count)
                if case[0] == "W4" and index == 12 * FPS:
                    frame.save(poster, quality=94)
                    poster_frame = total_frames
                encoded = av.VideoFrame.from_image(frame)
                encoded.pts, encoded.time_base = total_frames, Fraction(1, FPS)
                for packet in stream.encode(encoded):
                    destination.mux(packet)
                total_frames += 1
            for clip in clips:
                clip.pop("frames")
                clip["final_frame_repetitions"] = count - clip["source_frames"]
            segments.append({"case": case[0], "group": case[1], "seed": case[3],
                             "first_output_frame": segment_start, "end_output_frame_exclusive": total_frames,
                             "start_s": segment_start / FPS, "end_s_exclusive": total_frames / FPS,
                             "duration_s": count / FPS, "source_clips": clips})
            print(f"Rendered {case[0]}: {count} frames, {count / FPS:.2f} seconds", flush=True)
        for packet in stream.encode():
            destination.mux(packet)
    temporary.replace(output)
    report = {
        "kind": "illustrative_paired_learned_policy_comparison", "width": WIDTH, "height": HEIGHT,
        "fps": FPS, "frame_count": total_frames, "duration_seconds": total_frames / FPS,
        "selection": "N1 is the first planned narrow start; W4 is the sole augmented regression against both original and continued training. This illustrative selection was made after inspecting all 30 outcomes; it is not an additional benchmark or model selection.",
        "all_trial_success_counts": {model: {key: summary["models"][model][key] for key in ("successes", "trials")}
                                     for model in MODELS},
        "segments": segments,
        "edits": "Two complete selected cases are joined in disclosed order at normal 20 Hz simulation speed. Each panel retains every source frame in order with no cuts, motion synthesis or speed change. Shorter attempts hold their final source frame with an explicit Finished label. Only padding, headings, aggregate outcomes and other textual/graphic annotations are added. Both source simulator cameras are retained at native 512x256 resolution.",
        "playback_scope": "Simulation time, not inference wall time or real-robot latency.",
        "controller": "Learned policy actions with common fixed downward wrist feedback; no grasping expert at evaluation.",
        "runtime_inputs": "Simulated camera images and robot state; no IMU inputs or sensor fusion.",
        "audio_present": False, "original_metadata_copied": False,
        "summary": {"path": str((study / "summary.json").relative_to(ROOT)), "sha256": digest(study / "summary.json")},
        "plan": {"path": str((study / "plan.json").relative_to(ROOT)), "sha256": digest(study / "plan.json")},
        "renderer": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": digest(__file__)},
        "output_sha256": digest(output),
        "poster": {"path": str(poster.relative_to(ROOT)), "output_frame_zero_based": poster_frame,
                   "timestamp_s": poster_frame / FPS, "sha256": digest(poster)},
        "limits": summary["limits"],
    }
    sidecar.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "duration_s": total_frames / FPS, "frames": total_frames}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "results/imu_pilot_20261015")
    parser.add_argument("--output", type=Path, default=ROOT / "media/imu_pilot/policy_comparison.mp4")
    args = parser.parse_args()
    render(args.study, args.output)
