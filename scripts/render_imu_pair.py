"""Render phone footage beside its recorded, fixed-shoulder IMU arm estimate.

The provisional timing relation is supplied by the reviewed events file:
sensor_elapsed_s = video_elapsed_s + imu_time_offset_s. This is an inspection
video, not a learned policy, sensor fusion result, or accuracy measurement.
Only the common time range is rendered. Invalid or missing poses remain blank.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import av
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 1600, 900
INK, MUTED = "#20343e", "#5e7078"
TEAL, ORANGE = "#007f83", "#c77826"
BACKGROUND, WHITE = "#eef2f3", "#ffffff"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name
    return ImageFont.truetype(str(path), size)


def read_video(path: Path):
    """Keep presentation timestamps; do not infer timing from frame numbers."""
    frames, timestamps = [], []
    with av.open(str(path)) as source:
        stream = source.streams.video[0]
        origin = None
        last_duration = None
        for frame in source.decode(stream):
            if frame.pts is None:
                raise ValueError("A video frame has no presentation timestamp")
            time_s = float(frame.pts * frame.time_base)
            if origin is None:
                origin = time_s
            timestamps.append(time_s - origin)
            # Scale without cropping or changing the content of the source frame.
            frames.append(frame.to_image().resize((960, 540), Image.Resampling.LANCZOS))
            if frame.duration:
                last_duration = float(frame.duration * frame.time_base)
        if not frames or any(b <= a for a, b in zip(timestamps, timestamps[1:])):
            raise ValueError("Video presentation timestamps must increase")
        if stream.width * 9 != stream.height * 16:
            raise ValueError("This layout expects the selected landscape 16:9 video")
        if not last_duration or last_duration <= 0:
            last_duration = timestamps[-1] - timestamps[-2]
    return frames, np.array(timestamps), timestamps[-1] + last_duration


class ArmPlot:
    """A fixed camera and equal metre scales keep the pose easy to inspect."""

    def __init__(self):
        self.figure = plt.figure(figsize=(5.5, 5.4), dpi=100, facecolor=WHITE)
        self.axes = self.figure.add_subplot(111, projection="3d")
        ax = self.axes
        ax.set_facecolor(WHITE)
        ax.set(xlim=(-.1, .6), ylim=(-.4, .3), zlim=(-.65, .05))
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=18, azim=-58)
        ax.set_xlabel("+X forward (m)", fontsize=10, labelpad=7, color=INK)
        ax.set_ylabel("+Y left (m)", fontsize=10, labelpad=7, color=INK)
        ax.set_zlabel("+Z up (m)", fontsize=10, labelpad=5, color=INK)
        ax.tick_params(labelsize=8, colors=MUTED)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis._axinfo["grid"]["color"] = (.80, .85, .87, .65)
        self.upper, = ax.plot([], [], [], color=TEAL, linewidth=9, solid_capstyle="round")
        self.forearm, = ax.plot([], [], [], color=ORANGE, linewidth=9, solid_capstyle="round")
        self.joints, = ax.plot([], [], [], "o", color=INK, markersize=8)
        ax.text(0, 0, .045, "Shoulder", color=MUTED, fontsize=10)
        self.figure.subplots_adjust(left=0, right=.88, bottom=.08, top=.97)

    def draw(self, pose: dict | None) -> Image.Image:
        if pose is None:
            for line in (self.upper, self.forearm, self.joints):
                line.set_data_3d([], [], [])
        else:
            points = np.array([pose["points"][key] for key in ("shoulder", "elbow", "wrist")])
            self.upper.set_data_3d(*points[:2].T)
            self.forearm.set_data_3d(*points[1:].T)
            self.joints.set_data_3d(*points.T)
        self.figure.canvas.draw()
        return Image.fromarray(np.asarray(self.figure.canvas.buffer_rgba())[..., :3].copy())

    def close(self):
        plt.close(self.figure)


def recorded_pose(frames: list, times: np.ndarray, target_s: float, tolerance_s: float):
    """Select one real nearby sample; never interpolate through missing data."""
    right = int(np.searchsorted(times, target_s))
    candidates = [index for index in (right - 1, right) if 0 <= index < len(times)]
    if not candidates or target_s < times[0] or target_s > times[-1]:
        return None, None
    index = min(candidates, key=lambda value: abs(times[value] - target_s))
    if abs(times[index] - target_s) > tolerance_s:
        return None, index
    pose = frames[index].get("pose", {})
    if not pose.get("valid"):
        return None, index
    points = pose.get("points", {})
    for key in ("shoulder", "elbow", "wrist"):
        values = np.asarray(points.get(key, []), dtype=float)
        if values.shape != (3,) or not np.isfinite(values).all():
            return None, index
    return pose, index


def compose(video_frame, plot, video_s, sensor_s, pose, source_s, offset, fps):
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 126), fill=INK)
    draw.text((32, 22), "PHONE TO PANDA  /  RECORDED MOTION PILOT", fill="#a7d6d4", font=font(17, True))
    draw.text((32, 51), "Phone video + two-IMU arm replay", fill=WHITE, font=font(34, True))
    draw.text((1055, 58), f"Approximate alignment {offset:+.2f}s", fill="#ffe0aa", font=font(24, True))

    draw.rounded_rectangle((24, 148, 992, 786), radius=14, fill=WHITE)
    draw.rounded_rectangle((1008, 148, 1576, 786), radius=14, fill=WHITE)
    draw.text((40, 164), "ORIGINAL PHONE VIDEO", fill=INK, font=font(19, True))
    draw.text((1024, 164), "RECORDED RIGHT-ARM ESTIMATE", fill=INK, font=font(19, True))
    image.paste(video_frame, (28, 202))
    image.paste(plot, (1015, 202))
    draw.text((40, 753), f"Video {video_s:5.2f} s   |   Source frame {source_s:5.2f} s", fill=MUTED, font=font(18))
    draw.text((1028, 739), f"IMU {sensor_s:5.2f} s", fill=MUTED, font=font(18))
    if pose is not None:
        angle = pose.get("elbow_flexion_deg", pose.get("elbow_angle_deg"))
        angle_text = f"Elbow {angle:.0f} degrees" if isinstance(angle, (int, float)) else "Recorded valid pose"
        draw.text((1240, 739), angle_text, fill=INK, font=font(18))
    else:
        draw.rectangle((1050, 400, 1540, 475), fill=WHITE)
        draw.text((1110, 425), "IMU pose unavailable", fill="#a63d2f", font=font(23, True))

    draw.line((1040, 194, 1072, 194), fill=TEAL, width=5)
    draw.text((1078, 184), "Upper arm", fill=MUTED, font=font(14))
    draw.line((1230, 194, 1262, 194), fill=ORANGE, width=5)
    draw.text((1268, 184), "Forearm", fill=MUTED, font=font(14))
    draw.text((32, 810), "IMU wrist estimate; fixed shoulder", fill=INK, font=font(22, True))
    draw.text((32, 845), "Not a learned policy or fused tracker", fill=MUTED, font=font(20))
    draw.text((920, 808), "Timing is provisional; this is not an accuracy test.", fill=MUTED, font=font(19))
    draw.text((920, 844), f"Normal-speed playback  |  {fps} fps preview  |  Overlap only", fill=MUTED, font=font(19))
    return image


def render(video: Path, session: Path, events: Path, output: Path, fps: int):
    if fps < 1 or fps > 30:
        raise ValueError("Choose a preview frame rate from 1 to 30")
    if output.suffix.lower() != ".mp4":
        raise ValueError("Output must be an MP4")
    sidecar = output.with_suffix(".json")
    partial = output.with_name(output.stem + ".partial.mp4")
    if any(path.exists() for path in (output, sidecar, partial)):
        raise ValueError("Choose a new output filename; existing renders are not overwritten")
    data = json.loads(session.read_text())
    labels = json.loads(events.read_text())
    if labels["video_file"] != video.name or labels["sensor_session_id"] != data["metadata"]["id"]:
        raise ValueError("Events refer to a different video or sensor session")
    if labels.get("time_relation") != "sensor_elapsed_s = video_elapsed_s + imu_time_offset_s":
        raise ValueError("Unsupported synchronization convention")
    offset = float(labels["imu_time_offset_s"])
    if not math.isfinite(offset):
        raise ValueError("Time offset must be finite")
    frames = data["frames"]
    times = np.array([frame["host_elapsed_s"] for frame in frames], dtype=float)
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Sensor timestamps must be finite and strictly increasing")
    video_frames, video_times, video_end = read_video(video)
    start_s, end_s = max(0., times[0] - offset), min(video_end, times[-1] - offset)
    count = math.floor((end_s - start_s) * fps)
    if count < 1:
        raise ValueError("Video and sensor recording have no usable overlapping time range")
    output.parent.mkdir(parents=True, exist_ok=True)
    plot = ArmPlot()
    samples, snapshots = [], []
    try:
        with av.open(str(partial), mode="w", options={"movflags": "+faststart"}) as destination:
            stream = destination.add_stream("libx264", rate=fps)
            stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
            stream.options = {"crf": "20", "preset": "medium"}
            stream.time_base = Fraction(1, fps)
            for index in range(count):
                video_s = start_s + index / fps
                video_index = bisect_right(video_times, video_s + 1e-9) - 1
                pose, sensor_index = recorded_pose(frames, times, video_s + offset, .04)
                frame = compose(video_frames[video_index], plot.draw(pose), video_s,
                                video_s + offset, pose, video_times[video_index], offset, fps)
                if index in (0, count // 2, count - 1):
                    snapshot = output.parent / f"paired_preview_{index:03d}.jpg"
                    frame.save(snapshot, quality=93)
                    snapshots.append(snapshot.name)
                samples.append({"preview_frame": index, "video_time_s": video_s,
                                "source_video_pts_s": float(video_times[video_index]),
                                "sensor_target_s": video_s + offset,
                                "sensor_sample_s": None if sensor_index is None else float(times[sensor_index]),
                                "pose_available": pose is not None})
                encoded = av.VideoFrame.from_image(frame)
                encoded.pts, encoded.time_base = index, Fraction(1, fps)
                for packet in stream.encode(encoded):
                    destination.mux(packet)
            for packet in stream.encode():
                destination.mux(packet)
        partial.replace(output)
    finally:
        plot.close()
    report = {
        "kind": "video_and_recorded_imu_inspection", "learned_policy": False,
        "fused_tracker": False, "normal_speed": True, "fps": fps,
        "frames": count, "duration_s": count / fps,
        "video_range_s": [start_s, start_s + count / fps],
        "available_overlap_s": [start_s, end_s], "source_video_duration_s": video_end,
        "excluded_video_start_s": start_s,
        "excluded_video_end_s": video_end - (start_s + count / fps),
        "imu_time_offset_s": offset, "alignment": labels["synchronization"],
        "offset_uncertainty_s": labels.get("offset_uncertainty_s"),
        "time_relation": labels["time_relation"],
        "pose_source": "Recorded valid pose.points; nearest sample within 40 ms; no interpolation",
        "coordinate_frame": "Fixed shoulder: +X forward, +Y left, +Z up; metres",
        "limb_lengths_m": data["metadata"]["calibration"]["lengths_m"],
        "missing_pose_preview_frames": sum(not sample["pose_available"] for sample in samples),
        "audio_included": False, "original_metadata_copied": False,
        "sources": {name: {"filename": path.name, "sha256": digest(path)}
                    for name, path in (("video", video), ("session", session), ("events", events))},
        "snapshots": snapshots, "samples": samples,
        "limitations": ["Alignment is provisional, not measured synchronization.",
                        "Arm pose uses configured limb lengths and a fixed shoulder.",
                        "This view is not a learned policy, fused tracker, or tracking accuracy test.",
                        "Only the overlapping source time range is shown; absent poses are never filled."],
    }
    sidecar.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("frames", "duration_s", "video_range_s",
                     "missing_pose_preview_frames", "excluded_video_end_s")}, indent=2))
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    render(args.video, args.session, args.events, args.output, args.fps)


if __name__ == "__main__":
    main()
