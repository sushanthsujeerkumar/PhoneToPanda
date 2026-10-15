"""Known rotations and capture safety, using simulated clocks only."""
import json
import math

import numpy as np
import pytest

from arm_checker.motion import ArmMotion, IDENTITY, angle_between, average, axis_angle, inverse, multiply


def frame(t, upper=IDENTITY, forearm=IDENTITY, ids=("3", "4"), **changes):
    timestamp = round((t + 1) * 1_000_000)
    return {"type": "frame", "t_us": timestamp,
            "sensors": [{"id": identity, "t_us": timestamp, "q": np.asarray(q).tolist(), **changes}
                        for identity, q in zip(ids, (upper, forearm))]}


def feed_window(motion, start, duration, upper=IDENTITY, forearm=IDENTITY, ids=("3", "4"), **changes):
    for index in range(1, round(duration * 20) + 1):
        now = start + index / 20
        motion.update(frame(now, upper, forearm, ids, **changes), now=now)
    return now


def calibrated(motion=None, table=(IDENTITY, IDENTITY), neutral=(IDENTITY, IDENTITY), ids=("3", "4")):
    motion = motion or ArmMotion()
    motion.update(frame(100, *table, ids), now=100)
    motion.start_table(now=100)
    now = feed_window(motion, 100, 2.15, *table, ids)
    assert motion.table_ready
    motion.update(frame(now + .05, *neutral, ids), now=now + .05)
    now += .05
    motion.start_neutral(now=now)
    now = feed_window(motion, now, 3.15, *neutral, ids)
    assert motion.neutral_ready
    return motion, now


def test_two_timed_stages_and_neutral_geometry():
    motion = ArmMotion(upper_length_m=.31, forearm_length_m=.27)
    assert motion.pose(now=100)["points"] is None
    with pytest.raises(ValueError, match="missing"):
        motion.start_table(now=100)
    motion.update(frame(100), now=100)
    with pytest.raises(ValueError, match="table"):
        motion.start_neutral(now=100)
    motion.start_table(now=100)
    feed_window(motion, 100, 1.95)
    assert not motion.table_ready and motion.status(now=101.95)["countdown_s"] > 0
    feed_window(motion, 101.95, .20)
    assert motion.table_ready and not motion.neutral_ready
    motion.start_neutral(now=102.15)
    feed_window(motion, 102.15, 2.95)
    assert not motion.neutral_ready
    now = feed_window(motion, 105.10, .20)
    pose = motion.pose(now=now)
    assert pose["valid"] and pose["elbow_flexion_deg"] == pytest.approx(0)
    assert pose["points"]["shoulder"] == [0, 0, 0]
    assert pose["points"]["elbow"] == pytest.approx([0, 0, -.31])
    assert pose["points"]["wrist"] == pytest.approx([0, 0, -.58])
    assert "Missing gyro" in motion.status(now=now)["stationarity_note"]
    json.dumps(motion.snapshot(), allow_nan=False)


def test_known_forearm_bend_is_shoulder_relative():
    motion, now = calibrated()
    bent = axis_angle((0, 1, 0), -math.pi / 2)
    motion.update(frame(now + .05, forearm=bent), now=now + .05)
    pose = motion.pose(now=now + .05)
    assert pose["points"]["elbow"] == pytest.approx([0, 0, -.30])
    assert pose["points"]["wrist"] == pytest.approx([.26, 0, -.30])
    assert pose["elbow_flexion_deg"] == pytest.approx(90)


def test_independent_world_yaw_and_mount_offsets_preserve_rotation_axes():
    table = (axis_angle((0, 0, 1), .8), axis_angle((0, 0, 1), -.6))
    neutral = (multiply(table[0], axis_angle((1, 0, 0), 1.1)),
               multiply(table[1], axis_angle((0, 1, 0), -.4)))
    motion, now = calibrated(table=table, neutral=neutral)
    desired = {"upper": axis_angle((0, 1, 0), -math.pi / 2),
               "forearm": axis_angle((1, 0, 0), math.pi / 2)}
    raw = {role: multiply(multiply(inverse(motion.alignment[role]), desired[role]), inverse(motion.offset[role]))
           for role in desired}
    motion.update(frame(now + .05, raw["upper"], raw["forearm"]), now=now + .05)
    pose = motion.pose(now=now + .05)
    assert pose["points"]["elbow"] == pytest.approx([.3, 0, 0], abs=1e-8)
    assert pose["points"]["wrist"] == pytest.approx([.3, .26, 0], abs=1e-8)
    for role in desired:
        assert angle_between(pose["segment_quaternions"][role], desired[role]) < 1e-7


def test_quaternion_sign_does_not_change_capture_or_mean():
    q = axis_angle((0, 0, 1), .7)
    assert angle_between(average([q, -q, q, -q]), q) < 1e-7
    motion = ArmMotion()
    motion.update(frame(100, q, q), now=100)
    motion.start_table(now=100)
    for index in range(1, 45):
        now = 100 + index / 20
        signed = q * (-1 if index % 2 else 1)
        motion.update(frame(now, signed, signed), now=now)
    assert motion.table_ready


@pytest.mark.parametrize("changes", [{"g": [.11, 0, 0]}, {"a": [0, 0, 1.16]}, {"flags": 27}])
def test_motion_or_invalid_flags_prevent_capture(changes):
    motion = ArmMotion()
    motion.update(frame(100), now=100)
    motion.start_table(now=100)
    now = feed_window(motion, 100, 2.5, **changes)
    assert not motion.table_ready and motion.status(now=now)["capture_samples"] == 0
    now = feed_window(motion, now, 2.15, g=[0, 0, 0], a=[0, 0, 1])
    assert motion.table_ready


def test_orientation_motion_restarts_capture_and_tilt_blocks_table():
    motion = ArmMotion()
    motion.update(frame(100), now=100)
    motion.start_table(now=100)
    feed_window(motion, 100, 1.5)
    turned = axis_angle((0, 0, 1), math.radians(4))
    motion.update(frame(101.55, turned, turned), now=101.55)
    assert motion.status(now=101.55)["capture_samples"] == 0
    now = feed_window(motion, 101.55, 1.5, turned, turned)
    assert not motion.table_ready
    tilted = axis_angle((1, 0, 0), math.radians(20))
    now = feed_window(motion, now, 2.5, tilted, tilted)
    assert not motion.table_ready and motion.status(now=now)["capture_samples"] == 0


def test_frozen_timestamps_and_missing_sensor_do_not_resume_as_fresh():
    motion, now = calibrated()
    frozen = frame(now)
    motion.update(frozen, now=now + .20)
    assert motion.pose(now=now + .20)["valid"]
    motion.update(frozen, now=now + .26)
    assert not motion.pose(now=now + .26)["valid"]
    missing = frame(now)
    missing["sensors"].pop()
    motion.update(missing, now=now + .27)
    motion.update(frozen, now=now + .28)
    assert not motion.pose(now=now + .28)["valid"]
    assert motion.pose(now=now + .28)["points"] is None
    motion.update(frame(now + .30), now=now + .30)
    assert motion.pose(now=now + .30)["valid"]


def test_capture_requires_continuous_fresh_pairs_and_twenty_distinct_samples():
    motion = ArmMotion(stale_after_s=5)
    motion.update(frame(100), now=100)
    motion.start_table(now=100)
    for index in range(1, 15):
        now = 100 + index * .2
        motion.update(frame(now), now=now)
    assert not motion.table_ready  # Duration alone cannot approve fourteen samples.
    motion = ArmMotion()
    motion.update(frame(100), now=100)
    motion.start_table(now=100)
    feed_window(motion, 100, 1.5)
    motion.update(frame(103), now=103)  # A gap resets even without intervening UI polls.
    assert motion.status(now=103)["capture_samples"] == 1
    now = feed_window(motion, 103, 1.)
    assert not motion.table_ready
    stale = frame(now + .05)
    stale["sensors"][1]["t_us"] -= 100_000
    motion.update(stale, now=now + .05)
    assert motion.status(now=now + .05)["capture_samples"] == 0


@pytest.mark.parametrize("change", ["generation", "device_clock", "sensor_clock", "host_clock", "explicit"])
def test_restarts_invalidate_calibration(change):
    motion, now = calibrated()
    motion.update({**frame(now + .05), "generation": 1}, now=now + .05)
    next_frame, next_now = frame(now + .10), now + .10
    if change == "generation":
        next_frame["generation"] = 2
    elif change == "device_clock":
        next_frame = frame(1)
    elif change == "sensor_clock":
        next_frame["sensors"][1]["t_us"] -= 60_000
    elif change == "host_clock":
        next_now = now
    else:
        motion.invalidate("Mount changed")
    motion.update(next_frame, now=next_now)
    assert not motion.neutral_ready and not motion.pose(now=next_now)["valid"]
    assert motion.generation >= 1


def test_config_mapping_is_explicit_atomic_and_invalidates_offsets():
    motion, now = calibrated(ArmMotion(upper_id="upper", forearm_id="forearm"), ids=("upper", "forearm"))
    with pytest.raises(ValueError):
        motion.configure(upper_id="forearm")
    assert motion.mapping == {"upper": "upper", "forearm": "forearm"}
    motion.configure(upper_length_m=.4)
    assert motion.pose(now=now)["points"]["elbow"] == pytest.approx([0, 0, -.4])
    motion.configure(upper_id="forearm", forearm_id="upper")
    assert not motion.neutral_ready
    assert motion.pose(now=now)["points"] is None


def test_optional_generation_omission_does_not_erase_last_known_generation():
    motion, now = calibrated()
    motion.update({**frame(now + .05), "generation": 1}, now=now + .05)
    motion.update(frame(now + .10), now=now + .10)
    motion.update({**frame(now + .15), "generation": 2}, now=now + .15)
    assert not motion.neutral_ready
    assert motion.snapshot()["device_generation"] == 2


def test_bend_reveals_swapped_assignment_without_claiming_automatic_detection():
    bent = axis_angle((0, 1, 0), -math.pi / 2)
    normal, now = calibrated()
    swapped, _ = calibrated(ArmMotion(upper_id="4", forearm_id="3"))
    for motion in (normal, swapped):
        motion.update(frame(now + .05, forearm=bent), now=now + .05)
    assert normal.pose(now=now + .05)["points"]["elbow"] == pytest.approx([0, 0, -.3])
    assert swapped.pose(now=now + .05)["points"]["elbow"] == pytest.approx([.3, 0, 0])


@pytest.mark.parametrize("value", [None, [0, 0, 0, 0], [float("nan"), 0, 0, 0]])
def test_bad_selected_quaternion_never_leaves_last_pose_visible(value):
    motion, now = calibrated()
    invalid = frame(now + .05)
    invalid["sensors"][1]["q"] = value
    motion.update(invalid, now=now + .05)
    pose = motion.pose(now=now + .05)
    assert not pose["valid"] and pose["points"] is None and pose["segment_quaternions"] is None
    json.dumps(pose, allow_nan=False)
