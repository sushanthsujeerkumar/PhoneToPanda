"""Wire compatibility and invalid-data boundaries, without serial hardware."""
import json
import struct
import zlib

import pytest

from arm_checker.protocol import (
    DEFAULT_FLAGS, HEADER, SENSOR, StreamDecoder, cobs_decode, cobs_encode,
    decode_json_line, decode_packet, encode_frame, encode_packet, normalize_sensor,
)


def sample(identity="upper", timestamp=1_000_000, **changes):
    return {"id": identity, "t_us": timestamp, "q": [1, 0, 0, 0], **changes}


def wire_frame():
    # Construct independently of encode_frame to verify record order and units.
    records = [struct.pack("<Q4f3f3fHBBI", 990_000, 1, 0, 0, 0,
                           0, 0, 1, .01, .02, .03, 31, 0x68, 2, index)
               for index in range(5)]
    data = struct.pack("<2sBBIQ", b"AM", 1, 1, 42, 1_000_000) + b"".join(records)
    return cobs_encode(data + struct.pack("<I", zlib.crc32(data))) + b"\0"


def test_am_v1_independent_fixture_and_round_trip():
    packet = wire_frame()
    assert HEADER.size == 16 and SENSOR.size == 56
    assert len(cobs_decode(packet[:-1])) == 300
    frame = decode_packet(packet[:-1])
    assert frame["seq"] == 42 and frame["t_us"] == 1_000_000
    assert [s["id"] for s in frame["sensors"]] == ["0", "1", "2", "3", "4"]
    right = frame["sensors"][3]
    assert right["name"] == "right_upper_arm" and right["ok"]
    assert right["q"] == [1, 0, 0, 0]
    assert right["g"] == pytest.approx([.01, .02, .03])
    assert right["a"] == [0, 0, 1] and right["age_ms"] == 10
    assert right["errors"] == 3 and right["whoami"] == 0x68 and right["mode"] == 2
    assert decode_packet(encode_frame(frame)[:-1]) == frame


def test_am_partial_reads_corrupt_crc_and_recovery():
    good = wire_frame()
    raw = bytearray(cobs_decode(good[:-1]))
    raw[20] ^= 1
    corrupt = cobs_encode(raw) + b"\0"
    decoder = StreamDecoder()
    assert decoder.feed(good[:83]) == []
    assert len(decoder.feed(good[83:] + corrupt + b"invalid\0" + good)) == 2
    assert decoder.bad_packets == 2
    assert not decoder.buffer


@pytest.mark.parametrize("data", [b"", b"\0", b"abc\0def", bytes(range(256)), b"x" * 600])
def test_cobs_round_trip(data):
    encoded = cobs_encode(data)
    assert b"\0" not in encoded
    assert cobs_decode(encoded) == data


def test_status_packet_preserves_fusion_generation():
    status = {"orientation_generation": [0, 0, 0, 2, 0], "event": "sensor_init"}
    decoded = decode_packet(encode_packet(2, 3, 9, json.dumps(status).encode())[:-1])
    assert decoded == {"type": "device_status", "seq": 3, "t_us": 9, "status": status}


def test_ndjson_optional_fields_custom_ids_and_normalization():
    sensors = [sample("upper", q=[1.01, 0, 0, 0], gyro=[0, 0, .05]), sample(17)]
    frame = decode_json_line(json.dumps({"sensors": sensors}))
    assert frame["t_us"] == 1_000_000 and frame["seq"] is None
    assert frame["sensors"][0]["q"] == [1, 0, 0, 0]
    assert frame["sensors"][0]["g"] == [0, 0, .05]
    assert frame["sensors"][1]["id"] == "17"
    assert frame["sensors"][1]["a"] is None and frame["sensors"][1]["g"] is None
    assert all(s["ok"] and s["flags"] == DEFAULT_FLAGS for s in frame["sensors"])


@pytest.mark.parametrize("changes,reason", [
    ({"q": [0, 0, 0, 0]}, "norm"),
    ({"q": [2, 0, 0, 0]}, "norm"),
    ({"q": None}, "norm"),
    ({"flags": 27}, "valid"),
    ({"flags": 31 | 32}, "calibration"),
    ({"flags": 31 | 64}, "error"),
    ({"flags": 31 | 128}, "error"),
    ({"t_us": 900_000}, "stale"),
    ({"t_us": 1_000_001}, "ahead"),
])
def test_unhealthy_sensor_is_retained_and_cannot_become_valid(changes, reason):
    frame = decode_json_line(json.dumps({"t_us": 1_000_000,
                                        "sensors": [sample(**changes), sample("forearm")]}))
    unhealthy, healthy = frame["sensors"]
    assert not unhealthy["ok"] and reason in unhealthy["reason"]
    assert healthy["ok"]
    assert not normalize_sensor(unhealthy, frame["t_us"])["ok"]


@pytest.mark.parametrize("sensors", [
    [sample("3"), sample(3)],
    [sample(q=[float("nan"), 0, 0, 0])],
    [sample(a=[0, 0, float("inf")])],
    [sample(q=[1, 0, 0])],
    [sample(t_us=-1)],
    [sample(id=True)],
    [sample(g=[0, 0, 0], gyro=[0, 0, 0])],
    [None], [],
])
def test_malformed_json_frames_are_rejected_then_stream_recovers(sensors):
    decoder = StreamDecoder("ndjson")
    invalid = json.dumps({"sensors": sensors}).encode() + b"\n"
    valid = json.dumps({"sensors": [sample()]}).encode() + b"\n"
    assert len(decoder.feed(invalid + valid)) == 1
    assert decoder.bad_packets == 1


def test_bounded_decoder_and_fragmented_json():
    decoder = StreamDecoder("ndjson", max_buffer=300)
    assert decoder.feed(b"x" * 301) == []
    assert decoder.bad_packets == 1 and not decoder.buffer
    assert decoder.feed(b" \r\n") == []
    valid = json.dumps({"sensors": [sample()]}).encode() + b"\n"
    assert not decoder.feed(valid[:13])
    assert len(decoder.feed(valid[13:])) == 1
    decoder.reset()
    assert decoder.bad_packets == 0 and not decoder.buffer


def test_overflowed_json_number_is_rejected():
    decoder = StreamDecoder("ndjson")
    assert not decoder.feed(b'{"sensors":[],"unknown":1e999}\n')
    assert decoder.bad_packets == 1


@pytest.mark.parametrize("format", ["am_v1", "ndjson"])
def test_overflow_discards_entire_packet_including_valid_looking_suffix(format):
    decoder = StreamDecoder(format, max_buffer=400)
    valid = wire_frame() if format == "am_v1" else json.dumps({"sensors": [sample()]}).encode() + b"\n"
    assert not decoder.feed(b"x" * 401)
    assert not decoder.feed(b"y" * 401)
    assert not decoder.feed(valid)  # No delimiter separated this suffix from x/y.
    assert decoder.bad_packets == 1
    assert len(decoder.feed(valid)) == 1


def test_am_buffer_minimum_includes_cobs_overhead():
    with pytest.raises(ValueError, match="302"):
        StreamDecoder("am_v1", max_buffer=300)
    decoder = StreamDecoder("am_v1", max_buffer=302)
    assert len(decoder.feed(wire_frame())) == 1


def test_am_recovers_exact_crc_valid_frame_suffix_after_noise_prefix():
    decoder = StreamDecoder()
    # Reproduce the 38-byte unsolicited prefix seen before otherwise intact
    # hardware frames, using synthetic data rather than a user's recording.
    prefixed = b"\xad\xa9" * 19 + wire_frame()
    assert decoder.feed(prefixed[:80]) == []
    recovered = decoder.feed(prefixed[80:])
    assert recovered == [decode_packet(wire_frame()[:-1])]
    assert decoder.bad_packets == 1 and decoder.recovered_frames == 1
    decoder.reset()
    assert decoder.bad_packets == 0 and decoder.recovered_frames == 0


def test_am_recovers_302_byte_encoded_frame_suffix():
    # All-nonzero raw bytes force an extra COBS block; the parser may still mark
    # the synthetic quaternion invalid, but framing and CRC remain exact.
    timestamp = 0x0101010101010101
    raw = struct.pack("<2sBBIQ", b"AM", 1, 1, 0x01010101, timestamp) + b"\x01" * 280
    packet = cobs_encode(raw + struct.pack("<I", zlib.crc32(raw)))
    assert len(packet) == 302
    decoder = StreamDecoder()
    assert len(decoder.feed(b"\xad" * 38 + packet + b"\0")) == 1
    assert decoder.bad_packets == decoder.recovered_frames == 1


@pytest.mark.parametrize("damage", ["crc", "partial", "wrong_type", "wrong_header", "trailing_noise", "embedded_not_suffix"])
def test_am_suffix_recovery_rejects_corruption_or_non_frame_packets(damage):
    raw = bytearray(cobs_decode(wire_frame()[:-1]))
    if damage == "crc":
        raw[40] ^= 1
    elif damage == "wrong_type":
        raw[3] = 2
        raw[-4:] = struct.pack("<I", zlib.crc32(raw[:-4]))
    elif damage == "wrong_header":
        raw[0] = ord("B")
        raw[-4:] = struct.pack("<I", zlib.crc32(raw[:-4]))
    packet = cobs_encode(raw)
    if damage == "partial":
        packet = packet[:-4]
    elif damage == "trailing_noise":
        packet += b"\xad"
    elif damage == "embedded_not_suffix":
        packet += b"\xad" * 40
    decoder = StreamDecoder()
    assert decoder.feed(b"\xad" * 38 + packet + b"\0") == []
    assert decoder.bad_packets == 1 and decoder.recovered_frames == 0


def test_status_is_not_suffix_recovered_and_normal_status_still_decodes():
    # A valid status can itself occupy301 encoded bytes. Type checking must
    # distinguish it from the only kind allowed for suffix recovery.
    status = None
    for length in range(250, 290):
        candidate = encode_packet(2, 1, 1_000_000, json.dumps({"padding": "x" * length}).encode())
        if len(candidate) - 1 in (301, 302):
            status = candidate
            break
    assert status is not None
    decoder = StreamDecoder()
    assert decoder.feed(b"\xad" * 38 + status) == []
    assert decoder.bad_packets == 1 and decoder.recovered_frames == 0
    assert decoder.feed(status)[0]["type"] == "device_status"


@pytest.mark.parametrize("split", [False, True])
def test_am_overflow_never_recovers_a_valid_suffix(split):
    decoder = StreamDecoder(max_buffer=400)
    prefix = b"\xad" * 401
    if split:
        assert not decoder.feed(prefix)
        assert not decoder.feed(wire_frame())
    else:
        assert not decoder.feed(prefix + wire_frame())
    assert decoder.bad_packets == 1 and decoder.recovered_frames == 0
    assert len(decoder.feed(wire_frame())) == 1
