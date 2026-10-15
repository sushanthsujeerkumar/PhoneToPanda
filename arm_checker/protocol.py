"""Read sensor data without opening a port or writing firmware commands.

AM v1 uses zero-terminated COBS packets. Decoded bytes are a little-endian
``<2sBBIQ`` header (AM, version 1, kind, sequence, transmit microseconds),
five ``<Q4f3f3fHBBI`` sensor records, and IEEE CRC32. Sensor records contain
read time, quaternion wxyz, acceleration in g, gyro in rad/s, flags, chip ID,
fusion mode and error count. Binary IDs are strings "0" through "4".

For other boards, send one JSON object per line::

    {"sensors": [{"id": "upper", "t_us": 123000, "q": [1, 0, 0, 0]},
                 {"id": "forearm", "t_us": 123000, "q": [1, 0, 0, 0]}]}

Each line contains both selected sensors. Optional frame fields are type
(``frame``), seq and t_us; t_us defaults to the newest sample time. Optional
sensor fields are a (g), g or gyro (rad/s), and the AM flags bitmask. If flags
are omitted, the adapter is responsible for sending initialized, valid data.
Quaternions are Hamilton, scalar first, and rotate sensor vectors into a
right-handed, Z-up world. Timestamps share a device clock, not the PC clock.
Small quaternion norm errors are normalized; norms outside [0.8, 1.2] are invalid.
"""
from __future__ import annotations

import json
import math
import struct
import zlib

HEADER = struct.Struct("<2sBBIQ")
SENSOR = struct.Struct("<Q4f3f3fHBBI")
NAMES = ("chest", "left_upper_arm", "left_forearm", "right_upper_arm", "right_forearm")
PRESENT, INITIALIZED, VALID, FRESH, BIAS_READY = 1, 2, 4, 8, 16
CALIBRATING, FIFO_OVERFLOW, I2C_ERROR, STATIONARY = 32, 64, 128, 256
REQUIRED = PRESENT | INITIALIZED | VALID | BIAS_READY
DEFAULT_FLAGS = REQUIRED | FRESH
DEVICE_STALE_MS = 100.0


def sensor_id(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("Sensor ID must be a string or integer")
    value = str(value).strip()
    if not value or len(value) > 80:
        raise ValueError("Sensor ID must contain 1 to 80 characters")
    return value


def _integer(value, name, maximum=2**64 - 1):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise ValueError(f"{name} must be a nonnegative integer no larger than {maximum}")
    return value


def _vector(value, size, name):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{name} must contain {size} finite numbers")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        raise ValueError(f"{name} must contain {size} finite numbers")
    return [float(x) for x in value]


def normalize_sensor(sample, frame_t_us):
    """Return JSON-safe data and health; an invalid quaternion is never a pose."""
    if not isinstance(sample, dict):
        raise ValueError("Each sensor must be an object")
    identity = sensor_id(sample.get("id"))
    timestamp = _integer(sample.get("t_us"), "Sensor t_us")
    flags = _integer(sample.get("flags", DEFAULT_FLAGS), "Sensor flags", 65535)
    if "q" not in sample:
        raise ValueError("Sensor is missing quaternion wxyz")
    q = None if sample["q"] is None else _vector(sample["q"], 4, "Quaternion wxyz")
    a = None if sample.get("a") is None else _vector(sample["a"], 3, "Acceleration")
    if "g" in sample and "gyro" in sample:
        raise ValueError("Use either g or gyro, not both")
    gyro = sample.get("g", sample.get("gyro"))
    g = None if gyro is None else _vector(gyro, 3, "Gyro")
    norm = math.hypot(*q) if q is not None else 0.0
    reasons = []
    if not .8 <= norm <= 1.2:
        q, flags = None, flags & ~VALID
        reasons.append("Quaternion norm is outside 0.8 to 1.2")
    else:
        q = [x / norm for x in q]
    if (flags & (REQUIRED | FRESH)) != (REQUIRED | FRESH):
        reasons.append("Sensor is not initialized, valid, bias-ready and fresh")
    if flags & (CALIBRATING | FIFO_OVERFLOW | I2C_ERROR):
        reasons.append("Sensor reports calibration or a read error")
    age_ms = max(0, frame_t_us - timestamp) / 1000.0 if timestamp else 1e9
    if timestamp > frame_t_us:
        reasons.append("Sample time is ahead of frame time")
    if age_ms >= DEVICE_STALE_MS:
        reasons.append("Device sample is stale")
    return {"id": identity, "name": str(sample.get("name", identity)), "t_us": timestamp,
            "q": q, "a": a, "g": g, "flags": flags, "age_ms": age_ms,
            "ok": not reasons, "reason": "; ".join(reasons),
            "whoami": sample.get("whoami"), "mode": sample.get("mode"),
            "errors": _integer(sample.get("errors", 0), "Error count", 2**32 - 1)}


def cobs_encode(data: bytes) -> bytes:
    output, marker, count = bytearray([0]), 0, 1
    for value in data:
        if value == 0:
            output[marker] = count
            marker, count = len(output), 1
            output.append(0)
        else:
            output.append(value)
            count += 1
            if count == 255:
                output[marker] = count
                marker, count = len(output), 1
                output.append(0)
    output[marker] = count
    return bytes(output)


def cobs_decode(data: bytes) -> bytes:
    output, position = bytearray(), 0
    while position < len(data):
        code = data[position]
        position += 1
        if code == 0 or position + code - 1 > len(data):
            raise ValueError("Invalid COBS packet")
        output.extend(data[position:position + code - 1])
        position += code - 1
        if code != 255 and position < len(data):
            output.append(0)
    return bytes(output)


def encode_packet(kind, seq, t_us, payload):
    """Encode fixtures or adapters; this function performs no device I/O."""
    raw = HEADER.pack(b"AM", 1, kind, seq & 0xffffffff, t_us) + payload
    return cobs_encode(raw + struct.pack("<I", zlib.crc32(raw))) + b"\0"


def encode_frame(frame):
    if len(frame["sensors"]) != 5:
        raise ValueError("AM v1 requires exactly five sensor records")
    payload = b"".join(SENSOR.pack(s["t_us"], *s["q"], *s["a"], *s["g"],
                                  s["flags"], s.get("whoami", 0), s.get("mode", 0), s.get("errors", 0))
                       for s in frame["sensors"])
    return encode_packet(1, frame["seq"], frame["t_us"], payload)


def _json_object(data):
    def invalid_constant(value):
        raise ValueError(f"Non-finite JSON value: {value}")
    result = json.loads(data, parse_constant=invalid_constant)
    if not isinstance(result, dict):
        raise ValueError("Packet must contain a JSON object")
    json.dumps(result, allow_nan=False)  # Also reject overflowing exponents.
    return result


def decode_packet(encoded):
    """Decode one AM packet without its terminating zero byte."""
    raw = cobs_decode(encoded)
    if len(raw) < HEADER.size + 4:
        raise ValueError("Short AM packet")
    if zlib.crc32(raw[:-4]) != struct.unpack("<I", raw[-4:])[0]:
        raise ValueError("CRC mismatch")
    magic, version, kind, seq, timestamp = HEADER.unpack_from(raw)
    if magic != b"AM" or version != 1:
        raise ValueError("Unknown AM protocol")
    payload = raw[HEADER.size:-4]
    if kind == 2:
        return {"type": "device_status", "seq": seq, "t_us": timestamp, "status": _json_object(payload)}
    if kind != 1 or len(payload) != 5 * SENSOR.size:
        raise ValueError("AM frame must contain five sensor records")
    sensors = []
    for index, name in enumerate(NAMES):
        values = SENSOR.unpack_from(payload, index * SENSOR.size)
        sample = dict(id=str(index), name=name, t_us=values[0], q=list(values[1:5]),
                      a=list(values[5:8]), g=list(values[8:11]), flags=values[11],
                      whoami=values[12], mode=values[13], errors=values[14])
        sensors.append(normalize_sensor(sample, timestamp))
    return {"type": "frame", "format": "am_v1", "seq": seq, "t_us": timestamp, "sensors": sensors}


def decode_json_line(line):
    frame = _json_object(line)
    if frame.get("type", "frame") != "frame":
        if frame.get("type") == "device_status" and isinstance(frame.get("status"), dict):
            return frame
        raise ValueError("NDJSON type must be frame or device_status")
    samples = frame.get("sensors")
    if not isinstance(samples, list) or not 1 <= len(samples) <= 64:
        raise ValueError("NDJSON frame requires 1 to 64 sensor objects")
    if any(not isinstance(sample, dict) for sample in samples):
        raise ValueError("Each sensor must be an object")
    timestamp = frame.get("t_us")
    if timestamp is None:
        timestamp = max(_integer(sample.get("t_us"), "Sensor t_us") for sample in samples)
    timestamp = _integer(timestamp, "Frame t_us")
    sensors = [normalize_sensor(sample, timestamp) for sample in samples]
    if len({sensor["id"] for sensor in sensors}) != len(sensors):
        raise ValueError("Sensor IDs must be unique within a frame")
    seq = frame.get("seq")
    if seq is not None:
        seq = _integer(seq, "Sequence", 2**32 - 1)
    result = {"type": "frame", "format": "ndjson", "seq": seq, "t_us": timestamp, "sensors": sensors}
    if "generation" in frame:
        result["generation"] = frame["generation"]
    return result


class StreamDecoder:
    """Bounded framing with recovery after partial reads or corrupted packets."""
    def __init__(self, format="am_v1", max_buffer=8192):
        if format not in ("am_v1", "ndjson"):
            raise ValueError("Format must be am_v1 or ndjson")
        self.format, self.max_buffer = format, int(max_buffer)
        minimum = 302 if format == "am_v1" else 300
        if self.max_buffer < minimum:
            raise ValueError(f"Decoder buffer must hold at least {minimum} bytes")
        self.buffer = bytearray()
        self.bad_packets = 0
        self.recovered_frames = 0
        self._discarding = False

    def reset(self):
        self.buffer.clear()
        self.bad_packets = 0
        self.recovered_frames = 0
        self._discarding = False

    def _recover_frame(self, packet):
        # Some streams insert non-packet bytes immediately before intact frames.
        # A 300-byte AM frame has only these two possible COBS lengths. Accept
        # only its exact suffix, still requiring the original header and CRC.
        for size in (301, 302):
            if len(packet) <= size:
                continue
            try:
                frame = decode_packet(packet[-size:])
            except (ValueError, TypeError, KeyError, UnicodeError, struct.error, OverflowError):
                continue
            if frame["type"] == "frame":
                return frame
        return None

    def feed(self, data):
        self.buffer.extend(data)
        delimiter = 0 if self.format == "am_v1" else 10
        packets = []
        if self._discarding:
            # Overflow may happen midway through a line or COBS packet. Its
            # remaining suffix is still corrupt, even if it looks parseable.
            if delimiter not in self.buffer:
                self.buffer.clear()
                return packets
            end = self.buffer.index(delimiter)
            del self.buffer[:end + 1]
            self._discarding = False
        while delimiter in self.buffer:
            end = self.buffer.index(delimiter)
            packet = bytes(self.buffer[:end])
            del self.buffer[:end + 1]
            empty = not packet.strip() if self.format == "ndjson" else not packet
            if empty:
                continue
            try:
                if len(packet) > self.max_buffer:
                    raise ValueError("Packet exceeds decoder buffer limit")
                packets.append(decode_packet(packet) if self.format == "am_v1" else decode_json_line(packet))
            except (ValueError, TypeError, KeyError, UnicodeError, struct.error, OverflowError):
                self.bad_packets += 1
                if self.format == "am_v1" and len(packet) <= self.max_buffer:
                    recovered = self._recover_frame(packet)
                    if recovered is not None:
                        self.recovered_frames += 1
                        packets.append(recovered)
        if len(self.buffer) > self.max_buffer:
            self.buffer.clear()
            self.bad_packets += 1
            self._discarding = True
        return packets
