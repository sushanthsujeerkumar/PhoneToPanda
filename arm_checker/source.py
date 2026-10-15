"""Read sensor packets off the UI loop, or generate explicitly synthetic input."""
import math
import threading
import time
import serial
from serial.tools import list_ports
from .protocol import StreamDecoder


def ports():
    return [{"device": item.device, "description": item.description or "Serial device"}
            for item in list_ports.comports()]


class SerialReader:
    """One thread owns the port. No calibration or flash commands are sent."""

    def __init__(self, port, protocol, baud, callback):
        self.port, self.protocol, self.baud = port, protocol, baud
        self.callback = callback
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self.run, daemon=True, name="arm-checker-serial")
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1)

    def resolve_port(self):
        if self.port != "auto":
            return self.port
        matches = [p.device for p in list_ports.comports() if p.vid in (0x1a86, 0x303a, 0x10c4)]
        if len(matches) != 1:
            raise OSError(f"Found {len(matches)} compatible ports. Choose the device from the port list.")
        return matches[0]

    def run(self):
        while not self.stop_event.is_set():
            connection = None
            try:
                device = self.resolve_port()
                connection = serial.Serial(port=None, baudrate=self.baud, timeout=.02)
                connection.dtr = False
                connection.rts = False
                connection.port = device
                connection.open()
                decoder = StreamDecoder(format="am_v1" if self.protocol == "am" else "ndjson")
                self.callback({"type": "connection", "connected": True, "message": device})
                while not self.stop_event.is_set():
                    data = connection.read(max(1, min(connection.in_waiting, 8192)))
                    received = time.monotonic()
                    for frame in decoder.feed(data):
                        frame["received_at"] = received
                        frame["bad_packets"] = decoder.bad_packets
                        frame["recovered_frames"] = decoder.recovered_frames
                        self.callback(frame)
            except (OSError, serial.SerialException, ValueError) as exc:
                self.callback({"type": "connection", "connected": False, "message": str(exc)})
                self.stop_event.wait(1)
            finally:
                if connection:
                    connection.close()


def demo_frame(elapsed, sequence, stationary=False):
    """Synthetic right-arm orientations; never presented as sensor measurements."""
    upper_angle = 0 if stationary else -.6 * (1 - math.cos(elapsed * .8))
    elbow_angle = 0 if stationary else .5 * (1 - math.cos(elapsed * 1.1))
    sensors = []
    for index in range(5):
        angle = upper_angle if index == 3 else upper_angle - elbow_angle if index == 4 else 0
        timestamp = int((elapsed + 1) * 1_000_000)
        sensors.append({"id": str(index), "name": ("chest", "left_upper_arm", "left_forearm", "right_upper_arm", "right_forearm")[index],
                        "t_us": timestamp, "q": [math.cos(angle / 2), 0, math.sin(angle / 2), 0],
                        "a": None, "g": None, "flags": 31, "age_ms": 0., "ok": True,
                        "reason": "synthetic", "whoami": None})
    return {"type": "frame", "seq": sequence, "t_us": timestamp, "sensors": sensors,
            "synthetic": True}
