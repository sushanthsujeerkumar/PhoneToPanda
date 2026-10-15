"""Serve the checker locally. Sensor measurements and rendered estimates stay distinct."""
import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
import json
import math
from pathlib import Path
import queue
import time
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .motion import ArmMotion
from .protocol import sensor_id
from .recording import SessionStore
from .source import SerialReader, demo_frame, ports

ROOT = Path(__file__).resolve().parent


class Checker:
    def __init__(self, source="demo", port="auto", protocol="am", baud=921600, sessions_dir=None):
        if source not in ("demo", "serial") or protocol not in ("am", "json"):
            raise ValueError("Choose demo or serial input, with am or json protocol.")
        self.source, self.port, self.protocol, self.baud = source, port, protocol, baud
        self.config = {"upper_id": "3", "forearm_id": "4", "upper_length": .30, "forearm_length": .26}
        self.motion = ArmMotion()
        self.store = SessionStore(sessions_dir or ROOT / "sessions")
        self.reader = None
        self.incoming = queue.Queue(maxsize=200)
        self.connected = source == "demo"
        self.message = "Synthetic input. Calibrate to begin." if source == "demo" else "Waiting for serial input."
        self.sensors = []
        self.last_received = None
        self.last_device_time = None
        self.device_identity = {}
        self.rates = {}
        self.samples = {}
        self.bad_packets = 0
        self.recovered_frames = 0
        self.dropped_host_frames = 0
        self.generation = 0
        self.demo_started = time.monotonic()
        self.sequence = 0
        self.replay = None
        self.replay_index = 0
        self.replay_started = None

    def put(self, generation, packet):
        if generation != self.generation:
            return
        try:
            self.incoming.put_nowait(packet)
        except queue.Full:
            self.dropped_host_frames += 1

    def start_source(self):
        if self.source == "serial":
            generation = self.generation
            self.reader = SerialReader(self.port, self.protocol, self.baud,
                                       lambda packet: self.put(generation, packet))
            self.reader.start()

    def stop_source(self):
        self.generation += 1
        if self.reader:
            self.reader.stop()
            self.reader = None
        while not self.incoming.empty():
            try:
                self.incoming.get_nowait()
            except queue.Empty:
                break

    def require_idle(self):
        if self.store.current:
            raise ValueError("Stop recording before changing the setup.")

    def change_source(self, values):
        self.require_idle()
        source = values.get("source", self.source)
        protocol = values.get("protocol", self.protocol)
        port = values.get("port", self.port)
        if source not in ("demo", "serial") or protocol not in ("am", "json"):
            raise ValueError("Unknown input source or serial protocol.")
        if not isinstance(port, str) or not port.strip() or len(port) > 256:
            raise ValueError("Choose a valid serial port.")
        self.stop_source()
        self.source, self.protocol, self.port = source, protocol, port.strip()
        self.replay = None
        self.motion.invalidate("Input changed. Repeat table alignment and neutral pose.")
        self.sensors, self.rates, self.samples, self.device_identity = [], {}, {}, {}
        self.last_received = self.last_device_time = None
        self.connected = source == "demo"
        self.message = "Synthetic input. Calibrate to begin." if self.connected else "Connecting to serial input."
        self.demo_started = time.monotonic()
        self.start_source()

    def configure(self, values):
        self.require_idle()
        if self.replay:
            raise ValueError("Return to live input before changing the setup.")
        allowed = set(self.config)
        if set(values) - allowed:
            raise ValueError("Unknown configuration field.")
        config = {**self.config, **values}
        for key in ("upper_id", "forearm_id"):
            config[key] = str(config[key]).strip()
            if not config[key] or len(config[key]) > 64:
                raise ValueError("Sensor IDs must be short, non-empty values.")
        if config["upper_id"] == config["forearm_id"]:
            raise ValueError("Assign two different sensors to the arm segments.")
        for key in ("upper_length", "forearm_length"):
            value = config[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not .1 <= value <= .6:
                raise ValueError("Arm lengths must be between 0.10 and 0.60 metres.")
        self.motion.configure(upper_id=config["upper_id"], forearm_id=config["forearm_id"],
                              upper_length_m=config["upper_length"], forearm_length_m=config["forearm_length"])
        self.config = config

    def calibrate(self, step):
        self.require_idle()
        if self.replay:
            raise ValueError("Return to live input before calibrating.")
        if step == "table":
            self.motion.start_table()
        elif step == "neutral":
            self.motion.start_neutral()
        elif step == "reset":
            self.motion.invalidate("Calibration cleared. Align the sensors on the table.")
        else:
            raise ValueError("Choose table, neutral or reset.")

    def ingest(self, frame):
        self.bad_packets = frame.get("bad_packets", self.bad_packets)
        self.recovered_frames = frame.get("recovered_frames", self.recovered_frames)
        kind = frame.get("type")
        if kind == "connection":
            self.connected = bool(frame["connected"])
            self.message = frame["message"]
            self.motion.invalidate("Serial connection changed. Repeat calibration with fresh input.")
            self.last_device_time = None
            return
        if kind == "device_status":
            status = frame.get("status", {})
            reset = False
            if "boot_id" in status:
                if "boot_id" in self.device_identity and self.device_identity["boot_id"] != status["boot_id"]:
                    reset = True
                    self.device_identity.clear()
                self.device_identity["boot_id"] = deepcopy(status["boot_id"])
            selected = set(self.motion.mapping.values())
            try:
                identity = sensor_id(status["sensor"]) if "sensor" in status else None
            except ValueError:
                identity = None
            generation = status.get("orientation_generation")
            # AM summaries contain all sensor generations; individual events
            # contain one scalar plus its sensor ID. They share a per-ID history.
            indexed = dict(enumerate(generation)) if isinstance(generation, list) else (
                {identity: generation} if identity is not None else {})
            known = self.device_identity.setdefault("orientation_generation", {})
            for index, value in indexed.items():
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    continue
                index = str(index)
                if index in selected and index in known and known[index] != value:
                    reset = True
                known[index] = value
            if "sensor" not in status and isinstance(generation, int) and not isinstance(generation, bool) and generation >= 0:
                # Adapters may report one orientation generation for the board.
                key = "global_orientation_generation"
                if key in self.device_identity and self.device_identity[key] != generation:
                    reset = True
                self.device_identity[key] = generation
            event = status.get("event")
            if event in ("bias_start", "rescan_complete"):
                reset = True
            elif event in ("sensor_init", "sample_gap", "dmp_fallback", "dmp_start", "sensor_stalled"):
                reset = reset or identity is None or identity in selected
            if reset:
                self.motion.invalidate("Selected sensor orientation or device reset detected. Repeat both calibration steps.")
            return
        if kind != "frame":
            return
        received = frame.get("received_at", time.monotonic())
        device_time = frame.get("t_us")
        if self.last_device_time is not None and device_time is not None and device_time < self.last_device_time:
            self.motion.invalidate("Device clock restarted. Repeat calibration.")
        self.last_device_time = device_time
        self.last_received = received
        self.bad_packets = frame.get("bad_packets", self.bad_packets)
        self.sensors = deepcopy(frame["sensors"])
        for sensor in self.sensors:
            key = str(sensor["id"])
            sample = sensor.get("t_us")
            old = self.samples.get(key)
            if sample and old and sample > old:
                rate = 1_000_000 / (sample - old)
                self.rates[key] = rate if key not in self.rates else .9 * self.rates[key] + .1 * rate
            self.samples[key] = sample
        self.motion.update(frame, now=received)
        self.store.append(deepcopy(frame), self.pose(received), received)

    def pose(self, now=None):
        pose = self.motion.pose(now=now)
        points = pose.get("points") or {}
        quaternions = pose.get("segment_quaternions") or {}
        return {**pose, "shoulder": points.get("shoulder"), "elbow": points.get("elbow"),
                "wrist": points.get("wrist"), "upper_q": quaternions.get("upper"),
                "forearm_q": quaternions.get("forearm"), "elbow_angle_deg": pose.get("elbow_flexion_deg")}

    def state(self):
        now = time.monotonic()
        calibration = self.motion.status(now=now)
        sensors = []
        host_age = (now - self.last_received) * 1000 if self.last_received is not None else 1e9
        for item in self.sensors:
            sensor = dict(item)
            health = calibration["sensors"].get(str(item["id"]), {})
            sensor.update(id=str(item["id"]), label=item.get("name", str(item["id"])),
                          age_ms=health.get("age_ms", round(max(host_age, item.get("age_ms", 0)), 1)),
                          ok=bool(health.get("ok", False)), reason=health.get("reason", item.get("reason", "")),
                          rate_hz=round(self.rates.get(str(item["id"]), 0.), 1),
                          gyro_dps=math.degrees(math.sqrt(sum(v*v for v in item["g"]))) if item.get("g") else None)
            sensors.append(sensor)
        state = {"source": "replay" if self.replay else self.source, "live_source": self.source,
                 "connected": self.connected, "message": self.message, "config": self.config,
                 "port": self.port, "protocol": self.protocol, "baud": self.baud,
                 "sensors": sensors, "calibration": calibration, "pose": self.pose(now),
                 "recording": self.store.status(), "sessions": self.store.list(),
                 "bad_packets": self.bad_packets, "dropped_host_frames": self.dropped_host_frames,
                 "recovered_frames": self.recovered_frames,
                 "fixed_shoulder": True, "camera_synchronized": False}
        if self.replay:
            saved = self.replay["frames"][self.replay_index]
            state.update(pose=saved["pose"], config=self.replay["metadata"]["config"],
                         sensors=[], calibration={"stage": "replay", "message": "Recorded pose and calibration; live input is not shown.",
                                                   "table_ready": True, "neutral_ready": True, "progress": 1},
                         replay={"id": self.replay["metadata"]["id"], "source": self.replay["metadata"]["source"],
                                 "frame": self.replay_index + 1, "frames": len(self.replay["frames"]),
                                 "finished": self.replay_index == len(self.replay["frames"]) - 1},
                         message="Replay of saved estimates. " + ("Synthetic source." if self.replay["metadata"].get("synthetic") else "Recorded serial input."))
        return state

    def record(self, action):
        if action == "stop":
            return self.store.stop()
        if action != "start":
            raise ValueError("Choose start or stop.")
        if self.replay or not self.pose().get("valid"):
            raise ValueError("Calibrate fresh live or demo input before recording.")
        self.store.start(self.source, self.config, self.motion.snapshot())

    def start_replay(self, session_id):
        self.require_idle()
        path = self.store.path(session_id)
        if not path.is_file():
            raise ValueError("Session not found.")
        session = json.loads(path.read_text())
        if not session.get("frames"):
            raise ValueError("This session contains no frames.")
        self.replay = session
        self.replay_index = 0
        self.replay_started = time.monotonic()

    def delete_sessions(self, session_ids):
        self.require_idle()
        deleted = self.store.delete(session_ids)
        if self.replay and self.replay["metadata"]["id"] in session_ids:
            self.replay = None
            self.replay_index = 0
            self.replay_started = None
        return {"deleted": deleted, "sessions": self.store.list()}

    async def tick(self):
        self.start_source()
        try:
            while True:
                now = time.monotonic()
                if self.source == "demo":
                    status = self.motion.status(now=now)
                    stationary = not status.get("neutral_ready") or status.get("stage") in ("table", "neutral", "table_capture", "neutral_capture")
                    self.sequence += 1
                    self.ingest(demo_frame(now - self.demo_started, self.sequence, stationary=stationary))
                for _ in range(200):
                    try:
                        self.ingest(self.incoming.get_nowait())
                    except queue.Empty:
                        break
                if self.replay:
                    elapsed = now - self.replay_started
                    frames = self.replay["frames"]
                    start = frames[0]["host_elapsed_s"]
                    while self.replay_index + 1 < len(frames) and frames[self.replay_index + 1]["host_elapsed_s"] - start <= elapsed:
                        self.replay_index += 1
                if self.store.current and (now - self.store.started >= 300 or len(self.store.current["frames"]) >= 30000):
                    self.store.stop()
                    self.message = "Recording saved at the five-minute limit."
                await asyncio.sleep(.01)
        finally:
            self.stop_source()
            if self.store.current:
                self.store.stop()


def create_app(source="demo", port="auto", protocol="am", baud=921600, sessions_dir=None):
    checker = Checker(source, port, protocol, baud, sessions_dir)

    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(checker.tick())
        yield
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    app = FastAPI(title="PhoneToPanda Arm Checker", lifespan=lifespan)
    app.state.checker = checker
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "testserver"])

    @app.middleware("http")
    async def same_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and urlparse(origin).netloc != request.headers.get("host"):
            return JSONResponse({"detail": "Use the checker's own local page."}, status_code=403)
        return await call_next(request)

    @app.exception_handler(ValueError)
    async def bad_request(request, error):
        return JSONResponse({"detail": str(error)}, status_code=400)

    @app.get("/api/state")
    async def get_state():
        return checker.state()

    @app.get("/api/ports")
    async def get_ports():
        return {"ports": ports()}

    @app.get("/guide/recording_guide.pdf")
    async def recording_guide():
        return FileResponse(ROOT / "guide" / "PhoneToPanda_Phone_IMU_Recording_Guide.pdf",
                            media_type="application/pdf")

    @app.post("/api/source")
    async def source_change(values: dict):
        checker.change_source(values)
        return checker.state()

    @app.post("/api/config")
    async def configure(values: dict):
        checker.configure(values)
        return checker.state()

    @app.post("/api/calibrate")
    async def calibrate(values: dict):
        checker.calibrate(values.get("step"))
        return checker.state()

    @app.post("/api/record")
    async def record(values: dict):
        checker.record(values.get("action"))
        return checker.state()

    @app.get("/api/sessions")
    async def sessions():
        return {"sessions": checker.store.list()}

    @app.post("/api/sessions/delete")
    async def delete_sessions(values: dict):
        if set(values) != {"ids"}:
            raise ValueError("Provide the explicit saved session IDs to delete.")
        return checker.delete_sessions(values["ids"])

    @app.get("/api/sessions/{session_id}/download")
    async def download(session_id: str):
        path = checker.store.path(session_id)
        if not path.is_file():
            raise HTTPException(404, "Session not found.")
        return FileResponse(path, media_type="application/json", filename=path.name)

    @app.post("/api/replay")
    async def replay(values: dict):
        checker.start_replay(values.get("id"))
        return checker.state()

    @app.post("/api/live")
    async def live():
        checker.replay = None
        return checker.state()

    @app.websocket("/ws")
    async def websocket(ws: WebSocket):
        origin = ws.headers.get("origin")
        if origin and urlparse(origin).netloc != ws.headers.get("host"):
            await ws.close(code=1008)
            return
        await ws.accept()
        try:
            while True:
                await ws.send_json(checker.state())
                await asyncio.sleep(1 / 30)
        except WebSocketDisconnect:
            pass

    app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
    return app
