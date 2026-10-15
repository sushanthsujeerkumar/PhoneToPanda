"""Watch one experimental Panda replay locally, then inspect its saved video.

The browser only observes. The existing simulation controller produces the
actions, physics, episode and success report. This is not a learned-policy run.
"""
from __future__ import annotations

import argparse
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import sys
import threading
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>IMU pilot · Panda replay</title>
<style>
:root{color-scheme:dark;font-family:Inter,system-ui,sans-serif;background:#10151d;color:#e9eef5}
*{box-sizing:border-box}body{margin:0}main{max-width:1180px;margin:auto;padding:32px 28px}
header{display:flex;gap:16px;justify-content:space-between;align-items:center;margin-bottom:24px}
.eyebrow{font-size:12px;letter-spacing:.15em;color:#81b9be;text-transform:uppercase}h1{font-size:30px;margin:7px 0}
.sub{color:#a9b4c4;font-size:14px;line-height:1.6;margin:0;max-width:760px}
.badge{border:1px solid #344654;border-radius:24px;padding:9px 14px;color:#9fd8d3;white-space:nowrap;font-size:12px}
.layout{display:grid;grid-template-columns:minmax(0,1fr) 290px;gap:20px}
.panel{background:#19212d;border:1px solid #2c3849;border-radius:15px;overflow:hidden}
.title{padding:14px 18px;font-size:13px;color:#a9b4c4;border-bottom:1px solid #2c3849}
.image-stage{background:#080c11;aspect-ratio:1;display:grid;place-items:center}
.image-stage img{width:100%;height:100%;object-fit:contain;display:none}#preparing{color:#9caabe;font-size:15px}
.side .panel{margin-bottom:18px}.wrist{width:100%;display:none}.readout{padding:19px}.label{color:#93a4b9;font-size:12px;margin:0 0 5px}
#phase{font-size:21px;line-height:1.3;margin:0 0 22px}#clock{font-size:27px;font-variant-numeric:tabular-nums;margin:0 0 20px}
#result{font-size:15px;line-height:1.5;margin:0}#failure{font-size:13px;line-height:1.5;color:#efbaa7;overflow-wrap:anywhere}
.note{padding:17px 0;font-size:13px;line-height:1.6;color:#9fadc0}.note strong{color:#d4dde8}
#videoPanel{display:none}video{display:block;width:100%;background:#080c11}button{border:1px solid #40566b;background:#243548;color:#e9eef5;border-radius:8px;padding:9px 12px;cursor:pointer;margin-top:14px}
@media(max-width:800px){main{padding:20px 15px}.layout{grid-template-columns:1fr}.side{display:grid;grid-template-columns:1fr 1fr;gap:12px}.badge{display:none}h1{font-size:25px}}
</style></head><body><main>
<header><div><div class="eyebrow">PhoneToPanda / experimental extension</div><h1>IMU-assisted Panda replay</h1>
<p class="sub">A recorded arm path drives the carry segment. A scripted controller handles approach, grasp and release.</p></div><div class="badge" id="badge">Preparing physics</div></header>
<div class="layout"><section>
<div class="panel" id="livePanel"><div class="title">Simulator camera · live physics frames</div><div class="image-stage"><span id="preparing">Waiting for the first simulated frame…</span><img id="frame" alt="Live Panda simulator camera"></div></div>
<div class="panel" id="videoPanel"><div class="title">Saved replay · simulator and wrist cameras · looping playback</div><video id="video" controls muted loop playsinline></video></div>
<p class="note"><strong>Controller replay, not a trained policy.</strong> Success means this replay passed the simulator's lift-and-place check. It does not establish model performance, tracking accuracy or precise video–IMU synchronization.</p>
</section><aside class="side"><div class="panel"><div class="readout"><p class="label">Current phase</p><p id="phase">Preparing</p><p class="label">Simulation time</p><p id="clock">0.0 s</p><p id="result">Loading the estimated wrist path and simulator.</p><p id="failure"></p><button id="toggle" hidden>Show final live frame</button></div></div>
<div class="panel"><div class="title">Wrist camera</div><img class="wrist" id="wrist" alt="Panda wrist camera"></div></aside></div>
</main><script>
const $=id=>document.getElementById(id);let lastFrame=-1,videoReady=false,showVideo=false;
function selectVideo(value){showVideo=value;$('videoPanel').style.display=value?'block':'none';$('livePanel').style.display=value?'none':'block';$('toggle').textContent=value?'Show final live frame':'Play saved replay';if(value)$('video').play().catch(()=>{});else $('video').pause();}
$('toggle').onclick=()=>selectVideo(!showVideo);
async function poll(){try{const response=await fetch('/state',{cache:'no-store'});if(!response.ok)throw Error();const state=await response.json();
 $('phase').textContent=state.phase.replaceAll('_',' ');$('clock').textContent=state.simulation_seconds.toFixed(1)+' s';
 $('badge').textContent=state.status==='running'?'Live controller replay':state.status==='preparing'?'Preparing physics':state.success?'Replay succeeded':'Replay did not succeed';
 $('result').textContent=state.status==='running'?'MuJoCo contact physics · no learned policy':state.status==='preparing'?'Loading the estimated wrist path and simulator.':state.success?'The lift-and-place check passed.':'This attempt was rejected; inspect the reason below.';
 $('failure').textContent=state.failure||'';
 if(state.frames>0&&state.frames!==lastFrame){lastFrame=state.frames;for(const [id,path] of [['frame','/frame.jpg'],['wrist','/wrist.jpg']]){$(id).src=path+'?frame='+lastFrame;$(id).style.display='block';}$('preparing').style.display='none';}
 if(state.has_video&&!videoReady){videoReady=true;$('video').src='/replay.mp4';$('toggle').hidden=false;selectVideo(true);}
 }catch(error){$('badge').textContent='Viewer disconnected';}setTimeout(poll,150);}
poll();</script></body></html>"""


class ReplayView:
    """Share immutable JPEG snapshots between physics and HTTP threads."""

    def __init__(self, output: Path):
        self.output = output
        self.lock = threading.Lock()
        self.images: dict[str, bytes] = {}
        self.state = {"status": "preparing", "phase": "preparing", "frames": 0,
                      "simulation_seconds": 0., "success": None, "failure": None,
                      "has_video": False}
        self.next_frame_at = None

    def observe(self, env, observation: dict, phase: str):
        """Called after a real physics step; never modifies the observation."""
        from PIL import Image
        images = {}
        for route, key in (("/frame.jpg", "observation.images.image"),
                           ("/wrist.jpg", "observation.images.image2")):
            buffer = io.BytesIO()
            Image.fromarray(observation[key]).save(buffer, format="JPEG", quality=88)
            images[route] = buffer.getvalue()
        with self.lock:
            self.images = images
            self.state.update(status="running", phase=phase, frames=int(env.steps),
                              simulation_seconds=env.steps / env.control_hz)
        # Pace the display to simulation time, without changing simulated actions.
        now = time.monotonic()
        if self.next_frame_at is not None:
            time.sleep(max(0., self.next_frame_at - now))
        self.next_frame_at = time.monotonic() + 1 / env.control_hz

    def finish(self, report: dict):
        with self.lock:
            self.state.update(status="finished", phase="finished", success=bool(report.get("success")),
                              failure=report.get("failure"),
                              has_video=bool(report.get("video") and self.output.with_suffix(".mp4").is_file()))


def make_handler(view: ReplayView):
    """Expose only this replay's pages, current images and optional video."""
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, data: bytes, content_type: str, status=200, extra_headers=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for key, value in (extra_headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/":
                return self.reply(HTML.encode(), "text/html; charset=utf-8")
            with view.lock:
                state = dict(view.state)
                image = view.images.get(path)
            if path == "/state":
                return self.reply(json.dumps(state).encode(), "application/json")
            if path in {"/frame.jpg", "/wrist.jpg"}:
                return self.reply(image or b"Frame not ready", "image/jpeg" if image else "text/plain", 200 if image else 503)
            if path == "/replay.mp4" and state["has_video"]:
                # A single byte range allows the browser's video seek control.
                video = view.output.with_suffix(".mp4")
                size = video.stat().st_size
                start, end, status = 0, size - 1, 200
                headers = {"Accept-Ranges": "bytes"}
                requested = self.headers.get("Range")
                if requested:
                    match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
                    if not match or not any(match.groups()):
                        return self.reply(b"Invalid range", "text/plain", 416, {"Content-Range": f"bytes */{size}"})
                    first, last = match.groups()
                    start = int(first) if first else max(0, size - int(last))
                    end = min(size - 1, int(last)) if first and last else size - 1
                    if not 0 <= start <= end < size:
                        return self.reply(b"Invalid range", "text/plain", 416, {"Content-Range": f"bytes */{size}"})
                    status = 206
                    headers["Content-Range"] = f"bytes {start}-{end}/{size}"
                with video.open("rb") as stream:
                    stream.seek(start)
                    data = stream.read(end - start + 1)
                return self.reply(data, "video/mp4", status, headers)
            self.reply(b"Not found", "text/plain", 404)
    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--http-port", type=int, default=8767)
    parser.add_argument("--start-delay", type=float, default=0., help="Seconds to open the browser before physics starts")
    args = parser.parse_args(argv)
    if args.start_delay < 0:
        parser.error("--start-delay must not be negative")
    output = args.output.expanduser().resolve().with_suffix(".npz")
    for suffix in (".npz", ".json", ".mp4"):
        if output.with_suffix(suffix).exists():
            parser.error(f"Refusing to overwrite {output.with_suffix(suffix)}")
    if not args.trajectory.is_file():
        parser.error("The trajectory file does not exist")
    from phone_to_panda.common import load_config
    config = load_config(args.config)
    config.setdefault("simulation", {})["save_replay_video"] = True
    view = ReplayView(output)
    server = ThreadingHTTPServer(("127.0.0.1", args.http_port), make_handler(view))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Watch experimental controller replay: http://127.0.0.1:{args.http_port}/", flush=True)
    print("The viewer stays open after completion. Press Ctrl-C to close it.", flush=True)
    try:
        time.sleep(args.start_delay)
        from phone_to_panda.simulation import record_demo
        try:
            report = record_demo(config, output, trajectory_path=args.trajectory,
                                 seed=args.seed, observer=view.observe)
        except Exception as exc:
            report = {"success": False, "failure": str(exc), "video": None}
        view.finish(report)
        print(json.dumps(report, indent=2), flush=True)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
