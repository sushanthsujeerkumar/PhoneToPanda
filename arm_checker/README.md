# Right-arm IMU checker

A local browser viewer for checking two wearable orientation sensors on the
right upper arm and forearm. It shows sensor health, calibration and a simple
two-link arm, and saves sessions for replay. It is separate from the
PhoneToPanda training and simulation pipeline.

For camera placement, mounting and a proposed paired capture procedure, use the
[illustrated recording guide](guide/PhoneToPanda_Phone_IMU_Recording_Guide.pdf).
The earlier [compact reference](guide/recording_guide.pdf) remains available.

The viewer uses a **fixed shoulder** and user-supplied segment lengths. It does
not measure shoulder translation, solve Panda inverse kinematics or estimate
independent hand and finger poses. Physical tracking accuracy, drift and
movement-to-screen latency still need measurement.

## Quick start without hardware

Requirements: Python 3.11 or 3.12, Bash and a browser with WebGL. The launcher
installs this checker's Python dependencies on first use; that installation
requires internet access. The browser assets are included locally. No trained
model, GPU or MuJoCo installation is required.

From the repository root:

```sh
bash arm_checker/run.sh --demo
```

Or from this directory:

```sh
bash run.sh --demo
```

Open **http://127.0.0.1:8765/**. Demo input is synthetic and is labelled as such;
it does not establish hardware accuracy. Stop the server with Ctrl+C. The Bash
launcher is intended for Linux/macOS; Windows use has not been verified.

## Use the existing ESP32 rig

The [wiring images](hardware_setup.md#right-arm-wiring) show the two right-arm
sensors, multiplexer connections and forearm address jumper.

Close any other serial viewer first. Keep the existing firmware, USB connection
and wiring unchanged:

```sh
bash arm_checker/run.sh --port auto --protocol am
```

`--port auto --protocol am` is the live default. The AM adapter reads the existing
five-record binary stream at 921600 baud and selects sensor **3** for the right
upper arm and **4** for the right forearm. The other records are not arm inputs.
Automatic port discovery does not prove that a sensor is mounted on the correct
segment. Use an explicit device path if discovery is ambiguous, for example
`--port /dev/ttyACM0`; the actual path depends on the computer.

1. Confirm the two selected sensors respond when moved separately. Check their
   IDs before mounting; do not compensate for a swapped sensor by reversing an
   angle in the display.
2. Keep the sensors still while the firmware completes its startup bias capture
   (ten continuous seconds on the existing rig).
3. Lay both flat, chip side up, with both +X axes pointing toward your intended
   forward direction. Perform **Table alignment**, holding still until it
   finishes (at least two seconds).
4. Mount one sensor securely on the upper arm and the other on the forearm.
   Keep the ESP32 powered. Stand still with the right arm straight down, then
   perform **Neutral pose** and wait for completion (at least three seconds).
5. Enter measured upper-arm and forearm lengths in centimetres. Check slow
   movements and return to neutral before recording. Detailed checks are in
   [hardware_setup.md](hardware_setup.md).

Changing the input source, sensor assignments or device orientation identity
requires fresh calibration. Repeat both steps after remounting or a firmware
restart. A brief stale sample is a data-quality fault; a held pose must not be
interpreted as a current measurement.

## Other orientation sources

Use `--protocol json` for newline-delimited JSON supplied over serial. Select
the corresponding sensor IDs in the viewer. The adapter requires explicit
sensor timestamps and Hamilton `[w, x, y, z]` quaternions that actively rotate
sensor vectors into a right-handed, Z-up world frame. It does not convert
arbitrary Euler-angle streams or perform IMU fusion for a new board.

See the [JSON adapter contract](hardware_setup.md#json-adapter) before connecting
another device. Missing optional health or motion fields cannot become evidence
that its measurements are valid.

## Record and replay

After checking calibration and sensor health, click **Start recording · 5s**.
Choose **Record for**: 10, 15 (default), or 30 seconds. Use 15 seconds for a short
practice take, or 30 seconds for the guide's full 28-second movement sequence.
The browser shows a five-second countdown so you can start your phone video.
No sensor frames are saved during the countdown; **Cancel countdown** stops it.
Keep the page open. Reloading the page or losing the selected sensors' fresh
signal cancels the countdown. Setup and calibration changes also cancel it.
Once **Stop & save** appears, recording is active. Hold neutral for three seconds,
then perform the elbow timing cue. The countdown does not synchronize the phone
and sensors; use the visible movement to align them afterward.

After the selected recording time, the browser requests **Stop & save**. The
five-second preparation countdown is additional to that time. Keep the page
open and the computer awake; closing the page prevents its automatic stop.
The target time is approximate because browser scheduling and local network
requests add latency. Saved session timestamps give the actual duration.
Reloading this same tab restores its pending stop timer when browser storage is
available. A failed save is retried while the page is open, after checking the
session identity. Use one control tab at a time. Stop the phone video separately.

Stop to save the session. Select a saved session to replay it; return to live input when
finished. A recording is capped at five minutes or 30,000 frames. Stop recording
before changing the source, setup or calibration. Files stay on this computer;
the checker does not upload them. The default directory is `arm_checker/sessions/`;
use `--sessions-dir PATH` to choose another location.

Each session is a JSON file containing metadata and frames with decoded device
fields, host elapsed/monotonic times, processed poses and calibration context.
The download control exports that session for inspection. Demo sessions remain
identified as synthetic. Replay displays saved poses rather than applying the
current live calibration to old sensor readings.

Use **Delete** beside a saved session, or **Delete all** above the list, then
confirm the selection. Deletion is disabled while recording or counting down.
The recording and its metadata move together into a batch folder under
`arm_checker/sessions/.trash/` (or `.trash/` in your chosen sessions directory).
These files are recoverable by moving both JSON files back to the sessions
directory and refreshing the list. Deleting saved sessions does not change
calibration. The app's Trash folder is separate from the desktop Trash.

ESP32 and host clocks are stored separately and are **not synchronized**. The
ESP32 sensor times estimate sequential register-read times, not simultaneous
hardware acquisition. There is no camera synchronization in this checker.
Sample rate and browser refresh rate are useful diagnostics, not measurements
of physical motion-to-screen delay.

## Interpretation and validation

The model uses +X forward, +Y left and +Z up, with its shoulder at the origin.
Calibrated upper-arm and forearm orientations determine the two segment
directions and an angle between them. Segment endpoints are kinematic estimates
from the chosen lengths, not independently measured positions. Without a torso
sensor, trunk motion is not separated from arm motion.

Six-axis IMUs observe gravity and angular motion, but not absolute heading.
Yaw drift and mounting movement can change the displayed pose. A prior version
of the five-sensor rig had a reported inward/outward reversal; its physical cause
has not been verified or claimed fixed here. The one-sensor-at-a-time identity
and direction checks remain necessary.

Software tests and demo playback check the application and its mathematics.
They do not establish mounted joint accuracy or latency. Keep a physical
validation record before using these sessions as robot-learning data.
See [the dated software checks](VALIDATION.md) for the tests performed here.

To rebuild the illustrated guide, install `reportlab==4.4.9` in a separate Python
environment and run `python arm_checker/guide/build_guide.py` from the repository
root. The script generates its own vector illustrations and uses bundled fonts.

## Find your way through the code

Follow one sample in this order:

| File | What it does |
| --- | --- |
| `source.py` | Reads serial bytes on a background thread, or generates labelled demo input. |
| `protocol.py` | Checks packet framing, timestamps, validity flags and quaternions. |
| `motion.py` | Captures calibration, checks freshness and calculates segment endpoints. |
| `app.py` | Connects input, calibration, recording and the browser API. |
| `recording.py` | Stores raw frames and pose estimates, keeping their clocks separate. |
| `web/app.js` | Updates controls, signal status and recording actions. |
| `web/scene.js` | Draws the server's estimated pose; it does not calculate calibration. |
| `guide/build_guide.py` | Rebuilds the illustrated recording PDF. |

The Python tests use synthetic inputs, including corrupt packets, known arm
angles and stale streams. Run them from the repository root:

```sh
arm_checker/.venv/bin/python -m pip install -r arm_checker/requirements-test.txt
arm_checker/.venv/bin/python -m pytest -q arm_checker/tests
```

The bundled browser libraries keep their upstream license in `web/vendor/`.
