# A walkthrough of PhoneToPanda

Start here to follow the code from a phone recording to a robot action.
The method is **visual motion retargeting followed by imitation learning**.
It uses a simulated Panda arm, five accepted personal recordings, and a
pretrained SmolVLA model fine-tuned on successful robot replays.

A **policy** maps observations to robot commands. A **checkpoint** saves the
model's weights and settings. An **episode** is one attempt, from reset until
success or timeout. A **seed** initializes random choices so an experiment can
be repeated under the same setup. **Normalization** rescales input and output
values using statistics saved with the model.

```text
Phone video + calibration + grasp/release annotations
  → measured object path in table coordinates
  → scripted, physical Panda replay of that path
  → robot images + robot state + controller actions
  → LeRobot dataset → SmolVLA fine-tuning
  → policy actions in the simulator → measured placement outcome
```

The phone video provides the carried object's path and timing. It does not
contain robot action labels. The replay controller supplies those labels.
During policy evaluation, the replay script no longer chooses the
translation or gripper commands: SmolVLA does, with downward wrist feedback.

## Read these files in order

| File | What to look for |
| --- | --- |
| [configs/main.yaml](../configs/main.yaml) | Paths, five accepted clips, tracking, simulation and training settings. |
| [cli.py](../phone_to_panda/cli.py) | `main()` routes each command to the relevant module. |
| [video.py](../phone_to_panda/video.py) | `extract_video()` selects the tracker; `interpolate_short_gaps()` handles brief missing measurements. |
| [multimarker.py](../phone_to_panda/multimarker.py) | `fit_reference_board()` and `extract_video_multimarker()` estimate object motion relative to the table. |
| [simulation.py](../phone_to_panda/simulation.py) | `retarget_carried_path()`, `_Recorder`, `record_demo()` and `PandaBoxEnv.step()`. |
| [learning.py](../phone_to_panda/learning.py) | Validation, export, training command construction and `run_policy()`. |
| [test_simulation.py](../tests/test_simulation.py) | Small examples of timing, contact and wrist-controller behaviour. |

`phone_to_panda/` contains the reusable pipeline. `scripts/` coordinates batches,
experiments and reports; it is not another learned model. For example,
`scripts/build_dataset.py` calls the tracker, replay and exporter, while
`scripts/evaluate_candidate.py` applies the study's fixed evaluation settings.
`scripts/summarize_study.py` reads saved reports without running a policy.
`native_demo.py` is the viewer; start with `learning.run_policy()` to understand inference.

## Follow one recording

The accepted clips are `IMG_2092`, `IMG_2093`, `IMG_2095`, `IMG_2096` and `IMG_2098`.
For `IMG_2092`, inspect these three committed files:

- [The recording](../data/raw/IMG_2092.MOV).
- [The annotations](../data/annotations/IMG_2092.json): grasp at 3.6 s and release at 8.35 s. These are estimated event times, with uncertainty recorded.
- [The measured path](../data/processed/IMG_2092.json): timestamped XYZ values in metres, in the table-marker coordinate frame, plus quality checks and source hashes.

The tracker uses calibrated camera geometry and measured marker sizes: 50 mm
table references and a 30 mm object marker. With `tracking_mode: multi_marker`,
`video.extract_video()` delegates to `extract_video_multimarker()`. Reference
regions distinguish fixed table markers from the moving object marker.

At least 90% of sampled positions must be measured. Short internal gaps can be
interpolated, but hidden endpoints are rejected. A separate candidate clip,
`IMG_2097`, failed the unchanged 90% threshold and is outside the accepted five.
Marker motion is object motion; this pipeline does not reconstruct a human skeleton.

`retarget_carried_path()` takes the section between grasp and release. It rotates
and scales horizontal motion to connect the simulated cube and box, preserves
path curvature and timing, and constrains height to the robot's clearance range.
These deliberate modifications make it a retargeted demonstration.

`record_demo()` then performs approach, descend, grasp, lift, measured carry,
lower, release and retreat. In the configured contact mode, the grasp advances
when both finger pads carry at least 1 N; release requires no cube contact and
enough finger opening. These checks belong to the demonstration controller.
They do not secretly complete a failed learned-policy grasp.

## What one training example contains

`_Recorder.tick()` saves the observation **before** applying its matching action.
An episode NPZ contains aligned `state`, `image`, `image2` and `action` arrays.
Its JSON sidecar records success, source recording/hash, seed and replay phases.
`learning._validate_episode()` checks these before `export_dataset()` writes
LeRobot data. Failed replays and synthetic installation demos are excluded.

| Field | Shape per time step | Meaning and units |
| --- | --- | --- |
| `observation.state[0:3]` | 3 | Robot end-effector XYZ in the simulator world frame, metres. |
| `observation.state[3:6]` | 3 | End-effector axis-angle rotation vector, radians; not Euler angles. |
| `observation.state[6:8]` | 2 | Two signed finger joint positions, metres. |
| `observation.images.image` | 256 × 256 × 3 | Front-view RGB, stored as uint8. |
| `observation.images.image2` | 256 × 256 × 3 | Wrist-view RGB, stored as uint8. |
| `action[0:3]` | 3 | Normalized world-frame XYZ controller commands in [-1, 1]. |
| `action[3:6]` | 3 | Normalized rotation-command slots; zero in the current replay labels and replaced by wrist feedback during execution. |
| `action[6]` | 1 | Gripper command: -1 opens, +1 closes. |
| `task` | text | “Pick up the cube and place it inside the box.” |

That is an **8-value robot state and a 7-value action**, plus two images and text.
The cube's true XYZ and contact forces are available to the replay controller
and diagnostics, but are not fields in the learned policy's observation batch.

Robosuite's OSC_POSE controller converts a translation command of 1 into a
0.05 m target displacement. This is a target increment, not a guarantee that
the robot moves 5 cm in one step. Its rotation scale is 0.5 rad, but the current
`fixed_downward` mode replaces those three action slots before physics runs.
The controller turns these task-space targets into joint torques.

For a concrete saved example, the local `IMG_2092_seed_0.npz` has 212 rows.
Its first replay action is approximately `[0.65, -0.65, -0.65, 0, 0, 0, -1]`:
move toward the approach target while keeping the fingers open. Generated NPZs
and training datasets are local artifacts, not included in a normal Git clone.

## What is learned, and what happens at inference

`learning.training_command()` constructs the upstream LeRobot training command.
It starts from `lerobot/smolvla_base`, freezes the vision encoder, enables action
expert training and trains the state projection. This is fine-tuning, not a
new model trained from scratch. `learning.py` integrates the upstream model;
it does not implement the SmolVLA architecture or optimizer loop itself.

At inference, follow `run_policy()` → `_load_policy()` → `_observation_batch()`:

1. Load the saved weights, input/output configuration and normalization processors.
2. Convert each RGB image from HWC uint8 to CHW float in [0, 1]; attach state and task text.
3. Preprocess, call `policy.select_action()`, then postprocess the returned action.
4. Pass the seven values to `PandaBoxEnv.step()`, which clips them and applies wrist feedback.
5. Step physics, obtain new observations and record success/contact/lift diagnostics.

The saved setup predicts a chunk of 20 actions and executes five before asking
for another prediction. At 20 control steps per simulated second, five actions
span 0.25 s. This describes the simulated control schedule, not measured real-time speed.

For example, the original policy's first command on final-test seed 6100 was
approximately `[0.543, -0.568, -0.733, 0, 0, 0, -0.935]`. The applied translation
and gripper values stayed the same; wrist feedback replaced the rotation slots.
The complete trace is in the [evidence archive](../results/improvement_20261011/README.md).

Success requires a previous lift above cube-centre Z = 0.97 m, followed by a
released, slow-moving cube inside the box. `PandaBoxEnv.success()` and the inner
`PickPlaceDomain._check_success()` implement those checks. A convincing-looking
video, low training loss or a completed training job is not the success test.

## Configuration controls worth knowing

| Setting | Current main value | What it changes |
| --- | --- | --- |
| `video.sample_hz`, `video.max_gap_s` | 30, 0.2 | Tracking sample rate and largest internally bounded gap allowed. |
| `simulation.control_hz`, `horizon` | 20, 600 | Control frequency and episode limit: up to 30 simulated seconds. |
| `simulation.randomize_start_m` | 0.035 | Independent uniform X/Y offsets around the cube's nominal start. |
| `simulation.orientation_control` | `fixed_downward` | Must agree with the dataset/checkpoint controller contract. |
| `simulation.video_speed` | implicit 1.0 | Carry-time divisor; 2.0 makes the retargeted carry twice as fast. |
| `training.steps`, `batch_size` | 5000, 4 | Optimizer updates and examples per batch. |
| `training.chunk_size`, `n_action_steps` | 20, 5 | Predicted chunk length and actions executed between predictions. |
| `training.policy_path`, `output_dir` | base model, `runs/train` | Starting weights and destination. |
| `evaluation.save_trace`, `save_video` | true, false | Diagnostic JSON and optional episode video. |

`training.optimizer_lr` is optional; the continuation study sets it explicitly.
Its `preserve_normalization: true` selects `scripts/train_with_frozen_stats.py`,
which retains the source checkpoint's feature transforms during further training.
Changing controller mode, image dimensions or control frequency is not a cosmetic
edit: the dataset/checkpoint interface must still match. An evaluation-only
`evaluation.n_action_steps` override changes the action queue, not saved weights.

## Three small edits to practise

Run these from the repository root, with the existing `.venv` installed.
They use configuration copies, array calculations and unit tests. They do not
train a model, start a simulator or alter the evaluated source/configuration.
Use the same terminal for the commands so `practice_dir` remains available.

### 1. Change the training budget and inspect the command

Trace how a YAML setting becomes a LeRobot argument. Change 100 to 200 below,
rerun the dry run, and find `--steps`, `--batch_size` and scheduler settings.

```bash
practice_dir="$(mktemp -d /tmp/phonetopanda-practice.XXXXXX)"
.venv/bin/python - "$practice_dir/config.yaml" <<'PY'
from pathlib import Path
import sys, yaml
config = yaml.safe_load(Path('configs/main.yaml').read_text())
config['training'].update(steps=100, batch_size=2, save_freq=100,
    output_dir=str(Path(sys.argv[1]).parent / 'training'))
Path(sys.argv[1]).write_text(yaml.safe_dump(config))
PY
bash run.sh --config "$practice_dir/config.yaml" train --dry-run
```

Expected: JSON includes `training_started: false`. `--config` goes before `train`.
The dry run checks command construction; it does not validate available data or VRAM.

### 2. Change carry speed without running the robot

Edit `video_speed` below. The same path points should take half as long at 2.0;
the annotations and source recording remain untouched.

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
import numpy as np
from phone_to_panda.simulation import retarget_carried_path
track = json.loads(Path('data/processed/IMG_2092.json').read_text())
start, end = [0, -.17, 1.03], [0, .18, 1.03]
times, path = retarget_carried_path(track, start, end, {})
video_speed = 2.0
faster, same_path = retarget_carried_path(track, start, end,
                                         {'video_speed': video_speed})
np.testing.assert_allclose(same_path, path)
np.testing.assert_allclose(faster, times / video_speed)
print('Carry seconds:', times[-1], '→', faster[-1])
PY
```

Expected: approximately 4.75 → 2.375 s. This verifies the mapping's mathematics;
it does not establish that a faster physical replay would succeed.

### 3. Add a controller boundary test

This creates a scratch test of `_Recorder.action_to()`. Edit the offsets and
calculate the expected command: `clip(offset / 0.05 × 0.65, -0.65, 0.65)`.
It tests demonstration labels, not predictions from the learned policy.

```bash
cat > "$practice_dir/test_action_scale.py" <<'PY'
from types import SimpleNamespace
import numpy as np
import pytest
from phone_to_panda.simulation import _Recorder

@pytest.mark.parametrize('offset, expected', [(0.01, 0.13), (0.10, 0.65)])
def test_translation_scale_and_limit(offset, expected):
    recorder = _Recorder.__new__(_Recorder)
    recorder.env = SimpleNamespace(orientation_control='fixed_downward')
    recorder.obs = {'observation.state': np.array([0, 0, 1, 0, 0, 0, .04, -.04])}
    action = recorder.action_to([offset, 0, 1], -1)
    np.testing.assert_allclose(action, [expected, 0, 0, 0, 0, 0, -1])
PY
.venv/bin/python -m pytest -q "$practice_dir/test_action_scale.py"
.venv/bin/python -m pytest -q tests/test_simulation.py -k orientation_hold
```

Expected: two scratch cases pass; the existing two-mode test confirms that wrist
feedback changes rotation only. Try adding `(-0.10, -0.65)` as a third case.

## What the saved evidence supports

The original dataset has 20 robot replays from five recordings: 4,471 time steps,
not 4,471 independent human demonstrations. Extra replays reuse those recordings.
The [11 October study](improvement_20261011.md) tested five frozen checkpoints
on the same 60 new starts each. Original: 43/60; continuation and uniform: 44/60;
targeted: 38/60; longer uniform: 42/60. The original was selected before those
tests and remains the default. One extra success does not establish a reliable gain.

Model weights are local and excluded from Git; see [checkpoints/README.md](../checkpoints/README.md).
A clone can inspect code, source recordings and evidence, but needs the saved
checkpoint or a new training run to execute a learned demo. New training is a
new experiment, not an exact reconstruction of the reported result.
For future experiments, use `evaluate.sh` with a fresh `--output` directory and
choose a new `--seed` range that does not overlap training or earlier evaluations.
The advanced study tools deliberately reject changed source under the old final
lock—even comments change a file hash. Historical reproduction uses the
[exact evaluated source archive](../results/improvement_20261011/evaluated_source.tar.gz)
and the study's saved evidence; keep that separate from new development.
The evidence covers one simulated scene and one training seed. It does not show
real-robot transfer, unseen-object generalization or an implemented IMU pipeline.
