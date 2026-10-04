# Phone to Panda

This project uses five phone recordings of a person moving an object to train a
simulated Franka Panda arm to pick up a cube and place it in a box.

## Demo

https://github.com/user-attachments/assets/be2a7a18-ede6-4dbe-93df-fda033dd7acc

**2 min 32 sec · silent screen recording.** Launching the saved model, choosing cube
positions and testing pick and place in the desktop MuJoCo viewer. The full session
is shown at normal speed. These interactive attempts are separate from the
measured evaluation below. [Download the MP4](media/demo.mp4).

The method is **object-motion retargeting followed by imitation learning**.
Markers make the object's carry path measurable. That path is replayed by the
robot in a small custom LIBERO scene, and successful replays become the training
data for SmolVLA. The model learns movement and gripper commands; a separate
feedback controller keeps the wrist pointing down.

The original model achieved **15/20 placements** in a paired test of new simulator
starts. An additional training experiment achieved **14/20**, so the original
model remains the default. These are results in this one simulation task, not
official LIBERO benchmark scores or tests on a real robot.

![Original and adapted model results](results/paired_results.png)

## Run it

Open a terminal in this folder. The tested system is Ubuntu 24.04 with Python
3.12, an NVIDIA RTX 5060 with 8 GB of VRAM, and a working NVIDIA driver with EGL.
The setup creates a local `.venv`. Python needs venv support; the interactive
desktop viewer also needs Tkinter (`python3-tk` on Ubuntu) and a desktop session.
Installing dependencies and downloading the pretrained model need internet access.

```bash
bash setup.sh
bash run.sh doctor
```

To build the dataset from the five supplied recordings, train, and run a demo:

```bash
.venv/bin/python scripts/build_dataset.py
bash train.sh
bash demo.sh --checkpoint runs/train/checkpoints/last/pretrained_model
bash evaluate.sh --checkpoint runs/train/checkpoints/last/pretrained_model --episodes 20 --seed 4000
```

The dataset script prepares the recordings, generates 20 physics replays and
exports the successful episodes for LeRobot. Training uses 5,000 updates with
batch size four. Settings are in [configs/main.yaml](configs/main.yaml).

Without `--checkpoint`, the demo and evaluation first look in `checkpoints/main`,
then in `runs/train/checkpoints/last/pretrained_model`. The commands above select
the newly trained model explicitly.

For the saved local model, run `bash demo.sh`. In the placement window, choose a
cube start and click **Run / Pause**. You can rotate and zoom in the MuJoCo window. Use
**Reset** for another attempt. Positions outside the marked training area are
harder and may fail. The viewer does not save a video.

The saved checkpoint is about 1.2 GB and is excluded from Git. There is no public
checkpoint download yet. A fresh copy without that checkpoint needs the dataset
build and training steps before the learned demo can run. No new phone recordings
are needed for those steps.

This copy was checked in a fresh Python environment on the same Ubuntu machine:
44 tests passed, the 20-episode dataset was rebuilt, and the saved model completed
a native demo. The full 5,000-update training was not repeated during cleanup.
See [development notes](docs/development_notes.md) for the checks and limits.

## What is in the project?

| Path | Contents |
| --- | --- |
| `data/raw/` | Five accepted phone recordings, with private metadata and audio removed |
| `data/calibration/camera.json` | Camera calibration used for tracking |
| `data/annotations/` | Visually labelled grasp and release times |
| `data/processed/` | Measured paths and tracking reports |
| `phone_to_panda/` | Tracking, simulation, dataset and policy code |
| `configs/main.yaml` | Original 20-demonstration experiment |
| `configs/expanded.yaml` | Additional 40-replay experiment |
| `results/` | Saved evaluation reports and figures |

The five source recordings are `IMG_2092`, `IMG_2093`, `IMG_2095`, `IMG_2096` and
`IMG_2098`. Replaying them at several object starts adds robot demonstrations,
but the dataset still contains only five independently recorded human trials.

The most useful details are in:

- [Recording and calibration](docs/recording.md): the physical setup and accepted clips.
- [Method](docs/method.md): what comes from the videos and what the controllers do.
- [Results](docs/results.md): training settings, tests and failure cases.
- [Development notes](docs/development_notes.md): dated changes and earlier attempts.

## Built with

[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) supplies the simulation
framework, [LeRobot](https://github.com/huggingface/lerobot) handles the dataset and
training, and [SmolVLA](https://huggingface.co/lerobot/smolvla_base) is the pretrained
policy being fine-tuned. OpenCV tracks the printed markers; MuJoCo simulates
contacts between the robot, cube and box.

Project code uses the [MIT license](LICENSE). Bundled LIBERO assets retain their
[upstream license and provenance](assets/libero/PROVENANCE.md). Pretrained model
weights have their own license.
