"""Interactive desktop MuJoCo exploration with the saved SmolVLA policy."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import queue
import time

import numpy as np

from .common import load_config, project_path
from .learning import _load_policy, _observation_batch, validate_checkpoint
from .simulation import make_env

# Choose starts in front of the robot and away from the fixed target box.
X_LIMITS = (-0.20, 0.20)
Y_LIMITS = (-0.28, 0.04)
TRAIN_CENTER = (0.0, -0.17)
TRAIN_RADIUS = 0.035


def validate_position(xy):
    xy = np.asarray(xy, dtype=float)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError("Choose two finite cube coordinates.")
    if not (X_LIMITS[0] <= xy[0] <= X_LIMITS[1] and Y_LIMITS[0] <= xy[1] <= Y_LIMITS[1]):
        raise ValueError("Choose a cube position inside the blue placement area.")
    return xy.copy()


def in_training_area(xy):
    return bool(np.all(np.abs(np.asarray(xy) - TRAIN_CENTER) <= TRAIN_RADIUS + 1e-9))


def mirror_simulation(source_model, source_data, display_model, display_data):
    """Mirror dynamic state using APIs exposed by MuJoCo 3.3 Python bindings."""
    import mujoco
    specification = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(source_model, specification))
    mujoco.mj_getState(source_model, source_data, state, specification)
    mujoco.mj_setState(display_model, display_data, state, specification)
    mujoco.mj_forward(display_model, display_data)


class PlacementPanel:
    """A small click-to-place map; simulation mutations stay on the main thread."""
    SIZE, PAD = 380, 35
    MAP_X, MAP_Y = (-0.25, 0.25), (-0.31, 0.29)

    def __init__(self, commands):
        import tkinter as tk
        from tkinter import ttk
        self.root = tk.Tk()
        self.root.title("Phone to Panda — choose a cube position")
        self.root.geometry("490x720+30+50")
        self.root.protocol("WM_DELETE_WINDOW", lambda: commands.put(("quit", None)))
        self.status = tk.StringVar(value="Loading trained model…")
        self.position = tk.StringVar(value="")
        self.area = tk.StringVar(value="")
        self.preview = None
        ttk.Label(self.root, text="Choose a start, then let the robot try", font=("Sans", 14, "bold")).pack(pady=(12, 6))
        ttk.Label(self.root, textvariable=self.status, wraplength=460).pack(pady=4)
        self.canvas = tk.Canvas(self.root, width=450, height=400, background="#f3f6fa", highlightthickness=0)
        self.canvas.pack(padx=12, pady=6)
        self.canvas.bind("<Button-1>", lambda event: self.select(event, commands, preview=True))
        self.canvas.bind("<B1-Motion>", lambda event: self.select(event, commands, preview=True))
        self.canvas.bind("<ButtonRelease-1>", lambda event: self.select(event, commands))
        ttk.Label(self.root, textvariable=self.position).pack()
        ttk.Label(self.root, textvariable=self.area, wraplength=460).pack(pady=(2, 8))
        row = ttk.Frame(self.root)
        row.pack()
        for label, command in [("Run / Pause", "toggle"), ("Reset", "reset"), ("Random trained start", "random")]:
            ttk.Button(row, text=label, command=lambda c=command: commands.put((c, None))).pack(side="left", padx=3)
        ttk.Label(self.root, text="Click the light-blue area to move the cube. Green = training area.\nIn the 3D window: mouse to orbit, wheel to zoom.\nSpace: run/pause · R: reset · arrows: move cube before running", justify="center").pack(pady=10)
        ttk.Label(self.root, text="SmolVLA movement + gripper; fixed downward wrist feedback.\nInteractive exploration; no video is recorded.", justify="center").pack()
        self.root.update()

    def pixel(self, xy):
        return (self.PAD + (xy[0] - self.MAP_X[0]) / (self.MAP_X[1] - self.MAP_X[0]) * self.SIZE,
                self.PAD + (self.MAP_Y[1] - xy[1]) / (self.MAP_Y[1] - self.MAP_Y[0]) * (self.SIZE - 30))

    def world(self, x, y):
        return np.array([self.MAP_X[0] + (x - self.PAD) / self.SIZE * (self.MAP_X[1] - self.MAP_X[0]),
                         self.MAP_Y[1] - (y - self.PAD) / (self.SIZE - 30) * (self.MAP_Y[1] - self.MAP_Y[0])])

    def select(self, event, commands, preview=False):
        try:
            xy = validate_position(self.world(event.x, event.y))
        except ValueError:
            return
        # Dragging previews in the map; release applies one reset.
        if preview:
            self.draw(xy)
        else:
            commands.put(("place", xy))

    def draw(self, xy):
        c = self.canvas
        c.delete("all")
        a = self.pixel((X_LIMITS[0], Y_LIMITS[1]))
        b = self.pixel((X_LIMITS[1], Y_LIMITS[0]))
        c.create_rectangle(*a, *b, fill="#deebff", outline="#6087b6", width=2)
        a = self.pixel(np.asarray(TRAIN_CENTER) + [-TRAIN_RADIUS, TRAIN_RADIUS])
        b = self.pixel(np.asarray(TRAIN_CENTER) + [TRAIN_RADIUS, -TRAIN_RADIUS])
        c.create_rectangle(*a, *b, fill="#ccecd9", outline="#24764d", width=2)
        a, b = self.pixel((-.07, .25)), self.pixel((.07, .11))
        c.create_rectangle(*a, *b, fill="#518fc9", outline="#204a70", width=3)
        c.create_text(*self.pixel((0, .18)), text="Target box", fill="white")
        x, y = self.pixel(xy)
        c.create_rectangle(x-9, y-9, x+9, y+9, fill="#e95432", outline="#8f250f", width=2)
        c.create_text(225, 18, text="Table viewed from above · selected cube position")
        self.position.set(f"Cube start: x {xy[0]*100:+.1f} cm, y {xy[1]*100:+.1f} cm")
        self.area.set("Inside the training area" if in_training_area(xy) else "Outside the training area — exploratory test; the policy may fail")


def run_desktop(config, checkpoint, seed=1000, auto_start=False, exit_when_done=False, max_steps=None, smoke_controls=False):
    import mujoco
    import mujoco.viewer
    import torch
    import tkinter as tk

    validate_checkpoint(config, checkpoint, seed)
    commands = queue.SimpleQueue()
    panel = PlacementPanel(commands)
    env = viewer = None
    report = {"kind": "interactive_native_policy", "benchmark_evaluation": False, "video_recorded": False}
    try:
        panel.root.update()
        policy, pre, post = _load_policy(checkpoint, config.get("training", {}).get("device", "cuda"))
        env = make_env(copy.deepcopy(config), seed=seed)
        obs = env.reset()
        selected = env.cube_position[:2].copy()
        # The desktop renderer receives a copy. Its mouse perturbations and GUI
        # physics controls cannot alter the policy's actual simulation state.
        display_model = copy.copy(env.raw_env.sim.model._model)
        display_data = mujoco.MjData(display_model)
        mirror_simulation(env.raw_env.sim.model._model, env.raw_env.sim.data._data, display_model, display_data)
        keys = {32: "toggle", 257: "toggle", 82: "reset", 78: "random", 256: "quit",
                263: "left", 262: "right", 265: "up", 264: "down"}
        viewer = mujoco.viewer.launch_passive(display_model, display_data,
                    key_callback=lambda key: commands.put((keys[key], None)) if key in keys else None,
                    show_left_ui=False, show_right_ui=False)
        with viewer.lock():
            viewer.cam.lookat[:] = [0., 0., .95]
            viewer.cam.distance = 1.65
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -35
        mode, success, steps = "ready", False, 0
        random = np.random.default_rng(seed + 9000)
        policy.reset()
        torch.manual_seed(seed)
        panel.draw(selected)
        panel.status.set("Ready. Choose a position, then press Run.")

        def reset_at(xy):
            nonlocal obs, selected, mode, success, steps
            selected = validate_position(xy)
            env.settings["cube_start_xy"] = selected.tolist()
            env.settings["randomize_start_m"] = 0.
            obs = env.reset()
            policy.reset()
            torch.manual_seed(seed)
            mode, success, steps = "ready", False, 0
            panel.draw(selected)
            panel.status.set("Ready. Press Run to test this position.")

        if smoke_controls:
            original = selected.copy()
            reset_at([.02, -.18])
            assert np.allclose(env.cube_position[:2], [.02, -.18], atol=1e-5)
            reset_at(original)
            assert env.steps == 0
            report["position_reset_smoke_passed"] = True
        if auto_start:
            commands.put(("toggle", None))
        while viewer.is_running():
            started = time.perf_counter()
            panel.root.update()
            quit_requested = False
            while not commands.empty():
                command, value = commands.get()
                if command == "quit":
                    quit_requested = True
                    break
                if command == "toggle":
                    if mode == "finished":
                        reset_at(selected)
                    mode = "paused" if mode == "running" else "running"
                    panel.status.set("Paused. Press Run to continue, or Reset to start again." if mode == "paused" else "Running the trained policy…")
                elif command == "reset":
                    reset_at(selected)
                elif command in ("place", "random", "left", "right", "up", "down"):
                    if mode == "running":
                        panel.status.set("Pause or Reset before changing the cube position.")
                        panel.draw(selected)
                        continue
                    if command == "random":
                        value = np.asarray(TRAIN_CENTER) + random.uniform(-TRAIN_RADIUS, TRAIN_RADIUS, 2)
                    elif command != "place":
                        offset = {"left": [-.01, 0], "right": [.01, 0], "up": [0, .01], "down": [0, -.01]}[command]
                        value = np.clip(selected + offset, [X_LIMITS[0], Y_LIMITS[0]], [X_LIMITS[1], Y_LIMITS[1]])
                    reset_at(value)
            if quit_requested:
                break
            if mode == "running":
                with torch.inference_mode():
                    action = post(policy.select_action(pre(_observation_batch(obs, config["task"]))))
                action = action.detach().cpu().numpy().reshape(-1)
                if action.shape != (7,) or not np.isfinite(action).all():
                    raise ValueError("The trained policy produced an invalid action.")
                obs, _, done, _ = env.step(np.clip(action, -1., 1.))
                steps += 1
                success = bool(env.success())
                limited = max_steps is not None and steps >= max_steps and not (success or done)
                if success or done or limited:
                    mode = "finished"
                    result = "Success: cube placed inside the box." if success else "Preview step limit reached." if limited else "Trial finished: the robot did not complete the task."
                    panel.status.set(result + " Choose another position or press Run to repeat.")
                    report.update(success=None if limited else success, steps=steps, start_xy=selected.tolist(),
                                  inside_training_area=in_training_area(selected),
                                  orientation_control=env.orientation_control, step_limited=limited)
                    print(json.dumps(report), flush=True)
            # Synchronizing a separate display copy prevents manual GUI forces
            # from contaminating this policy rollout. Camera orbit still works.
            with viewer.lock():
                mirror_simulation(env.raw_env.sim.model._model, env.raw_env.sim.data._data, display_model, display_data)
                viewer.perturb.active = 0
                viewer.perturb.active2 = 0
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                             "Phone to Panda — trained policy", f"{mode.upper()} | {steps} steps\nSPACE run/pause · R reset · arrows move start\nUse the placement map to choose a cube position"))
            viewer.sync()
            if exit_when_done and mode == "finished":
                break
            time.sleep(max(0., 1 / env.control_hz - (time.perf_counter() - started)))
        return report
    finally:
        if viewer is not None:
            viewer.close()
        if env is not None:
            env.close()
        try:
            panel.root.destroy()
        except tk.TclError:
            pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--checkpoint", help="Policy directory; default checkpoints/main, then the latest training checkpoint")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument("--exit-when-done", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--smoke-controls", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--report")
    args = parser.parse_args(argv)
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    config = load_config(args.config)
    try:
        from .cli import latest_checkpoint
        checkpoint = project_path(config, args.checkpoint) if args.checkpoint else Path(latest_checkpoint(config))
        result = run_desktop(config, checkpoint, seed=args.seed,
                             auto_start=args.auto_start, exit_when_done=args.exit_when_done,
                             max_steps=args.max_steps, smoke_controls=args.smoke_controls)
        if args.report:
            destination = project_path(config, args.report)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(result, indent=2))
        return 0
    except KeyboardInterrupt:
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Native demo: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
