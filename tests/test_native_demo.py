"""Placement UI coordinates must preserve the chosen physical cube position."""
import unittest
from unittest.mock import Mock, patch
import numpy as np

from phone_to_panda.native_demo import PlacementPanel, validate_position, in_training_area


class PlacementTests(unittest.TestCase):
    def test_invalid_and_outside_positions_rejected(self):
        for xy in ([float('nan'), 0], [.21, -.17], [0, .18], [0]):
            with self.assertRaises(ValueError):
                validate_position(xy)

    def test_canvas_mapping_round_trip(self):
        panel = PlacementPanel.__new__(PlacementPanel)
        for xy in ([0, -.17], [-.2, -.28], [.2, .04]):
            self.assertTrue(np.allclose(panel.world(*panel.pixel(xy)), xy))

    def test_extrapolation_is_explicit(self):
        self.assertTrue(in_training_area([0, -.17]))
        self.assertFalse(in_training_area([.1, -.17]))

    def test_viewer_state_is_a_separate_copy(self):
        import copy
        import mujoco
        from phone_to_panda.native_demo import mirror_simulation
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".02" mass="1"/></body></worldbody></mujoco>')
        data = mujoco.MjData(model)
        display_model = copy.copy(model)
        display_data = mujoco.MjData(display_model)
        data.qpos[:3] = [.12, -.17, .92]
        mirror_simulation(model, data, display_model, display_data)
        self.assertTrue(np.allclose(display_data.qpos, data.qpos))
        display_data.qpos[0] = 4
        display_model.opt.gravity[:] = 0
        self.assertEqual(data.qpos[0], .12)
        self.assertLess(model.opt.gravity[2], 0)

    def test_desktop_forwards_inference_override_and_records_effective_settings(self):
        """Exercise desktop setup without a window, renderer, or neural model."""
        from contextlib import nullcontext
        import json
        from pathlib import Path
        import tempfile
        from types import SimpleNamespace
        from phone_to_panda.native_demo import run_desktop
        for override in (None, 1):
            with self.subTest(override=override), tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory)
                saved = b'{"n_action_steps": 5, "chunk_size": 20}'
                (checkpoint / "config.json").write_bytes(saved)
                config = {"training": {"device": "cpu"}, "evaluation": {}}
                if override is not None:
                    config["evaluation"]["n_action_steps"] = override
                config_before = json.loads(json.dumps(config))
                policy = SimpleNamespace(config=SimpleNamespace(n_action_steps=override or 5, chunk_size=20, num_steps=10), reset=Mock())
                panel = SimpleNamespace(root=SimpleNamespace(update=Mock(), destroy=Mock()), draw=Mock(), status=SimpleNamespace(set=Mock()))
                viewer = SimpleNamespace(lock=nullcontext, cam=SimpleNamespace(lookat=np.zeros(3)),
                                         is_running=lambda: False, close=Mock())
                env = SimpleNamespace(cube_position=np.array([0., -.17, .92]), reset=lambda: {}, close=Mock(),
                                      raw_env=SimpleNamespace(sim=SimpleNamespace(model=SimpleNamespace(_model=object()),
                                                                               data=SimpleNamespace(_data=object()))))
                with patch("phone_to_panda.native_demo.validate_checkpoint"), \
                     patch("phone_to_panda.native_demo.PlacementPanel", return_value=panel), \
                     patch("phone_to_panda.native_demo._load_policy", return_value=(policy, None, None)) as load, \
                     patch("phone_to_panda.native_demo.make_env", return_value=env), \
                     patch("phone_to_panda.native_demo.mirror_simulation"), \
                     patch("mujoco.MjData", return_value=object()), \
                     patch("mujoco.viewer.launch_passive", return_value=viewer):
                    report = run_desktop(config, checkpoint, seed=1000)
                load.assert_called_once_with(checkpoint, "cpu", n_action_steps=override)
                self.assertEqual(report["policy_settings"], {
                    "checkpoint_n_action_steps": 5, "n_action_steps": override or 5,
                    "chunk_size": 20, "denoising_steps": 10, "evaluation_override": override,
                })
                self.assertEqual(report["seed"], 1000)
                self.assertFalse(report["benchmark_evaluation"])
                self.assertEqual(config, config_before)
                self.assertEqual((checkpoint / "config.json").read_bytes(), saved)
                viewer.close.assert_called_once()
                env.close.assert_called_once()
                panel.root.destroy.assert_called_once()


if __name__ == '__main__':
    unittest.main()
