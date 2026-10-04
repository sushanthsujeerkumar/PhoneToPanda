"""Placement UI coordinates must preserve the chosen physical cube position."""
import unittest
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


if __name__ == '__main__':
    unittest.main()
