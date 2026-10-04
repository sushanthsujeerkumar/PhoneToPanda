"""Prevent accidental pure-policy versus assisted-wrist comparisons."""
import unittest

from phone_to_panda.learning import controller_mode, validate_controller_contract


class ControllerContractTests(unittest.TestCase):
    def test_existing_checkpoints_keep_full_policy_controller(self):
        provenance = {"provenance": [{"simulation_settings": {}}]}
        self.assertEqual(validate_controller_contract({}, provenance), "policy")

    def test_assisted_checkpoint_requires_matching_environment(self):
        provenance = {"provenance": [{"simulation_settings": {"orientation_control": "fixed_downward"}}]}
        with self.assertRaisesRegex(ValueError, "differs"):
            validate_controller_contract({}, provenance)
        config = {"simulation": {"orientation_control": "fixed_downward"}}
        self.assertEqual(validate_controller_contract(config, provenance), "fixed_downward")

    def test_mixed_controller_data_rejected(self):
        with self.assertRaisesRegex(ValueError, "mix"):
            controller_mode({"provenance": [{}, {"simulation_settings": {"orientation_control": "fixed_downward"}}]})


if __name__ == "__main__":
    unittest.main()
