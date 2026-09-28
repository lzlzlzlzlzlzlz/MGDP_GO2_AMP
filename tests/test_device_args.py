"""Unit tests for selecting the Isaac Gym simulation device."""

import importlib.util
from pathlib import Path
import unittest


DEVICE_MODULE = (
    Path(__file__).resolve().parents[1]
    / "legged_gym/legged_gym/utils/device.py"
)


def load_device_module():
    spec = importlib.util.spec_from_file_location("device_args", DEVICE_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSimulationDeviceSelection(unittest.TestCase):
    def test_compute_device_is_used_when_render_device_is_omitted(self):
        device = load_device_module()
        self.assertEqual(device.resolve_sim_device_id(None, 0), 0)

    def test_explicit_render_device_takes_precedence(self):
        device = load_device_module()
        self.assertEqual(device.resolve_sim_device_id(1, 0), 1)


if __name__ == "__main__":
    unittest.main()
