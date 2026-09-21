"""Source-level contract checks while Isaac Gym is unavailable locally."""

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1] / "legged_gym"


class TestGo2TaskConfig(unittest.TestCase):
    def test_new_tasks_are_registered(self):
        source = (ROOT / "legged_gym/envs/__init__.py").read_text(encoding="utf-8")
        self.assertIn('task_registry.register("go2_amp_stage1"', source)
        self.assertIn('task_registry.register("go2_amp_stage2"', source)

    def test_original_tasks_remain_registered(self):
        source = (ROOT / "legged_gym/envs/__init__.py").read_text(encoding="utf-8")
        self.assertIn('task_registry.register("random_dog_stage1"', source)
        self.assertIn('task_registry.register("random_dog_stage2"', source)

    def test_new_config_limits_robot_and_disables_style_scales(self):
        config = (ROOT / "legged_gym/envs/go2_amp/config.py").read_text(encoding="utf-8")
        ast.parse(config)
        self.assertEqual(config.count('asset_name = ["go2"]'), 2)
        self.assertEqual(config.count("motion_trot = 0.0"), 2)
        self.assertEqual(config.count("feet_air_time = 0.0"), 2)
        self.assertIn("amp_stage = 1", config)
        self.assertIn("amp_stage = 2", config)


if __name__ == "__main__":
    unittest.main()
