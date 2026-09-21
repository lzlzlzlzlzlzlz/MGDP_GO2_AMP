import ast
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "legged_gym"))
HAS_TORCH = importlib.util.find_spec("torch") is not None


class TestAmpEnvironmentSource(unittest.TestCase):
    def test_order_validated_and_reset_captured(self):
        source = (ROOT / "legged_gym/legged_gym/envs/go2_amp/env.py").read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn("validate_go2_dofs(self.dof_names)", source)
        self.assertIn('self.extras["terminal_amp_states"]', source)


@unittest.skipUnless(HAS_TORCH, "PyTorch is tested on the training host")
class TestAmpTransition(unittest.TestCase):
    def test_done_uses_terminal_state(self):
        import torch
        from rl.MGDP.amp.transition import select_next_amp_state
        after = torch.zeros(3, 30)
        packet = (torch.tensor([1]), torch.ones(1, 30))
        result = select_next_amp_state(after, torch.tensor([0, 1, 0]), packet)
        self.assertEqual(result[1, 0].item(), 1)
        self.assertEqual(after[1, 0].item(), 0)
        with self.assertRaises(ValueError):
            select_next_amp_state(after, torch.tensor([0, 1, 0]),
                                  (torch.tensor([0]), torch.ones(1, 30)))
        self.assertTrue(torch.equal(select_next_amp_state(after, torch.zeros(3), None), after))

    def test_exact_joint_order(self):
        from rl.MGDP.amp.state import GO2_DOF_NAMES, validate_go2_dofs
        validate_go2_dofs(GO2_DOF_NAMES)
        with self.assertRaisesRegex(ValueError, "DOF order"):
            validate_go2_dofs(GO2_DOF_NAMES[3:6] + GO2_DOF_NAMES[:3] + GO2_DOF_NAMES[6:])


if __name__ == "__main__":
    unittest.main()
