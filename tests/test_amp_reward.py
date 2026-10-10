import ast
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "legged_gym"))
HAS_TORCH = importlib.util.find_spec("torch") is not None
ROOT = Path(__file__).resolve().parents[1]


def _load_go2_reward_class(torch):
    path = ROOT / "legged_gym/legged_gym/envs/go2_amp/env.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    original = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Go2AmpRandomDog"
    )
    reward = next(
        node for node in original.body
        if isinstance(node, ast.FunctionDef) and node.name == "_reward_feet_air_time"
    )

    class Randomdog:
        def _reward_feet_air_time(self):
            self.parent_reward_called = True
            return "legacy-stage2"

    isolated = ast.ClassDef(
        name="Go2AmpRandomDog",
        bases=[ast.Name(id="Randomdog", ctx=ast.Load())],
        keywords=[],
        body=[reward],
        decorator_list=[],
    )
    module = ast.Module(body=[isolated], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"Randomdog": Randomdog, "torch": torch}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["Go2AmpRandomDog"]


class TestAmpRewardSource(unittest.TestCase):
    def test_stage1_reward_has_cap_marker_and_legacy_fallback(self):
        path = ROOT / "legged_gym/legged_gym/envs/go2_amp/env.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        class_node = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Go2AmpRandomDog"
        )
        reward = next(
            (node for node in class_node.body
             if isinstance(node, ast.FunctionDef) and node.name == "_reward_feet_air_time"),
            None,
        )
        self.assertIsNotNone(reward)
        reward_source = ast.unparse(reward)
        self.assertIn("amp_air_time_cap", reward_source)
        self.assertIn("super()._reward_feet_air_time()", reward_source)
        self.assertIn("torch.clamp(self.feet_air_time, max=air_time_cap)", reward_source)


@unittest.skipUnless(HAS_TORCH, "PyTorch is tested on the training host")
class TestAmpReward(unittest.TestCase):
    def test_stage1_air_time_is_capped_and_stage2_delegates(self):
        import torch

        reward_class = _load_go2_reward_class(torch)
        legacy = reward_class()
        legacy.cfg = SimpleNamespace(rewards=SimpleNamespace())
        self.assertEqual(legacy._reward_feet_air_time(), "legacy-stage2")
        self.assertTrue(legacy.parent_reward_called)

        stage1 = reward_class()
        stage1.cfg = SimpleNamespace(rewards=SimpleNamespace(amp_air_time_cap=0.75))
        stage1.contact_forces = torch.zeros(1, 4, 3)
        stage1.contact_forces[:, :, 2] = 2.0
        stage1.feet_indices = [0, 1, 2, 3]
        stage1.last_contacts = torch.zeros(1, 4, dtype=torch.bool)
        stage1.feet_air_time = torch.tensor([[0.40, 0.74, 0.80, 0.0]])
        stage1.commands = torch.tensor([[0.7, 0.0, 0.0, 0.0]])
        stage1.dt = 0.02
        self.assertTrue(torch.allclose(stage1._reward_feet_air_time(), torch.tensor([0.42])))

        randomdog_source = (
            ROOT / "legged_gym/legged_gym/envs/random_dog/random_dog.py"
        ).read_text(encoding="utf-8")
        self.assertIn("(self.feet_air_time - 0.5) * first_contact", randomdog_source)

    def test_gate_zero_and_bounded_style(self):
        import torch
        from rl.MGDP.amp.session import combine_reward
        task = torch.tensor([0.02, 0.02, 0.02])
        logits = torch.tensor([1.0, 1.0, 5.0])
        classes = torch.tensor([2, 6, 2])
        actual = combine_reward(task, logits, classes, stage=1, coef=0.01)
        self.assertTrue(torch.allclose(actual, torch.tensor([0.03, 0.0225, 0.02])))
        self.assertTrue(torch.equal(combine_reward(task, logits, classes, 1, 0.0), task))
        stage2 = combine_reward(task, logits, classes, stage=2, coef=0.01)
        self.assertTrue(torch.allclose(stage2[:2], torch.tensor([0.03, 0.0225])))

    def test_replay_and_discriminator(self):
        import torch
        from rl.MGDP.amp.replay import AMPReplayBuffer
        from rl.MGDP.amp.discriminator import AMPDiscriminator
        buffer = AMPReplayBuffer(4, 30, "cpu")
        with self.assertRaises(ValueError):
            buffer.sample(1)
        buffer.insert(torch.arange(180, dtype=torch.float).reshape(6, 30),
                      torch.ones(6, 30))
        self.assertEqual(buffer.size, 4)
        a, b = buffer.sample(8)
        self.assertEqual(a.shape, (8, 30))
        self.assertTrue(torch.isfinite(b).all())
        discriminator = AMPDiscriminator()
        pair = torch.randn(8, 60)
        penalty = discriminator.gradient_penalty(pair)
        self.assertTrue(torch.isfinite(penalty))


if __name__ == "__main__":
    unittest.main()
