import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "legged_gym"))
HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "PyTorch is tested on the training host")
class TestAmpReward(unittest.TestCase):
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
