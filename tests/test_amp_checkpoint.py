import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "legged_gym"))
HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "PyTorch is tested on the training host")
class TestAmpCheckpoint(unittest.TestCase):
    def test_missing_state_and_policy_warm_start(self):
        from rl.MGDP.amp.session import restore_amp_checkpoint
        class Stub:
            loaded = False
            def load_state_dict(self, state):
                self.loaded = True
        session = Stub()
        with self.assertRaisesRegex(ValueError, "AMP checkpoint"):
            restore_amp_checkpoint(session, {}, False)
        restore_amp_checkpoint(session, {}, True)
        self.assertFalse(session.loaded)
        restore_amp_checkpoint(session, {"amp_state": {}}, False)
        self.assertTrue(session.loaded)

    def test_session_update_and_round_trip(self):
        import torch
        from rl.MGDP.amp.session import AMPSession
        class Dataset:
            def sample(self, count, device):
                return torch.randn(count, 30, device=device) + 2, torch.randn(count, 30, device=device) + 2
        config = {"amp_reward_coef": 0.01, "amp_batch_size": 16,
                  "amp_replay_capacity": 32}
        session = AMPSession(Dataset(), 1, "cpu", config)
        session.record(torch.randn(20, 30), torch.randn(20, 30))
        before = next(session.discriminator.parameters()).detach().clone()
        metrics = session.update()
        after = next(session.discriminator.parameters()).detach().clone()
        self.assertFalse(torch.equal(before, after))
        self.assertTrue(torch.isfinite(torch.tensor(metrics["discriminator_loss"])))
        restored = AMPSession(Dataset(), 1, "cpu", config)
        restored.load_state_dict(session.state_dict())
        self.assertEqual(restored.iteration, 1)
        self.assertTrue(torch.equal(next(restored.discriminator.parameters()), after))


if __name__ == "__main__":
    unittest.main()
