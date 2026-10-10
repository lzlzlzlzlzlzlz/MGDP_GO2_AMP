import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "legged_gym"))
HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "PyTorch is tested on the training host")
class TestAmpCheckpoint(unittest.TestCase):
    @staticmethod
    def _config(**overrides):
        config = {
            "amp_reward_coef": 0.01,
            "amp_batch_size": 512,
            "amp_replay_rollouts": 2,
            "num_steps_per_env": 24,
            "amp_updates_per_iter": 1,
            "amp_gradient_penalty_coef": 10.0,
            "amp_stratified_replay": True,
        }
        config.update(overrides)
        return config

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
        config = self._config(amp_batch_size=16)
        session = AMPSession(Dataset(), 1, "cpu", config)
        is_anchor = torch.tensor([True] * 10 + [False] * 10)
        session.record(torch.randn(20, 30), torch.randn(20, 30), is_anchor)
        before = next(session.discriminator.parameters()).detach().clone()
        metrics = session.update(iteration=100)
        after = next(session.discriminator.parameters()).detach().clone()
        self.assertFalse(torch.equal(before, after))
        self.assertTrue(torch.isfinite(torch.tensor(metrics["discriminator_loss"])))
        restored = AMPSession(Dataset(), 1, "cpu", config)
        restored.load_state_dict(session.state_dict())
        self.assertEqual(restored.iteration, 1)
        self.assertTrue(torch.equal(next(restored.discriminator.parameters()), after))
        with self.assertRaisesRegex(ValueError, "amp_batch_size"):
            AMPSession(Dataset(), 1, "cpu", dict(config, amp_batch_size=0))

    def test_replay_round_trip_validates_before_mutation(self):
        import torch
        from rl.MGDP.amp.replay import AMPReplayBuffer

        replay = AMPReplayBuffer(4, 30, "cpu")
        replay.insert(torch.arange(180, dtype=torch.float).reshape(6, 30),
                      torch.arange(180, 360, dtype=torch.float).reshape(6, 30))
        state = replay.state_dict()
        restored = AMPReplayBuffer(4, 30, "cpu")
        restored.load_state_dict(state)
        self.assertEqual(restored.cursor, replay.cursor)
        self.assertEqual(restored.size, replay.size)
        self.assertTrue(torch.equal(restored.states, replay.states))
        self.assertTrue(torch.equal(restored.next_states, replay.next_states))

        before = restored.states.clone()
        incompatible = dict(state, capacity=5)
        with self.assertRaisesRegex(ValueError, "capacity"):
            restored.load_state_dict(incompatible)
        self.assertTrue(torch.equal(restored.states, before))

    def test_stratified_sampling_is_one_to_one_and_empty_pool_skips(self):
        import torch
        from rl.MGDP.amp.session import AMPSession

        class Dataset:
            def sample(self, count, device):
                return torch.randn(count, 30, device=device), torch.randn(count, 30, device=device)

        session = AMPSession(Dataset(), 1, "cpu", self._config())
        state = torch.cat((torch.zeros(2, 30), torch.full((2, 30), 10.0)))
        mask = torch.tensor([True, True, False, False])
        session.record(state, state + 1, mask)
        policy_state, _ = session._sample_policy_batch()
        self.assertEqual(int((policy_state[:, 0] < 5).sum()), 256)
        self.assertEqual(int((policy_state[:, 0] > 5).sum()), 256)
        self.assertEqual(session.anchor_replay.size, 2)
        self.assertEqual(session.course_replay.size, 2)

        empty = AMPSession(Dataset(), 1, "cpu", self._config())
        empty.record(torch.zeros(2, 30), torch.ones(2, 30), torch.ones(2, dtype=torch.bool))
        metrics = empty.update(iteration=100)
        self.assertEqual(metrics["amp_updates"], 0)
        self.assertEqual(metrics["course_replay_empty"], 1)
        self.assertEqual(metrics["course_skipped_updates"], 1)

    def test_fixed_update_count_and_gradient_penalty_coefficient(self):
        import torch
        from rl.MGDP.amp.session import AMPSession

        class Dataset:
            def sample(self, count, device):
                return torch.randn(count, 30, device=device), torch.randn(count, 30, device=device)

        config = self._config(
            amp_batch_size=8,
            amp_updates_per_iter=2,
            amp_gradient_penalty_coef=7.0,
        )
        session = AMPSession(Dataset(), 1, "cpu", config)
        mask = torch.tensor([True, True, False, False])
        session.record(torch.randn(4, 30), torch.randn(4, 30), mask)
        coefficients = []
        original = session.discriminator.gradient_penalty

        def recording_penalty(pair, coefficient):
            coefficients.append(coefficient)
            return original(pair, coefficient)

        session.discriminator.gradient_penalty = recording_penalty
        before = next(session.discriminator.parameters()).detach().clone()
        metrics = session.update(iteration=100)
        after = next(session.discriminator.parameters()).detach().clone()
        self.assertFalse(torch.equal(before, after))
        self.assertEqual(metrics["amp_updates"], 2)
        self.assertEqual(session.iteration, 2)
        self.assertEqual(coefficients, [7.0, 7.0])

        for invalid in (0, -1):
            with self.assertRaisesRegex(ValueError, "amp_updates_per_iter"):
                AMPSession(Dataset(), 1, "cpu", self._config(amp_updates_per_iter=invalid))


if __name__ == "__main__":
    unittest.main()
