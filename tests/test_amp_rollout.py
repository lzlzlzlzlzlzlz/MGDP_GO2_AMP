"""Local source contract for AMP placement in the original MGDP runner."""

import ast
import unittest
from pathlib import Path


class TestAmpRolloutIntegration(unittest.TestCase):
    def test_reward_is_applied_before_ppo_transition(self):
        path = Path(__file__).resolve().parents[1] / "legged_gym/rl/MGDP/runners/policy_runner.py"
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
        before = source.index("amp_before = self.env.get_amp_observations()")
        step = source.index("obs_dict, rewards, dones, infos = self.env.step(actions)")
        reward = source.index("rewards, amp_rollout_metrics = self.amp.reward(")
        ppo = source.index("self.alg.process_env_step(rewards, dones")
        self.assertLess(before, step)
        self.assertLess(step, reward)
        self.assertLess(reward, ppo)
        self.assertIn("amp_after = select_next_amp_state(", source)

    def test_final_handoff_saves_matching_policy_and_world_model(self):
        root = Path(__file__).resolve().parents[1] / "legged_gym"
        runner = (root / "rl/MGDP/runners/policy_runner.py").read_text(encoding="utf-8")
        helper = (root / "legged_gym/utils/helpers.py").read_text(encoding="utf-8")
        environment = (root / "legged_gym/envs/random_dog/random_dog.py").read_text(encoding="utf-8")
        self.assertIn("self.save(os.path.join(self.nn_dir, 'last.pt'))", runner)
        self.assertIn("self.save_world_model(os.path.join(self.nn_dir, 'wm_last.pt'))", runner)
        self.assertIn("world_model_checkpoint = 'wm_last.pt'", helper)
        self.assertIn("wm_file = getattr(self.cfg.camera, 'world_model_checkpoint'", environment)
        self.assertIn("loaded_dict['iter'] + (1 if self.amp is not None else 0)", runner)


if __name__ == "__main__":
    unittest.main()
