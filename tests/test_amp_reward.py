import ast
import importlib.util
import math
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


def _load_schedule_helpers():
    path = ROOT / "legged_gym/rl/MGDP/amp/session.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        "effective_amp_coefficient",
        "discriminator_updates_enabled",
        "curriculum_is_unlocked",
    }
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in functions} != names:
        raise AssertionError("AMP schedule helpers are missing")
    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"math": math}
    exec(compile(module, str(path), "exec"), namespace)
    return tuple(namespace[name] for name in sorted(names))


class TestAmpRewardSource(unittest.TestCase):
    def test_stage2_session_keeps_legacy_reward_and_checkpoint_contract(self):
        path = ROOT / "legged_gym/rl/MGDP/amp/session.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        session = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "AMPSession"
        )
        methods = {
            node.name: ast.unparse(node)
            for node in session.body
            if isinstance(node, ast.FunctionDef)
        }
        reward_source = methods["reward"]
        update_source = methods["update"]
        checkpoint_source = methods["state_dict"]
        self.assertIn("if not self.stratified", reward_source)
        self.assertIn("'style_reward'", reward_source)
        self.assertIn("if not self.stratified", update_source)
        self.assertIn("'expert_logit'", update_source)
        self.assertIn("'policy_logit'", update_source)
        self.assertNotIn("state['replay']", checkpoint_source)

        state_dict = next(
            node for node in session.body
            if isinstance(node, ast.FunctionDef) and node.name == "state_dict"
        )
        state_assignment = next(
            node for node in state_dict.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "state"
                    for target in node.targets)
        )
        self.assertEqual(
            {ast.literal_eval(key) for key in state_assignment.value.keys},
            {"discriminator", "optimizer", "normalizer", "iteration"},
        )
        stratified_branch = next(
            node for node in state_dict.body
            if isinstance(node, ast.If) and ast.unparse(node.test) == "self.stratified"
        )
        self.assertIn("policy_iteration", ast.unparse(stratified_branch))

    def test_exact_schedule_boundaries(self):
        unlocked, updates_enabled, coefficient = _load_schedule_helpers()
        expected = {
            99: 0.0,
            100: 0.0,
            101: 0.01 / 399,
            200: 0.01 * 100 / 399,
            250: 0.01 * 150 / 399,
            300: 0.01 * 200 / 399,
            400: 0.01 * 300 / 399,
            498: 0.01 * 398 / 399,
            499: 0.01,
            500: 0.01,
        }
        for iteration, wanted in expected.items():
            self.assertAlmostEqual(coefficient(iteration, 0.01), wanted, places=12)
        self.assertAlmostEqual(coefficient(200, 0.01), 0.00250627, places=8)
        self.assertAlmostEqual(coefficient(250, 0.01), 0.00375940, places=8)
        self.assertAlmostEqual(coefficient(300, 0.01), 0.00501253, places=8)
        self.assertAlmostEqual(coefficient(400, 0.01), 0.00751880, places=8)

        self.assertFalse(updates_enabled(99))
        self.assertTrue(updates_enabled(100))
        self.assertFalse(unlocked(499))
        self.assertTrue(unlocked(500))
        for iteration in expected:
            self.assertEqual(coefficient(iteration, 0.0), 0.0)
        self.assertAlmostEqual(coefficient(499, 0.0005), 0.0005)

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
    def test_stage2_session_keeps_legacy_metric_keys_and_checkpoint_shape(self):
        import torch
        from rl.MGDP.amp.session import AMPSession

        class Dataset:
            def sample(self, count, device):
                return torch.zeros(count, 30, device=device), torch.zeros(count, 30, device=device)

        session = AMPSession(Dataset(), 2, "cpu", {"amp_reward_coef": 0.01})
        state = torch.zeros(2, 30)
        task = torch.full((2,), 0.02)
        _, metrics = session.reward(state, state, task, torch.tensor([0, 3]))
        self.assertEqual(
            set(metrics), {"task_reward", "style_reward", "total_reward", "policy_logit"}
        )
        self.assertNotIn("replay", session.state_dict())
        self.assertEqual(
            set(session.state_dict()),
            {"discriminator", "optimizer", "normalizer", "iteration"},
        )

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
        self.assertTrue(torch.allclose(actual, torch.tensor([0.03, 0.03, 0.02])))
        self.assertTrue(torch.equal(combine_reward(task, logits, classes, 1, 0.0), task))
        stage2 = combine_reward(task, logits, classes, stage=2, coef=0.01)
        self.assertTrue(torch.allclose(stage2[:2], torch.tensor([0.03, 0.0225])))

    def test_stage1_reward_uses_explicit_iteration_schedule(self):
        import torch
        from rl.MGDP.amp.session import AMPSession

        class Dataset:
            def sample(self, count, device):
                return torch.zeros(count, 30, device=device), torch.zeros(count, 30, device=device)

        class UnitLogit(torch.nn.Module):
            def forward(self, pair):
                return torch.ones(pair.shape[0], device=pair.device)

        config = {
            "amp_reward_coef": 0.01,
            "amp_stratified_replay": True,
            "amp_batch_size": 512,
            "amp_replay_rollouts": 2,
            "num_steps_per_env": 24,
        }
        session = AMPSession(Dataset(), 1, "cpu", config)
        session.discriminator = UnitLogit()
        state = torch.zeros(4, 30)
        task = torch.full((4,), 0.02)
        classes = torch.tensor([0, 5, 0, 5])
        anchors = torch.tensor([True, False, True, False])
        warmup, _ = session.reward(state, state, task, classes, iteration=100,
                                   is_anchor=anchors)
        full, metrics = session.reward(state, state, task, classes, iteration=499,
                                       is_anchor=anchors)
        self.assertTrue(torch.equal(warmup, task))
        self.assertTrue(torch.allclose(full, torch.full((4,), 0.03)))
        self.assertEqual(metrics["anchor_r_amp_raw_sum"], 2.0)
        self.assertEqual(metrics["course_r_amp_raw_sum"], 2.0)

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
        penalty = discriminator.gradient_penalty(pair, coefficient=7.0)
        self.assertTrue(torch.isfinite(penalty))


if __name__ == "__main__":
    unittest.main()
