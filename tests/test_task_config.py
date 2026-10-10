"""Source-level contract checks while Isaac Gym is unavailable locally."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1] / "legged_gym"


def _find_class(node, name):
    for child in node.body:
        if isinstance(child, ast.ClassDef) and child.name == name:
            return child
    raise AssertionError(f"class {name} not found")


def _literal_assignments(node):
    result = {}
    for child in node.body:
        if isinstance(child, ast.Assign) and len(child.targets) == 1:
            target = child.targets[0]
            if isinstance(target, ast.Name):
                try:
                    result[target.id] = ast.literal_eval(child.value)
                except ValueError:
                    continue
    return result


def _load_function(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name),
        None,
    )
    if function is None:
        raise AssertionError(f"function {name} not found in {path}")
    namespace = {"Path": Path}
    ast.fix_missing_locations(function)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class _LooseArgs(SimpleNamespace):
    def __getattr__(self, _name):
        return None


class TestGo2TaskConfig(unittest.TestCase):
    def test_new_tasks_are_registered(self):
        source = (ROOT / "legged_gym/envs/__init__.py").read_text(encoding="utf-8")
        self.assertIn('task_registry.register("go2_amp_stage1"', source)
        self.assertIn('task_registry.register("go2_amp_stage2"', source)

    def test_original_tasks_remain_registered(self):
        source = (ROOT / "legged_gym/envs/__init__.py").read_text(encoding="utf-8")
        self.assertIn('task_registry.register("random_dog_stage1"', source)
        self.assertIn('task_registry.register("random_dog_stage2"', source)

    def test_stage1_config_is_task_priority_forward_locomotion(self):
        path = ROOT / "legged_gym/envs/go2_amp/config.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        stage1 = _find_class(tree, "Go2AmpStage1Cfg")

        rewards = _find_class(stage1, "rewards")
        reward_values = _literal_assignments(rewards)
        scales = _literal_assignments(_find_class(rewards, "scales"))
        self.assertEqual(
            {name: scales[name] for name in ("motion_trot", "motion_bound", "motion_pace")},
            {"motion_trot": 0.0, "motion_bound": 0.0, "motion_pace": 0.0},
        )
        self.assertEqual(scales["feet_air_time"], 0.5)
        self.assertEqual(reward_values["amp_air_time_cap"], 0.75)

        terrain = _literal_assignments(_find_class(stage1, "terrain"))
        commands_node = _find_class(stage1, "commands")
        commands = _literal_assignments(commands_node)
        command_ranges = _literal_assignments(_find_class(commands_node, "ranges"))
        domain_rand = _literal_assignments(_find_class(stage1, "domain_rand"))
        self.assertEqual(terrain["max_init_terrain_level"], 0)
        self.assertIs(commands["curriculum"], False)
        self.assertIs(commands["heading_command"], True)
        self.assertIs(domain_rand["push_robots"], False)
        self.assertEqual(command_ranges["lin_vel_x"], [0.0, 0.8])
        self.assertEqual(command_ranges["lin_vel_y"], [0.0, 0.0])
        self.assertEqual(command_ranges["heading"], [0.0, 0.0])
        self.assertEqual(command_ranges["new_lin_vel_x"], [0.0, 0.8])
        self.assertEqual(command_ranges["new_lin_vel_y"], [0.0, 0.0])
        self.assertEqual(command_ranges["new_heading"], [0.0, 0.0])

        train = _find_class(tree, "Go2AmpStage1TrainCfg")
        runner = _literal_assignments(_find_class(train, "runner"))
        expected = {
            "experiment_name": "go2_amp_stage1_task_priority",
            "num_steps_per_env": 24,
            "amp_reward_coef": 0.01,
            "amp_easy_gate": 1.0,
            "amp_hard_gate": 1.0,
            "amp_updates_per_iter": 1,
            "amp_batch_size": 512,
            "amp_replay_rollouts": 2,
            "amp_learning_rate": 1e-4,
            "amp_gradient_penalty_coef": 10.0,
            "amp_discriminator_start_iteration": 100,
            "amp_ramp_end_iteration": 499,
            "amp_curriculum_unlock_iteration": 500,
            "amp_anchor_fraction": 0.15,
            "amp_stratified_replay": True,
        }
        self.assertEqual({name: runner[name] for name in expected}, expected)

    def test_stage2_config_keeps_legacy_values(self):
        path = ROOT / "legged_gym/envs/go2_amp/config.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        stage2 = _find_class(tree, "Go2AmpStage2Cfg")
        scales = _literal_assignments(_find_class(_find_class(stage2, "rewards"), "scales"))
        self.assertEqual(scales["feet_air_time"], 0.0)
        runner = _literal_assignments(
            _find_class(_find_class(tree, "Go2AmpStage2TrainCfg"), "runner")
        )
        self.assertEqual(runner["experiment_name"], "go2_amp_stage2")
        self.assertEqual(runner["amp_hard_gate"], 0.25)

    def test_amp_reward_coefficient_cli_overrides_only_runner(self):
        path = ROOT / "legged_gym/utils/helpers.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        get_args = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_args"
        )
        custom_parameters = next(
            node.value
            for node in get_args.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "custom_parameters"
                    for target in node.targets)
        )
        amp_option = None
        for item in custom_parameters.elts:
            fields = {
                key.value: value
                for key, value in zip(item.keys, item.values)
                if isinstance(key, ast.Constant)
            }
            if ast.literal_eval(fields["name"]) == "--amp_reward_coef":
                amp_option = fields
                break
        self.assertIsNotNone(amp_option)
        self.assertIsInstance(amp_option["type"], ast.Name)
        self.assertEqual(amp_option["type"].id, "float")

        update = _load_function(path, "update_cfg_from_args")
        for coefficient in (0.0, 0.01, 0.0005):
            runner = SimpleNamespace(amp_reward_coef=99.0, sentinel="unchanged")
            train_cfg = SimpleNamespace(
                seed=1, runner=runner, Encoder=SimpleNamespace(priv_info=False)
            )
            update(None, train_cfg, _LooseArgs(amp_reward_coef=coefficient))
            self.assertEqual(runner.amp_reward_coef, coefficient)
            self.assertEqual(runner.sentinel, "unchanged")

    def test_stage1_output_name_is_deterministic_but_explicit_name_wins(self):
        path = ROOT / "scripts/train_go2_amp_stage1.py"
        resolver = _load_function(path, "resolve_stage1_output_name")
        explicit = "outputs/custom/run-a"
        self.assertEqual(resolver(explicit, 0.01, 7, "repo"), explicit)
        self.assertEqual(
            Path(resolver("debug", 0.01, 1, "repo")).name,
            "stage1_task_priority_amp0p01_seed1",
        )
        self.assertEqual(
            Path(resolver("debug", 0.0, 1, "repo")).name,
            "stage1_task_priority_amp0_seed1",
        )
        self.assertEqual(
            Path(resolver(None, 0.0005, 3, "repo")).name,
            "stage1_task_priority_amp0p0005_seed3",
        )


if __name__ == "__main__":
    unittest.main()
