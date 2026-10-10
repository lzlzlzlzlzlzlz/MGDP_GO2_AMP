"""Go2-only variants of MGDP's two terrain stages."""

from legged_gym.envs.random_dog.random_dog_config_stage1 import RandomCfgStage1, RandomCfgPPOStage1
from legged_gym.envs.random_dog.random_dog_config_stage2 import RandomCfgStage2, RandomCfgPPOStage2

STAGE1_GROUPS = {
    "stance": ["go2_stance.txt"],
    "forward": ["go2_forward.txt", "go2_forward_fast.txt", "go2_forward_faster.txt"],
}
STAGE1_WEIGHTS = {"stance": 0.25, "forward": 0.75}
STAGE2_GROUPS = {
    "stance": ["go2_stance.txt"],
    "forward": ["go2_forward.txt", "go2_forward_fast.txt", "go2_forward_faster.txt"],
    "turn": ["go2_turn_left.txt", "go2_turn_right.txt"],
}
STAGE2_WEIGHTS = {"stance": 0.10, "forward": 0.80, "turn": 0.10}


class Go2AmpStage1Cfg(RandomCfgStage1):
    class asset(RandomCfgStage1.asset):
        asset_name = ["go2"]

    class terrain(RandomCfgStage1.terrain):
        max_init_terrain_level = 0

    class commands(RandomCfgStage1.commands):
        curriculum = False
        heading_command = True

        class ranges(RandomCfgStage1.commands.ranges):
            lin_vel_x = [0.0, 0.8]
            lin_vel_y = [0.0, 0.0]
            heading = [0.0, 0.0]
            new_lin_vel_x = [0.0, 0.8]
            new_lin_vel_y = [0.0, 0.0]
            new_heading = [0.0, 0.0]

    class domain_rand(RandomCfgStage1.domain_rand):
        push_robots = False

    class rewards(RandomCfgStage1.rewards):
        amp_air_time_cap = 0.75

        class scales(RandomCfgStage1.rewards.scales):
            motion_trot = 0.0
            feet_air_time = 0.5
            motion_bound = 0.0
            motion_pace = 0.0


class Go2AmpStage1TrainCfg(RandomCfgPPOStage1):
    class runner(RandomCfgPPOStage1.runner):
        amp_enabled = True
        amp_stage = 1
        amp_reward_coef = 0.01
        amp_easy_gate = 1.0
        amp_hard_gate = 1.0
        amp_updates_per_iter = 1
        amp_learning_rate = 1e-4
        amp_batch_size = 512
        amp_replay_rollouts = 2
        amp_replay_capacity = 100000
        amp_gradient_penalty_coef = 10.0
        amp_discriminator_start_iteration = 100
        amp_ramp_end_iteration = 499
        amp_curriculum_unlock_iteration = 500
        amp_anchor_fraction = 0.15
        amp_stratified_replay = True
        num_steps_per_env = 24
        experiment_name = "go2_amp_stage1_task_priority"
        amp_groups = STAGE1_GROUPS
        amp_group_weights = STAGE1_WEIGHTS


class Go2AmpStage2Cfg(RandomCfgStage2):
    class asset(RandomCfgStage2.asset):
        asset_name = ["go2"]

    class rewards(RandomCfgStage2.rewards):
        class scales(RandomCfgStage2.rewards.scales):
            motion_trot = 0.0
            feet_air_time = 0.0
            motion_bound = 0.0
            motion_pace = 0.0


class Go2AmpStage2TrainCfg(RandomCfgPPOStage2):
    class runner(RandomCfgPPOStage2.runner):
        amp_enabled = True
        amp_stage = 2
        amp_reward_coef = 0.01
        amp_easy_gate = 1.0
        amp_hard_gate = 0.25
        amp_learning_rate = 1e-4
        amp_batch_size = 512
        amp_replay_capacity = 100000
        experiment_name = "go2_amp_stage2"
        amp_groups = STAGE2_GROUPS
        amp_group_weights = STAGE2_WEIGHTS


Go2AmpStage1Cfg.encoder_ppo_ref = Go2AmpStage1TrainCfg
Go2AmpStage2Cfg.encoder_ppo_ref = Go2AmpStage2TrainCfg
