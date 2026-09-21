"""Go2-only variants of MGDP's two terrain stages."""

from legged_gym.envs.random_dog.random_dog_config_stage1 import RandomCfgStage1, RandomCfgPPOStage1
from legged_gym.envs.random_dog.random_dog_config_stage2 import RandomCfgStage2, RandomCfgPPOStage2


class Go2AmpStage1Cfg(RandomCfgStage1):
    class asset(RandomCfgStage1.asset):
        asset_name = ["go2"]

    class rewards(RandomCfgStage1.rewards):
        class scales(RandomCfgStage1.rewards.scales):
            motion_trot = 0.0
            feet_air_time = 0.0
            motion_bound = 0.0
            motion_pace = 0.0


class Go2AmpStage1TrainCfg(RandomCfgPPOStage1):
    class runner(RandomCfgPPOStage1.runner):
        amp_enabled = True
        amp_stage = 1
        amp_reward_coef = 0.01
        amp_easy_gate = 1.0
        amp_hard_gate = 0.25
        experiment_name = "go2_amp_stage1"


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
        experiment_name = "go2_amp_stage2"


Go2AmpStage1Cfg.encoder_ppo_ref = Go2AmpStage1TrainCfg
Go2AmpStage2Cfg.encoder_ppo_ref = Go2AmpStage2TrainCfg
