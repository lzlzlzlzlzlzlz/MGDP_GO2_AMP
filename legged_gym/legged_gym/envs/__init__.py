
from .base.legged_robot import LeggedRobot

from .baseline.legged_robot_terrains import Legged_terrains
from .baseline.legged_robot_camera import Legged_camera, CameraMixin
from .baseline.legged_robot_rewards import Legged_rewards

from legged_gym.utils.task_registry import task_registry


from .random_dog.random_dog import Randomdog
from .random_dog.random_dog_config_stage1 import RandomCfgStage1, RandomCfgPPOStage1
from .random_dog.random_dog_config_stage2 import RandomCfgStage2, RandomCfgPPOStage2

# Stage 1: data collection + world model training (mix terrains, etc.)
task_registry.register("random_dog_stage1", Randomdog, RandomCfgStage1(), RandomCfgPPOStage1())

# Stage 2: align with paper setup (gap_parkour, rewards, etc.); can resume from stage1 WM/policy
task_registry.register("random_dog_stage2", Randomdog, RandomCfgStage2(), RandomCfgPPOStage2())

from .go2_amp.config import (Go2AmpStage1Cfg, Go2AmpStage1TrainCfg,
                             Go2AmpStage2Cfg, Go2AmpStage2TrainCfg)
from .go2_amp.env import Go2AmpRandomDog

task_registry.register("go2_amp_stage1", Go2AmpRandomDog, Go2AmpStage1Cfg(), Go2AmpStage1TrainCfg())
task_registry.register("go2_amp_stage2", Go2AmpRandomDog, Go2AmpStage2Cfg(), Go2AmpStage2TrainCfg())

