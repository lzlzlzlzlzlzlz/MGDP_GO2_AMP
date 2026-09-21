"""Train the MGDP Go2 AMP Stage 1 task."""

import isaacgym  # noqa: F401; Isaac Gym must be imported before torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.scripts.train import train
from legged_gym.utils import get_args
import os


if __name__ == "__main__":
    args = get_args()
    args.task = "go2_amp_stage1"
    args.algo = "MGDP"
    args.output_name = os.path.join(LEGGED_GYM_ROOT_DIR, "outputs", "go2_amp", "stage1")
    if args.resume:
        args.resume_name = args.resume_name or args.output_name
        args.checkpoint_model = args.checkpoint_model or "last.pt"
    train(args)
