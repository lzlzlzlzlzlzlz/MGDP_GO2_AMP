"""Train Stage 2 from a Stage 1 policy and world-model checkpoint."""

import isaacgym  # noqa: F401

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.scripts.train import train
from legged_gym.utils import get_args
import os


if __name__ == "__main__":
    args = get_args()
    if not args.resume_name:
        raise SystemExit("Stage 2 requires --resume_name pointing to a Stage 1 run")
    args.task = "go2_amp_stage2"
    args.algo = "MGDP"
    args.resume = True
    args.checkpoint_model = args.checkpoint_model or "last.pt"
    args.output_name = os.path.join(LEGGED_GYM_ROOT_DIR, "outputs", "go2_amp", "stage2")
    train(args)
