"""Train Stage 2 from a Stage 1 policy and world-model checkpoint."""

import isaacgym  # noqa: F401

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.scripts.train import train
from legged_gym.utils import get_args
import os
from pathlib import Path


if __name__ == "__main__":
    args = get_args()
    if not args.resume_name:
        raise SystemExit("Stage 2 requires --resume_name pointing to a Stage 1 run")
    resume_path = Path(args.resume_name)
    if not resume_path.is_absolute():
        from_project = Path.cwd() / resume_path
        from_gym = Path(LEGGED_GYM_ROOT_DIR) / resume_path
        resume_path = from_project if from_project.is_dir() else from_gym
    if not resume_path.is_dir():
        raise SystemExit(f"Stage 1 run directory not found: {resume_path}")
    args.resume_name = str(resume_path.resolve())
    args.task = "go2_amp_stage2"
    args.algo = "MGDP"
    args.resume = True
    args.checkpoint_model = args.checkpoint_model or "last.pt"
    if args.checkpoint_model != "last.pt" and not (
            args.checkpoint_model.startswith("model_") and args.checkpoint_model.endswith(".pt")):
        raise SystemExit("Use last.pt or model_N.pt as --checkpoint_model")
    if args.checkpoint_model == "model_best.pt" and not args.amp_policy_only:
        raise SystemExit("model_best.pt cannot be used for full resume because its world-model iteration may differ")
    world_model_file = ("wm_last.pt" if args.checkpoint_model == "last.pt" else
                        "wm_" + args.checkpoint_model[len("model_"):])
    for filename in (args.checkpoint_model, world_model_file):
        if not (resume_path / "stage1_nn" / filename).is_file():
            raise SystemExit(f"Required Stage 1 checkpoint missing: {resume_path / 'stage1_nn' / filename}")
    args.output_name = os.path.join(LEGGED_GYM_ROOT_DIR, "outputs", "go2_amp", "stage2")
    train(args)
