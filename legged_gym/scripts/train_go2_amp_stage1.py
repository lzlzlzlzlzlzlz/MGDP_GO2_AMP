"""Train the MGDP Go2 AMP Stage 1 task."""

import isaacgym  # noqa: F401; Isaac Gym must be imported before torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.scripts.train import train
from legged_gym.utils import get_args
from pathlib import Path


def resolve_stage1_output_name(output_name, coefficient, seed, root):
    """Preserve explicit paths and isolate automatically named coefficient runs."""
    if output_name and output_name != "debug":
        return output_name
    coefficient_label = format(float(coefficient), ".15g").replace("-", "m").replace(".", "p")
    return str(
        Path(root)
        / "outputs"
        / "go2_amp"
        / f"stage1_task_priority_amp{coefficient_label}_seed{int(seed)}"
    )


if __name__ == "__main__":
    args = get_args()
    args.task = "go2_amp_stage1"
    args.algo = "MGDP"
    resolved_coefficient = 0.01 if args.amp_reward_coef is None else args.amp_reward_coef
    resolved_seed = 1 if args.seed is None else args.seed
    args.output_name = resolve_stage1_output_name(
        args.output_name, resolved_coefficient, resolved_seed, LEGGED_GYM_ROOT_DIR
    )
    if args.resume:
        resume_path = Path(args.resume_name or args.output_name)
        if not resume_path.is_absolute():
            from_project = Path.cwd() / resume_path
            from_gym = Path(LEGGED_GYM_ROOT_DIR) / resume_path
            resume_path = from_project if from_project.is_dir() else from_gym
        if not resume_path.is_dir():
            raise SystemExit(f"Stage 1 run directory not found: {resume_path}")
        args.resume_name = str(resume_path.resolve())
        args.checkpoint_model = args.checkpoint_model or "last.pt"
    train(args)
