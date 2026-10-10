# MGDP Go2 AMP

This project preserves MGDP's Go2 control, perception, world model, PPO, terrain tasks and curriculum, and replaces the hand-crafted gait-pattern terms with a WMP-style adversarial motion prior. The task-priority redesign in this document applies only to the new Go2 AMP Stage 1 task. The original MGDP tasks and Go2 AMP Stage 2 keep their existing defaults.

## Method

- The expert source is the copied Go2 trajectories in `datasets/go2_motion`. Task-priority Stage 1 uses only stance and the three forward clips, with group weights 0.25/0.75. Stage 2 retains its earlier expert groups and weights.
- Each discriminator input joins two consecutive 30-value states: 12 joint positions, three body-frame linear velocities, three body-frame angular velocities, and 12 joint velocities. Transitions never cross motion clips. Terminal policy transitions use the state captured before Isaac Gym resets an environment.
- The least-squares discriminator targets +1 for expert transitions and -1 for policy transitions, with an expert gradient penalty. Stage 1 uses the same terrain-independent AMP gate for every environment. Its effective coefficient is scheduled from zero to the selected target; Stage 2 retains its previous reward behavior.
- Stage 1 sets `motion_trot`, `motion_bound`, and `motion_pace` to zero. It deliberately retains `feet_air_time=0.5`, with the per-foot raw term capped at 0.75 s, as a weak hand-written locomotion scaffold. Results must disclose this: the project does not claim that AMP replaces every locomotion-shaping term.
- Stage 1 constructs exactly six original-MGDP simple terrain columns. There is no independent flat generator, flat column, or flat terrain class. Logical AMP anchors reuse column 0 at level 0 and are identified only by `is_amp_anchor`.

## Training host

Use a Linux machine with a supported NVIDIA GPU, CUDA, PyTorch and Isaac Gym installation. Start from MGDP's documented Python 3.8.20 / PyTorch 1.10.0+cu113 / CUDA 11.3 stack. Isaac Gym is a separate NVIDIA installation and is not redistributed here. From a compatible environment, install Isaac Gym from its own `python` directory, then run:

```bash
pip install -r legged_gym/requirement-gpu.txt
pip install -e ./legged_gym
pip install -e ./warp_sensor
export PYTHONPATH="$PWD/legged_gym:$PYTHONPATH"
```

Confirm the specific CUDA driver, Warp, Isaac Gym and PyTorch combination on the rented host before a long run.

Run the full CPU/PyTorch suite before simulation:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

## Task-priority Stage 1 experiment protocol

Launch the two primary runs from zero with the same seed and separate output directories:

```bash
python legged_gym/scripts/train_go2_amp_stage1.py --headless --num_envs 4096 --seed 1 --amp_reward_coef 0 --output_name outputs/go2_amp/stage1_task_priority_scaffold_seed1
python legged_gym/scripts/train_go2_amp_stage1.py --headless --num_envs 4096 --seed 1 --amp_reward_coef 0.01 --output_name outputs/go2_amp/stage1_task_priority_amp0p01_seed1
```

Use `--max_iterations 20` on the initial launch for the save/restore smoke. Resume only the same run, with the same coefficient and output directory. For example, the AMP run continues from its own checkpoint with:

```bash
python legged_gym/scripts/train_go2_amp_stage1.py --headless --num_envs 4096 --seed 1 --amp_reward_coef 0.01 --output_name outputs/go2_amp/stage1_task_priority_amp0p01_seed1 --resume --resume_name outputs/go2_amp/stage1_task_priority_amp0p01_seed1 --checkpoint_model last.pt --max_iterations 80
```

`--max_iterations` is the number of additional iterations after resume. Therefore the exact cumulative checkpoints are reached with 20 iterations from zero, followed by same-run increments of 80, 100, 50, 250, 500, and 9000 to reach 100, 200, 250, 500, 1000, and 10000. Stop after each target for human review; do not queue the whole sequence blindly.

The first comparison is Scaffold-only (`0`) against the primary AMP pressure test (`0.01`). The lower coefficient is not part of that pair. Only if a researcher decides after reviewing the `0.01` run that a lower-pressure experiment is warranted, launch this independent fallback from zero:

```bash
python legged_gym/scripts/train_go2_amp_stage1.py --headless --num_envs 4096 --seed 1 --amp_reward_coef 0.0005 --output_name outputs/go2_amp/stage1_task_priority_amp0p0005_seed1
```

Never resume one coefficient from another coefficient's checkpoint, and do not use an old-architecture checkpoint for formal task-priority training. A full same-run resume requires the matching `stage1_nn/last.pt` policy/optimizer/AMP checkpoint and `stage1_nn/wm_last.pt` world-model checkpoint. The loader rejects missing or incompatible AMP metadata before partially restoring state. `--amp_policy_only` remains an explicit policy warm-start facility, not a valid way to resume a formal run in this protocol.

The zero-based Stage 1 schedule is fixed:

- Iterations 0–99: level-0 task-only warmup; AMP effective coefficient is zero and the discriminator does not update, while both replay pools continue recording.
- Iteration 100: discriminator updates start after the rollout; that rollout still receives zero AMP reward.
- Iterations 101–498: the coefficient ramps linearly as `target * (iteration - 100) / 399`.
- Iteration 499 onward: the full selected target is active.
- Iteration 500 onward: non-anchor course environments may change level on later resets; anchors stay at column 0, level 0.

For target `0.01`, the effective values are approximately 0.00251 at iteration 200, 0.00376 at 250, 0.00501 at 300, 0.00752 at 400, and exactly 0.01 from 499 onward. The 20-iteration smoke therefore proves integration only; it says nothing about AMP style or terrain performance.

At 20 iterations verify finite values and the paired save/resume path. At 100 preserve the task-only warmup checkpoint. Review the paired 4096-environment runs at 200 and 250, the full-0.01/curriculum boundary at 500, and the early multi-terrain screen at 1000. Only after an explicit human admission decision should a run be extended to at least 10000 iterations.

Every comparison uses a fixed 0.7 m/s forward command, identical terrain seed, camera, and rollout length. Review task progress/tracking/falls, AMP and discriminator distributions, anchor/course diagnostics, `feet_air_time`, and fixed videos together. The stop gates and Stage 1 admission criteria are a researcher checklist, not training-loop control flow: the program only logs diagnostics and saves checkpoints. It never stops or continues a run, switches to `0.0005`, creates another experiment, or enters Stage 2 based on metric values. The researcher reviews the evidence with the collaborator before choosing the next command.

Stage 2 is outside this experiment protocol and is not automatically launched. Its existing manual launcher, configuration, checkpoint behavior, and defaults remain unchanged.

For evaluation, compare original MGDP and MGDP Go2 AMP under identical seeds and terrain distributions. Record tracking error, falls, distance/progress, obstacle success, energy, reward components and performance by terrain class. A short smoke rollout checks integration only; it does not establish multi-terrain performance.

## Verification status on this Windows workspace

Python 3.11 and NumPy are available locally; PyTorch, Isaac Gym, Warp runtime validation, and the matching CUDA training stack are not installed. Pure/source CPU checks run here, while PyTorch-dependent tests skip. The training host must still pass the complete PyTorch suite and perform Stage 1 config startup, the 20-iteration paired save/resume smoke, the 100-iteration task-only checkpoint, paired 4096×200/250 diagnosis, the 4096×500 full-0.01 check, and the 4096×1000 early screen. Each continue/stop choice is manual. No control-performance claim is made from the Windows checks or GPU smoke alone.

See `SOURCES.md` for provenance and `docs/superpowers/specs/2026-10-09-mgdp-go2-amp-training-redesign.md` for the confirmed task-priority design.
