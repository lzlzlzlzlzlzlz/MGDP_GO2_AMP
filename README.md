# MGDP Go2 AMP

This project preserves MGDP's Go2 control, perception, world model, PPO, terrain tasks and curriculum, and replaces the hand-crafted gait style terms with a WMP-style adversarial motion prior. Both Stage 1 (`mix` terrain) and Stage 2 (`gap_parkour`) use AMP. The sibling `MGDP`, `WMP` and `amp_go2` projects are unchanged.

## Method

- The expert source is the 17 copied Go2 trajectories in `datasets/go2_motion`. Stage 1 samples stance, translation and turn groups with weights 0.10/0.70/0.20. Stage 2 uses stance, forward and slow-turn groups with weights 0.10/0.80/0.10.
- Each discriminator input joins two consecutive 30-value states: 12 joint positions, three body-frame linear velocities, three body-frame angular velocities, and 12 joint velocities. Transitions never cross motion clips. Terminal policy transitions use the state captured before Isaac Gym resets an environment.
- The least-squares discriminator targets +1 for expert transitions and -1 for policy transitions, with an expert gradient penalty. The bounded style term is `0.01 * max(0, 1 - (D - 1)^2 / 4)` and is added to MGDP's task reward. It is gated at 1.0 on easy terrain classes and 0.25 otherwise. Stage 1 easy classes are 0–4; Stage 2 are 0–2.
- The original `motion_trot`, `feet_air_time`, `motion_bound`, and `motion_pace` scales are zero only in the two new tasks. The original task registrations remain available. The policy/world-model learning path stays MGDP's original path.

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

## Replay the expert motions on the project Go2

`legged_gym/scripts/replay_go2_amp_dataset.py` loads the registered
`go2_amp_stage1` task, and therefore the same
`legged_gym/resources/robots/all_dog/go2/urdf/go2.urdf` asset used by AMP
training. It writes each 61-D expert frame directly into the Isaac Gym root
and DOF state tensors. It does not load a policy or call `env.step`, so PD,
gravity, rewards and resets cannot change a frame before it is displayed.
The script deliberately forces Isaac Gym's CPU tensor pipeline because GPU
tensor setters are deferred until the next physics step; PhysX may still use
the selected CUDA simulation device. With one replay environment the tensor
copy cost is negligible, and no physics step is needed between expert frames.

At every displayed frame the script also compares the environment's real
`get_amp_observations()` result with expert columns 7-18 and 31-48. The replay
stops immediately if any of the 30 values differs by more than `1e-5` by
default.

From the project root on the Linux Isaac Gym host, open an interactive viewer
for one clip:

```bash
export PYTHONPATH="$PWD/legged_gym:$PYTHONPATH"
python legged_gym/scripts/replay_go2_amp_dataset.py \
  --clip go2_forward.txt \
  --graphics_device_num 0
```

Replay all 17 clips and record the Isaac Gym camera sensor directly to MP4:

```bash
python legged_gym/scripts/replay_go2_amp_dataset.py \
  --all \
  --headless \
  --record \
  --output outputs/go2_amp_replay/all_motions.mp4 \
  --graphics_device_num 0
```

An interactive viewer requires a working X11/VNC display. Headless recording
still requires a usable NVIDIA graphics device because frames are rendered by
an Isaac Gym camera sensor. Use `--record` without `--headless` to view and
record simultaneously. Other replay options include `--loop`, `--tolerance`,
`--width`, `--height`, and `--camera_distance`. Existing MP4 files are never
replaced unless `--overwrite` is supplied. `--loop` is intentionally limited
to the interactive viewer.

Run short Isaac Gym smoke jobs first:

```bash
python legged_gym/scripts/train_go2_amp_stage1.py --headless --num_envs 64 --max_iterations 2
python legged_gym/scripts/train_go2_amp_stage2.py --headless --num_envs 64 --max_iterations 2 --resume_name "$PWD/legged_gym/outputs/go2_amp/stage1"
```

Stage 2 loads the Stage 1 `stage1_nn/last.pt` policy and AMP state and its matching `stage1_nn/wm_last.pt` world model. The Stage 2 wrapper also accepts the project-relative `--resume_name legged_gym/outputs/go2_amp/stage1` or gym-relative `--resume_name outputs/go2_amp/stage1`. `--checkpoint_model model_N.pt` pairs with `wm_N.pt`. For an original MGDP checkpoint without AMP state, use `--amp_policy_only` explicitly with `--resume_name`; this initializes a fresh discriminator/normalizer and starts iteration numbering anew. A normal resume rejects a checkpoint missing `amp_state`.

To continue Stage 1 in its own output directory, pass `--resume --checkpoint_model last.pt`. An explicit `--resume_name` may be absolute or relative to the project or `legged_gym` directory.

The default run directories are `legged_gym/outputs/go2_amp/stage1` and `legged_gym/outputs/go2_amp/stage2`. Preserve them before running a new experiment. To ablate the style term, set `amp_reward_coef = 0.0` in the relevant new train config only; the task reward is then unchanged.

For evaluation, compare original MGDP and MGDP Go2 AMP under identical seeds and terrain distributions. Record tracking error, falls, distance/progress, obstacle success, energy, reward components and performance by terrain class. A short smoke rollout checks integration only; it does not establish multi-terrain performance.

## Verification status on this Windows workspace

Python 3.11 and NumPy are available locally; PyTorch, Isaac Gym and the matching CUDA training stack are not installed. The local suite runs 18 tests with 7 PyTorch-dependent skips; pure-data checks, all-clip validation, `compileall` and `git diff --check` pass. PyTorch unit tests must pass on the training host. Isaac Gym Stage 1/Stage 2 smoke runs, checkpoint reload and full training remain pending on that host; no control-performance claim is made yet.

See `SOURCES.md` for provenance and `docs/superpowers/specs/2026-09-21-mgdp-go2-amp-design.md` for design decisions.
