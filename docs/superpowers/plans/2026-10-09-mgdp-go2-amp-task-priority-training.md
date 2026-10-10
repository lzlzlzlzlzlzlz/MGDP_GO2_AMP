# MGDP Go2 AMP Task-Priority Stage 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the minimal Stage 1 training loop in which MGDP learns progressive multi-terrain forward locomotion while a stance/forward AMP prior improves motion style without controlling terrain-specific behavior.

**Architecture:** Keep MGDP perception, World Model, PPO, robot control, task rewards and original task registrations intact. Add Stage-1-only Go2 AMP configuration, a six-column terrain grid and logical anchor allocation in the Go2 AMP environment, then extend the existing AMP session with a deterministic warmup/ramp schedule and two replay pools. Anchors reuse the negative-slope column at level 0; there is no independent flat generator, column or terrain class. The existing MGDP runner remains the owner of rollout/PPO order and receives only the small hooks needed for iteration propagation, metric aggregation and complete AMP checkpointing; staged stop/admission decisions remain a human experiment workflow, not code control flow.

**Tech Stack:** Python 3.8, NumPy, PyTorch 1.10/CUDA 11.3, Isaac Gym, Warp, `unittest`.

**Spec:** `docs/superpowers/specs/2026-10-09-mgdp-go2-amp-training-redesign.md`

## Global Constraints

- Do not modify, delete, stage or commit `legged_gym/scripts/record_go2_amp_stage1.py` or `tests/test_record_go2_amp_stage1.py`.
- Do not change the behavior of `random_dog_stage1`, `random_dog_stage2`, or the existing Go2 AMP Stage 2 task.
- Preserve the existing 30-D AMP state mapping, terminal-state transition replacement and MGDP PPO/World Model update path.
- Stage 1 uses zero-based iterations, `num_steps_per_env=24`, first-pair target coefficients 0 and 0.01, one discriminator update per eligible iteration, batch 512, two replay rollouts, learning rate `1e-4` and gradient penalty coefficient 10.0; runtime also accepts 0.0005 only as a separately launched fallback experiment.
- The existing config key `amp_reward_coef` is the implementation name for the design's `lambda_target`; Stage 1 sets `amp_easy_gate=amp_hard_gate=1.0`, so terrain class never rescales its scheduled contribution.
- Stage 1 keeps `heading_command=True`; both ordinary and `new_*` command ranges are forward `[0.0, 0.8]`, lateral `[0.0, 0.0]`, heading `[0.0, 0.0]`.
- Stage 1 has exactly six explicit terrain columns. Anchors are identified only by `is_amp_anchor` and reuse column 0 at level 0; do not add a `flat` generator, column or terrain class.
- Keep MGDP's current highest-level success behavior: randomly fall back to a lower terrain level.
- Use `go2_amp_stage1_task_priority` as the new experiment name and never resume the new architecture from old checkpoints.
- Do not implement automatic stop/continue, admission scoring, coefficient fallback, next-run creation or metric-driven mutation. Code supplies configuration, logs, checkpoints and isolated output directories; the researcher makes every staged experiment decision.
- Tests must run without importing Isaac Gym whenever the behavior can be isolated in a pure Python/NumPy/PyTorch helper.

## Review Focus

1. `num_envs=2` must still create one anchor and one course environment even though both initially map to the same column-0 zero-slope geometry; Task 2 tests the boolean group identity and smallest supported split.
2. Hard-terrain command sampling must not fall back to the inherited `new_*=[-1,1]` ranges; Task 1 tests all six ordinary/new range values.
3. Iterations 99, 100, 101, 200, 250, 300, 400, 498, 499 and 500 must have exact 0.01 coefficient/update/unlock behavior, including after resume; Tasks 3 and 4 test these boundaries.
4. A course environment that succeeds above the maximum level must retain MGDP's random fallback, while anchors stay at column 0 level 0; Task 2 also proves the 4096-environment split is 614 anchors plus 698 course zero-slope environments, not a seventh flat column.
5. A checkpoint with mismatched coefficient, schedule, replay, optimizer or expert-group settings must fail before partial AMP state is loaded; Task 4 tests compatibility rejection.

---

## File Map

- `legged_gym/legged_gym/envs/go2_amp/config.py`: Stage 1 rewards, command ranges, terrain columns, expert groups and fixed AMP settings.
- `legged_gym/legged_gym/envs/go2_amp/terrain.py`: pure column/anchor allocation helpers shared by environment code and CPU tests.
- `legged_gym/legged_gym/envs/go2_amp/env.py`: anchor state, Stage 1 curriculum gating and capped feet-air-time reward.
- `legged_gym/legged_gym/utils/terrain.py`: opt-in explicit-column terrain construction; legacy path remains unchanged.
- `legged_gym/legged_gym/utils/new_terrains/add_mix_terrain.py`: named mix-terrain builder using the existing terrain primitives.
- `legged_gym/rl/MGDP/amp/replay.py`: serializable circular replay buffer used by both anchor and course pools.
- `legged_gym/rl/MGDP/amp/session.py`: schedule helpers, uniform Stage 1 reward, stratified replay sampling, discriminator updates and AMP checkpoint state.
- `legged_gym/rl/MGDP/amp/discriminator.py`: configurable gradient-penalty coefficient.
- `legged_gym/rl/MGDP/runners/policy_runner.py`: iteration propagation, anchor mask recording, rollout aggregation and AMP save/load hooks.
- `legged_gym/legged_gym/utils/helpers.py`: optional `--amp_reward_coef` override.
- `legged_gym/scripts/train_go2_amp_stage1.py`: isolated output naming while honoring an explicit output path.
- `tests/test_task_config.py`, `tests/test_amp_terrain.py`, `tests/test_amp_reward.py`, `tests/test_amp_checkpoint.py`, `tests/test_amp_rollout.py`: required CPU contracts.
- `README.md`: new Stage 1 launch, resume and comparison workflow.

### Task 1: Stage 1 configuration, capped scaffold reward and isolated launch

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2_amp/config.py`
- Modify: `legged_gym/legged_gym/envs/go2_amp/env.py`
- Modify: `legged_gym/legged_gym/utils/helpers.py`
- Modify: `legged_gym/scripts/train_go2_amp_stage1.py`
- Modify: `tests/test_task_config.py`
- Modify: `tests/test_motion_data.py`
- Modify: `tests/test_amp_reward.py`

**Interfaces:**
- Produces runner config keys `amp_reward_coef`, `amp_updates_per_iter`, `amp_batch_size`, `amp_replay_rollouts`, `amp_learning_rate`, `amp_gradient_penalty_coef`, `amp_discriminator_start_iteration`, `amp_ramp_end_iteration`, `amp_curriculum_unlock_iteration`, `amp_anchor_fraction` and `amp_stratified_replay`.
- Produces `Go2AmpRandomDog._reward_feet_air_time()` with the existing reward-method interface; a Stage 1-only config marker enables the 0.75 s cap, while Stage 2 delegates to its inherited behavior.
- Produces optional float CLI option `--amp_reward_coef`; Stage 1 defaults to 0.01, while an explicit 0 selects Scaffold-only and 0.0005 selects the documented fallback run. The option overrides only `cfg_train.runner.amp_reward_coef`.
- Preserves Stage 2 config values and launch behavior.

- [ ] **Step 1: Update the source-level config test so it describes the new Stage 1 contract**

In `tests/test_task_config.py`, replace the old assertion that both stages have `feet_air_time=0` with assertions that Stage 1 has:

```python
motion_trot = motion_bound = motion_pace = 0.0
feet_air_time = 0.5
max_init_terrain_level = 0
commands.curriculum is False
commands.heading_command is True
push_robots is False
```

Assert the exact ordinary and `new_*` ranges, `experiment_name="go2_amp_stage1_task_priority"`, default `amp_reward_coef=0.01`, `amp_easy_gate=amp_hard_gate=1.0`, and the remaining fixed AMP parameter values from the spec. Retain assertions that original tasks are registered.

- [ ] **Step 2: Update the motion-data test for the Stage 1 expert subset**

In `tests/test_motion_data.py`, assert `STAGE1_GROUPS` contains only stance plus the three forward clips, weights are `{"stance": 0.25, "forward": 0.75}`, and all 17 source clips still remain present and individually valid on disk.

- [ ] **Step 3: Add failing reward and argument contracts**

In `tests/test_amp_reward.py`, add a source-level or isolated-tensor test proving Go2 AMP caps each airborne duration at 0.75 before subtracting 0.5, while `Randomdog._reward_feet_air_time` remains unchanged. In `tests/test_task_config.py`, assert `helpers.py` defines and applies `--amp_reward_coef`, accepts 0, 0.01 and fallback 0.0005, and the Stage 1 wrapper does not overwrite an explicit non-default `--output_name`.

- [ ] **Step 4: Run the focused tests and verify they fail for the old behavior**

Run:

```bash
python -m unittest tests.test_task_config tests.test_motion_data tests.test_amp_reward -v
```

Expected: failures for the old expert groups, zero feet-air-time scale, missing range overrides, missing CLI override and uncapped reward.

- [ ] **Step 5: Implement the Stage 1 config and reward override**

Set all values from the spec in `Go2AmpStage1Cfg` and `Go2AmpStage1TrainCfg`, including a Stage 1-only `amp_air_time_cap=0.75` marker. Override `Go2AmpRandomDog._reward_feet_air_time`; when the marker is absent, call the inherited implementation unchanged. When present, reuse the existing contact filtering, first-contact and nonzero-command gate, changing only the raw airborne term to `torch.clamp(self.feet_air_time, max=0.75) - 0.5`.

- [ ] **Step 6: Implement coefficient override and deterministic default output naming**

Add `--amp_reward_coef` to `get_args()` and apply it in `update_cfg_from_args`. In `train_go2_amp_stage1.py`, honor an explicitly supplied output name; otherwise derive a new directory containing `stage1_task_priority`, the resolved coefficient label and seed. Keep existing Stage 1 resume-path resolution. Do not add metric-dependent launch, continuation or fallback behavior.

- [ ] **Step 7: Run the focused tests**

Run:

```bash
python -m unittest tests.test_task_config tests.test_motion_data tests.test_amp_reward -v
```

Expected: all focused tests pass; PyTorch-dependent cases may skip only when PyTorch is unavailable.

- [ ] **Step 8: Commit Task 1**

```bash
git add legged_gym/legged_gym/envs/go2_amp/config.py legged_gym/legged_gym/envs/go2_amp/env.py legged_gym/legged_gym/utils/helpers.py legged_gym/scripts/train_go2_amp_stage1.py tests/test_task_config.py tests/test_motion_data.py tests/test_amp_reward.py
git commit -m "feat: configure task-priority Go2 AMP stage one"
```

### Task 2: Explicit terrain columns, anchors and MGDP curriculum preservation

**Files:**
- Create: `legged_gym/legged_gym/envs/go2_amp/terrain.py`
- Modify: `legged_gym/legged_gym/envs/go2_amp/config.py`
- Modify: `legged_gym/legged_gym/envs/go2_amp/env.py`
- Modify: `legged_gym/legged_gym/utils/terrain.py`
- Modify: `legged_gym/legged_gym/utils/new_terrains/add_mix_terrain.py`
- Create: `tests/test_amp_terrain.py`

**Interfaces:**
- `GO2_AMP_TERRAIN_COLUMNS: Tuple[str, ...]` is exactly `("slope down", "slope up", "rough pyramid", "stairs down", "stairs up", "discrete obstacles")`.
- `GO2_AMP_COURSE_COLUMN_CYCLE: Tuple[int, ...]` is exactly `(0, 1, 2, 2, 3, 3, 4, 4, 5, 5)`, preserving the original Stage 1 effective terrain ratio `1:1:2:2:2:2` without cumulative thresholds.
- `compute_anchor_count(num_envs: int, fraction: float = 0.15) -> int` rejects `num_envs < 2` and implements the spec formula using Python `round`.
- `assign_amp_columns(num_envs: int, fraction: float = 0.15) -> Tuple[np.ndarray, np.ndarray]` returns column ids and a boolean anchor mask. The anchor prefix maps to column 0; course ids independently begin at the first element of `GO2_AMP_COURSE_COLUMN_CYCLE` and use its prefix when the course count is not divisible by 10.
- `add_mix_terrain.trimesh_terrain_by_name(terrain, terrain_name, difficulty, add_roughness, num_rows) -> None` builds one named terrain and assigns a stable class id. Negative and positive slopes both retain class 0, rough pyramid uses 1, stairs down/up use 2/3 and discrete obstacles uses 4. There is no `flat` name or new class id.
- `Go2AmpRandomDog.set_amp_training_iteration(iteration: int) -> None` controls curriculum locking only when the Stage 1 explicit-terrain marker is present; Stage 2 continues through the inherited path.

- [ ] **Step 1: Write pure allocation tests**

Create `tests/test_amp_terrain.py` covering:

```python
compute_anchor_count(2) == 1
compute_anchor_count(64) == 10
compute_anchor_count(4096) == 614

columns, anchors = assign_amp_columns(12)
columns.tolist() == [0, 0, 0, 1, 2, 2, 3, 3, 4, 4, 5, 5]
anchors.tolist() == [True, True, False, False, False, False, False, False, False, False, False, False]

columns, anchors = assign_amp_columns(8)
columns.tolist() == [0, 0, 1, 2, 2, 3, 3, 4]
anchors.tolist() == [True, False, False, False, False, False, False, False]
```

Assert there are exactly 6 named columns, every anchor maps to column 0, and every complete block of 10 course environments maps to `(0, 1, 2, 2, 3, 3, 4, 4, 5, 5)`; a partial block must equal that tuple's prefix. Explicitly prove that course environments may also occupy column 0 and group membership comes only from `is_amp_anchor`. Assert the resulting course ratio is `1:1:2:2:2:2`, invalid `num_envs` fails clearly, and `flat`, `hurdle`, `gap`, `ramp`, `bream`, `new stairs down` and `pit` are absent.

For `num_envs=4096`, assert 614 anchors, 3482 course environments, 698 course environments in columns 0/1 and 1312 total level-0 complete-flat environments during iterations 0--499 (`1312 / 4096 == 0.3203125`). Separately repeat `GO2_AMP_COURSE_COLUMN_CYCLE` directly to length 4096 (without calling the anchor allocator) and assert columns 0/1 contain exactly 820 environments in the no-anchor counterfactual.

- [ ] **Step 2: Write terrain-construction and curriculum source contracts**

Assert the explicit builder contains exactly the six required terrain names and stable class ids, both slope names use the existing slope primitive with opposite signs, and level-0 difficulty produces zero slope for columns 0/1. Assert there is no flat generator branch, flat column or new flat terrain class. Add source contracts showing the Go2 AMP environment owns `is_amp_anchor`, forces anchors to column 0 level 0, locks course levels before iteration 500, and delegates unlocked course ids to the existing MGDP curriculum method rather than reimplementing its maximum-level fallback.

- [ ] **Step 3: Run the terrain tests and verify they fail**

Run:

```bash
python -m unittest tests.test_amp_terrain -v
```

Expected: import failure for the missing helper module.

- [ ] **Step 4: Implement pure column and anchor allocation**

Create `go2_amp/terrain.py` with the four interfaces above. Keep it free of Isaac Gym imports so allocation tests run locally.

- [ ] **Step 5: Add the opt-in explicit terrain builder**

Set `Go2AmpStage1Cfg.terrain.num_cols=6` and an explicit-column config marker. In `Terrain.curiculum()`, use named construction only when that marker exists; leave the legacy cumulative-choice branch byte-for-byte equivalent for all other tasks. Implement named construction using the existing Stage 1 terrain primitives and difficulty formulas in `add_mix_terrain.py`; columns 0 and 1 use signed versions of `slope = difficulty * 0.4`, so both are complete flat at level 0. Do not add or expose a flat builder, and do not expose the later hurdle/gap/ramp/bream/new-stairs/pit builders through this Stage 1 marker.

- [ ] **Step 6: Implement Stage 1 environment allocation and curriculum gating**

Gate both overrides on the Stage 1 explicit-terrain marker; when absent, delegate directly to the inherited implementation so Stage 2 is unchanged. For Stage 1, override `_get_env_origins()` to install the computed column ids, anchor mask, level-zero starts, origins and terrain classes. Override `_update_terrain_curriculum(env_ids)` so anchors are reset to column 0 level 0; before unlock, course ids remain at level 0; after unlock, course ids are passed to `super()._update_terrain_curriculum(course_ids)`, preserving MGDP's random fallback above the maximum level. Never infer anchors from column id, terrain class, geometry or level.

- [ ] **Step 7: Run terrain and original-task tests**

Run:

```bash
python -m unittest tests.test_amp_terrain tests.test_task_config -v
```

Expected: all tests pass and original-task source assertions remain unchanged.

- [ ] **Step 8: Commit Task 2**

```bash
git add legged_gym/legged_gym/envs/go2_amp/terrain.py legged_gym/legged_gym/envs/go2_amp/config.py legged_gym/legged_gym/envs/go2_amp/env.py legged_gym/legged_gym/utils/terrain.py legged_gym/legged_gym/utils/new_terrains/add_mix_terrain.py tests/test_amp_terrain.py
git commit -m "feat: add anchored explicit terrain curriculum"
```

### Task 3: AMP schedule, serializable dual replay and fixed discriminator updates

**Files:**
- Modify: `legged_gym/rl/MGDP/amp/replay.py`
- Modify: `legged_gym/rl/MGDP/amp/session.py`
- Modify: `legged_gym/rl/MGDP/amp/discriminator.py`
- Modify: `tests/test_amp_reward.py`
- Modify: `tests/test_amp_checkpoint.py`

**Interfaces:**
- `effective_amp_coefficient(iteration: int, target: float, start: int = 100, end: int = 499) -> float` implements the exact spec ramp.
- `discriminator_updates_enabled(iteration: int, start: int = 100) -> bool` is true beginning at iteration 100.
- `curriculum_is_unlocked(iteration: int, unlock: int = 500) -> bool` is true beginning at iteration 500.
- `AMPReplayBuffer.state_dict() -> dict` and `load_state_dict(state: dict) -> None` round-trip storage, cursor, size, capacity and state width and reject incompatible shapes.
- A stratified Stage 1 `AMPSession` owns `anchor_replay` and `course_replay`; `reward(state, next_state, task_reward, terrain_class, iteration=None, is_anchor=None)` applies the scheduled coefficient and emits raw/contribution/logit tensors summarized by group, `record(state, next_state, is_anchor=None)` splits rows, and `update(iteration=None)` samples 256 rows from each pool for every configured update. Stage 2 retains its legacy behavior through the optional arguments.
- `AMPDiscriminator.gradient_penalty(expert_pair, coefficient: float) -> torch.Tensor` uses the configured coefficient.

- [ ] **Step 1: Write exact schedule-boundary tests**

Assert coefficients for target 0.01 at iterations 99, 100, 101, 200, 250, 300, 400, 498, 499 and 500. Pin the diagnostic values `200≈0.00250627`, `250≈0.00375940`, `300≈0.00501253`, `400≈0.00751880` and `499=0.01`; assert discriminator updates begin at 100 and curriculum unlock begins at 500. Repeat with target 0 to prove Scaffold-only remains zero while updates still enable, and with fallback target 0.0005 to prove the helper is coefficient-agnostic without making it the primary experiment.

- [ ] **Step 2: Write replay round-trip and stratification tests**

Extend `tests/test_amp_checkpoint.py` to assert each circular replay preserves content, cursor and size. Use distinguishable anchor/course tensors to prove a 512-policy batch contains exactly 256 from each pool, samples a nonempty undersized pool with replacement, and skips with an explicit metric when either pool is empty.

- [ ] **Step 3: Write fixed-update and gradient-penalty tests**

Assert `amp_updates_per_iter=1` changes discriminator parameters once, a configured value of 2 changes the actual update counter by two, zero/negative values fail, and the configured gradient-penalty coefficient reaches `AMPDiscriminator.gradient_penalty`.

- [ ] **Step 4: Run the AMP tests and verify they fail**

Run:

```bash
python -m unittest tests.test_amp_reward tests.test_amp_checkpoint -v
```

Expected: failures for missing schedule helpers, missing replay state and missing stratified session behavior.

- [ ] **Step 5: Implement the scheduled uniform reward and replay serialization**

Keep schedule helpers in `session.py` and independent of environment code. In the Stage 1 `AMPSession.reward(...)` path, compute `r_amp_raw = clamp(1 - 0.25 * (logit - 1)^2, min=0)`, apply `effective_amp_coefficient(iteration, amp_reward_coef)`, and use the same 1.0 gate for every terrain. Return the raw, contribution and rollout-logit data required by Task 4 without detaching the task reward from the runner's existing flow. Keep the Stage 2 legacy gate unchanged. Add strict replay shape/capacity validation before mutating an existing buffer during load.

- [ ] **Step 6: Implement Stage 1 stratified replay without changing Stage 2**

Activate dual pools only when `amp_stratified_replay=True`. Preserve the legacy single-replay path for Stage 2. Compute capacities from `amp_replay_rollouts * num_steps_per_env * group_env_count`; keep recording during iterations 0--99 and gate only reward coefficient and discriminator updates.

- [ ] **Step 7: Implement fixed repeated updates and metric means**

For every eligible iteration, draw a fresh stratified policy batch and expert batch for each configured update. Return discriminator-loss and gradient-penalty means; expert-logit and `policy_logit_update` mean/std pairs; pool sizes; anchor/course skipped-update counts and reasons; and the actual update counter.

- [ ] **Step 8: Run the AMP tests**

Run:

```bash
python -m unittest tests.test_amp_reward tests.test_amp_checkpoint -v
```

Expected: all tests pass; tests skip only when PyTorch is unavailable.

- [ ] **Step 9: Commit Task 3**

```bash
git add legged_gym/rl/MGDP/amp/replay.py legged_gym/rl/MGDP/amp/session.py legged_gym/rl/MGDP/amp/discriminator.py tests/test_amp_reward.py tests/test_amp_checkpoint.py
git commit -m "feat: stratify and schedule AMP updates"
```

### Task 4: Diagnostic aggregation, iteration propagation and compatible resume

**Files:**
- Modify: `legged_gym/rl/MGDP/runners/policy_runner.py`
- Modify: `tests/test_amp_rollout.py`
- Modify: `tests/test_amp_checkpoint.py`

**Interfaces:**
- Before each rollout, the runner calls `env.set_amp_training_iteration(it)` and supplies the same zero-based `it` to AMP reward/update methods.
- Stage 1 records `env.is_amp_anchor` with every transition; Stage 2 keeps its legacy record call.
- Rollout metrics aggregate all environment samples over all `num_steps_per_env` steps. Required tags are task/total reward means; `r_amp_raw` mean/std/zero fraction; AMP contribution mean/std; contribution-to-task absolute-mean ratio; rollout policy-logit mean/std; anchor/course policy-logit, raw-reward and contribution mean/std; replay sizes and skip counts; actual anchor count; level-0 course zero-slope count; complete-flat fraction; existing `feet_air_time` episode reward; and terrain-level mean. Update metrics include discriminator loss, gradient penalty, update policy-logit mean/std, expert-logit mean/std and actual update count.
- `policy_logit_rollout_mean/std` and `policy_logit_update_mean/std` remain distinct tags. No metric is interpreted by the runner as a pass/fail, stop, resume, coefficient-switch or next-run command.
- AMP checkpoint state contains a deterministic compatibility dictionary; `AMPSession.load_state_dict` validates it before loading discriminator, optimizer, normalizer or replay tensors.

- [ ] **Step 1: Replace the old last-step logging contract with distribution assertions**

In `tests/test_amp_rollout.py`, parse or exercise a small aggregation helper with three distinct steps and two distinguishable anchor/course groups. Assert exact task/total means; `r_amp_raw` mean, population std and zero fraction; AMP contribution mean/std; `mean(abs(contribution)) / max(mean(abs(task_reward)), 1e-8)`; and rollout policy-logit mean/std. Assert the same logit/raw/contribution summaries are independently correct for anchor and course samples. Assert TensorBoard keys contain distinct `policy_logit_rollout_mean/std` and `policy_logit_update_mean/std` tags and never the ambiguous shared `policy_logit` key.

Add a terrain-summary case with 4096 entries that asserts 614 anchors, 698 level-0 course zero-slope environments and complete-flat fraction `1312 / 4096`. Assert the existing `feet_air_time` episode reward remains visible and terrain-level mean is emitted. Add update-metric assertions for discriminator loss, gradient penalty, expert-logit mean/std, actual update count, both replay sizes and per-group skipped-update counts.

- [ ] **Step 2: Add ordering and iteration contracts**

Retain the existing assertion that AMP reward is applied before `process_env_step`. Add assertions that PPO update occurs before AMP update, the environment receives `it` before rollout, AMP reward and AMP update receive the same `it`, and the anchor mask is passed to Stage 1 reward summarization and replay recording. Add a source contract that the runner never branches on AMP/task metrics to stop training, resume, change `amp_reward_coef` or launch another run.

- [ ] **Step 3: Add checkpoint compatibility tests**

Construct a saved session, then change each compatibility field independently: target coefficient, schedule boundary, anchor fraction, update count, batch size, learning rate, replay rollouts, gradient penalty and expert groups. Assert every mismatch raises before any model parameter or replay cursor changes. Assert an identical state restores both pools and actual update count.

- [ ] **Step 4: Run runner/checkpoint tests and verify they fail**

Run:

```bash
python -m unittest tests.test_amp_rollout tests.test_amp_checkpoint -v
```

Expected: failures for last-step metrics, missing iteration propagation and incomplete compatibility state.

- [ ] **Step 5: Implement runner integration**

Accumulate sufficient statistics—not per-step averages—inside the 24-step loop: count, sum, squared sum and zero count for the required global and anchor/course distributions. Finalize mean/population-std/zero-fraction once after collection so uneven group sizes remain correctly weighted. Compute the contribution/task ratio from aggregated absolute sums. Read current column ids, terrain levels and `is_amp_anchor` only to emit counts/fractions; do not infer anchor identity from terrain fields. Keep AMP reward computation inside inference mode, PPO update first, and discriminator update afterward with gradients enabled. Pass anchor masks only for stratified Stage 1 sessions. Forward metrics to TensorBoard without automatic thresholds or control actions.

- [ ] **Step 6: Implement compatibility-first AMP restore**

Store the compatibility dictionary inside `amp_state`. Validate all fields before calling any nested `load_state_dict`; use exact equality for discrete/list fields and deterministic serialized float values from resolved config. Keep explicit `--amp_policy_only` behavior unchanged.

- [ ] **Step 7: Run runner/checkpoint tests**

Run:

```bash
python -m unittest tests.test_amp_rollout tests.test_amp_checkpoint -v
```

Expected: all tests pass; PyTorch-dependent cases may skip only when PyTorch is unavailable.

- [ ] **Step 8: Commit Task 4**

```bash
git add legged_gym/rl/MGDP/runners/policy_runner.py tests/test_amp_rollout.py tests/test_amp_checkpoint.py
git commit -m "feat: aggregate and restore task-priority AMP runs"
```

### Task 5: Documentation and full verification

**Files:**
- Modify: `README.md`
- Modify: existing tests only if full-suite integration exposes a contract mismatch

**Interfaces:**
- Documents separate from-zero commands for Scaffold-only and the primary AMP 0.01 pressure test using unique output directories, plus 0.0005 as a separately launched fallback only after a human decision.
- Documents same-run resume only, required `last.pt`/`wm_last.pt` pairing and rejection of old/incompatible AMP checkpoints.
- Documents that GPU smoke proves integration only and does not establish terrain performance.
- Documents the 20/100/200/250/500/1000 checkpoints, the human-only stop/admission checklist and the absence of automatic experiment orchestration.

- [ ] **Step 1: Update README commands and interpretation**

Add concrete Linux commands for:

```bash
python legged_gym/scripts/train_go2_amp_stage1.py --headless --num_envs 4096 --seed 1 --amp_reward_coef 0 --output_name outputs/go2_amp/stage1_task_priority_scaffold_seed1
python legged_gym/scripts/train_go2_amp_stage1.py --headless --num_envs 4096 --seed 1 --amp_reward_coef 0.01 --output_name outputs/go2_amp/stage1_task_priority_amp0p01_seed1
```

Also document, but do not automatically invoke, the fallback command using `--amp_reward_coef 0.0005` and a distinct `stage1_task_priority_amp0p0005_seed1` directory. Include the exact 20/100/200/250/500/1000/10000-iteration sequence, same-run resume syntax, effective 0.01 ramp values at 200/250/300/400/499, the fixed 0.7 m/s comparison, retained `feet_air_time` disclosure, and the fact that Stage 2 remains out of scope.

State explicitly that the stop gates and Stage 1 admission conditions are a researcher checklist: the training program only saves checkpoints and logs diagnostics. It must not stop, continue, switch to 0.0005, create a new run or enter Stage 2 based on metric values. The researcher reviews task behavior, AMP distributions and fixed videos with the collaborator before choosing the next command.

- [ ] **Step 2: Run the complete local test suite**

Run:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Expected: all locally runnable tests pass; only tests requiring unavailable PyTorch may skip. Do not include or modify the user-owned `tests/test_record_go2_amp_stage1.py` when resolving failures.

- [ ] **Step 3: Run syntax and whitespace verification**

Run:

```bash
python -m compileall -q legged_gym
git diff --check
```

Expected: both commands exit 0 with no diagnostics.

- [ ] **Step 4: Verify protected files and original tasks**

Run `git status --short` and confirm the two protected untracked files are still untracked and absent from the staged set. Re-run `tests.test_task_config` to confirm original registrations and configs remain unchanged.

- [ ] **Step 5: Record unavailable GPU verification without claiming performance**

If the current host lacks Linux/CUDA/Isaac Gym/Warp, state that the following remain for the training host: Stage 1 config startup; 20-iteration save/resume smoke; 100-iteration task-only checkpoint; 4096×200/250 paired diagnosis; 4096×500 full-0.01 check; and 4096×1000 early screen. Label every continue/stop decision as manual rather than an unexecuted code test.

- [ ] **Step 6: Commit Task 5**

```bash
git add README.md
git commit -m "docs: explain task-priority AMP training"
```

## Final Acceptance Checklist

- [ ] Stage 1 commands use identical ordinary/new forward and lateral ranges with `heading_command=True`.
- [ ] Gait-pattern rewards are zero; only capped `feet_air_time=0.5` remains as locomotion scaffold.
- [ ] Exactly six explicit original-MGDP Stage 1 columns are constructible; there is no independent flat generator, column or terrain class, and later complex terrains are absent.
- [ ] Anchors reuse column 0 at level 0 and remain identifiable only by `is_amp_anchor`; unlocked course environments retain MGDP maximum-level random fallback.
- [ ] With 4096 environments, allocation yields 614 anchors, 3482 course environments, 698 level-0 course zero-slope environments and 1312 complete-flat environments during iterations 0--499.
- [ ] Iterations 0--99, 100--499 and 500+ follow the exact reward/update/curriculum schedule.
- [ ] Policy replay is sampled anchor/course 1:1 and both pools survive checkpoint round-trip.
- [ ] PPO updates before the discriminator; the new discriminator affects only the next rollout.
- [ ] Rollout metrics include global and anchor/course AMP distributions, zero fraction, contribution/task ratio, flat allocation counts and separate rollout/update logit mean/std tags.
- [ ] Scaffold-only and AMP 0.01 primary runs are isolated; fallback 0.0005 uses a new directory, and incompatible resumes fail early.
- [ ] No code path automatically stops/continues training, decides admission, switches coefficient or launches the next experiment from logged metrics.
- [ ] Original MGDP tasks, Stage 2 and the two protected user files remain unchanged.
