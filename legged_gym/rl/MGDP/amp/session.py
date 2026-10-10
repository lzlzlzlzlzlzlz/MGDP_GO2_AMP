"""AMP reward, normalization, discriminator updates and checkpoint state."""

import math
import torch

from .discriminator import AMPDiscriminator
from .replay import AMPReplayBuffer


def effective_amp_coefficient(iteration: int, target: float,
                              start: int = 100, end: int = 499) -> float:
    if end <= start:
        raise ValueError("AMP ramp end must be greater than its start")
    if iteration <= start:
        return 0.0
    if iteration < end:
        return target * (iteration - start) / (end - start)
    return target


def discriminator_updates_enabled(iteration: int, start: int = 100) -> bool:
    return iteration >= start


def curriculum_is_unlocked(iteration: int, unlock: int = 500) -> bool:
    return iteration >= unlock


def combine_reward(task_reward, disc_output, terrain_class, stage, coef,
                   easy_gate=1.0, hard_gate=0.25):
    if stage not in (1, 2):
        raise ValueError("AMP stage must be 1 or 2")
    if coef == 0:
        return task_reward
    if stage == 1:
        gate = torch.full_like(task_reward, easy_gate)
    else:
        easy = terrain_class <= 2
        gate = torch.where(easy, torch.full_like(task_reward, easy_gate),
                           torch.full_like(task_reward, hard_gate))
    style = coef * torch.clamp(1 - 0.25 * (disc_output - 1).square(), min=0)
    return task_reward + gate * style


class RunningNormalizer:
    def __init__(self, width, device):
        self.mean = torch.zeros(width, device=device)
        self.var = torch.ones(width, device=device)
        self.count = 1e-4

    def update(self, batch):
        batch = batch.detach()
        count = batch.shape[0]
        if count == 0:
            return
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=False)
        total = self.count + count
        delta = batch_mean - self.mean
        self.mean += delta * count / total
        self.var = (self.var * self.count + batch_var * count + delta.square() * self.count * count / total) / total
        self.count = total

    def normalize(self, batch):
        return (batch - self.mean) / torch.sqrt(self.var + 1e-5)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state):
        self.mean.copy_(state["mean"])
        self.var.copy_(state["var"])
        self.count = state["count"]


class AMPSession:
    def __init__(self, dataset, stage, device, config):
        if stage not in (1, 2):
            raise ValueError("AMP stage must be 1 or 2")
        for key, default, allow_zero in (
            ("amp_reward_coef", None, True),
            ("amp_easy_gate", 1.0, False),
            ("amp_hard_gate", 0.25, False),
            ("amp_learning_rate", 1e-4, False),
        ):
            value = config[key] if default is None else config.get(key, default)
            if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
                qualifier = "nonnegative" if allow_zero else "positive"
                raise ValueError(f"{key} must be finite and {qualifier}")
        for key, default in (("amp_batch_size", 512), ("amp_replay_capacity", 100000),
                             ("amp_replay_rollouts", 2), ("num_steps_per_env", 24),
                             ("amp_updates_per_iter", 1)):
            if config.get(key, default) <= 0:
                raise ValueError(f"{key} must be positive")
        updates_per_iter = config.get("amp_updates_per_iter", 1)
        if int(updates_per_iter) != updates_per_iter:
            raise ValueError("amp_updates_per_iter must be a positive integer")
        if config.get("amp_stratified_replay", False) and config.get("amp_batch_size", 512) % 2:
            raise ValueError("amp_batch_size must be even for stratified replay")
        self.dataset = dataset
        self.stage = stage
        self.device = device
        self.config = config
        self.discriminator = AMPDiscriminator().to(device)
        self.optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=config.get("amp_learning_rate", 1e-4))
        self.normalizer = RunningNormalizer(30, device)
        self.stratified = bool(config.get("amp_stratified_replay", False))
        self.replay = None
        self.anchor_replay = None
        self.course_replay = None
        if self.stratified:
            if stage != 1:
                raise ValueError("stratified replay is only supported for AMP Stage 1")
        else:
            self.replay = AMPReplayBuffer(config.get("amp_replay_capacity", 100000), 30, device)
        self.iteration = 0
        self.policy_iteration = 0

    def _pair(self, state, next_state):
        return torch.cat((self.normalizer.normalize(state),
                          self.normalizer.normalize(next_state)), dim=-1)

    @staticmethod
    def _distribution_sums(prefix, values):
        values = values.detach().reshape(-1)
        return {
            f"{prefix}_count": int(values.numel()),
            f"{prefix}_sum": values.sum(),
            f"{prefix}_sq_sum": values.square().sum(),
            f"{prefix}_abs_sum": values.abs().sum(),
            f"{prefix}_zero_count": (values == 0).sum(),
        }

    def reward(self, state, next_state, task_reward, terrain_class,
               iteration=None, is_anchor=None):
        with torch.no_grad():
            logits = self.discriminator(self._pair(state, next_state))
            raw = torch.clamp(1 - 0.25 * (logits - 1).square(), min=0)
            if self.stratified:
                coefficient = self.config["amp_reward_coef"]
                if iteration is not None:
                    self.policy_iteration = int(iteration)
                    coefficient = effective_amp_coefficient(
                        self.policy_iteration,
                        coefficient,
                        self.config.get("amp_discriminator_start_iteration", 100),
                        self.config.get("amp_ramp_end_iteration", 499),
                    )
                contribution = coefficient * raw
                total = task_reward + contribution
            else:
                total = combine_reward(
                    task_reward,
                    logits,
                    terrain_class,
                    self.stage,
                    self.config["amp_reward_coef"],
                    self.config.get("amp_easy_gate", 1.0),
                    self.config.get("amp_hard_gate", 0.25),
                )
                contribution = total - task_reward

        if not self.stratified:
            return total, {
                "task_reward": task_reward.mean().item(),
                "style_reward": contribution.mean().item(),
                "total_reward": total.mean().item(),
                "policy_logit": logits.mean().item(),
            }

        metrics = {}
        metrics.update(self._distribution_sums("task_reward", task_reward))
        metrics.update(self._distribution_sums("total_reward", total))
        metrics.update(self._distribution_sums("r_amp_raw", raw))
        metrics.update(self._distribution_sums("amp_reward_contribution", contribution))
        metrics.update(self._distribution_sums("policy_logit_rollout", logits))
        if is_anchor is not None:
            if is_anchor.ndim != 1 or is_anchor.shape[0] != task_reward.shape[0]:
                raise ValueError("is_anchor must contain one boolean per AMP transition")
            mask = is_anchor.to(device=task_reward.device, dtype=torch.bool)
            for group, group_mask in (("anchor", mask), ("course", ~mask)):
                metrics.update(self._distribution_sums(
                    f"{group}_r_amp_raw", raw[group_mask]
                ))
                metrics.update(self._distribution_sums(
                    f"{group}_amp_reward_contribution", contribution[group_mask]
                ))
                metrics.update(self._distribution_sums(
                    f"{group}_policy_logit_rollout", logits[group_mask]
                ))
        return total, metrics

    def _initialize_stratified_replays(self, is_anchor):
        anchor_count = int(is_anchor.sum().item())
        course_count = int(is_anchor.numel() - anchor_count)
        if anchor_count + course_count == 0:
            raise ValueError("cannot initialize stratified replay from an empty rollout")
        multiplier = (
            self.config.get("amp_replay_rollouts", 2)
            * self.config.get("num_steps_per_env", 24)
        )
        for group, count in (("anchor", anchor_count), ("course", course_count)):
            if count == 0:
                continue
            capacity = multiplier * count
            replay = getattr(self, f"{group}_replay")
            if replay is None:
                setattr(self, f"{group}_replay", AMPReplayBuffer(capacity, 30, self.device))
            elif replay.capacity != capacity:
                raise ValueError(f"{group} environment count changed within an AMP run")

    def record(self, state, next_state, is_anchor=None):
        if not self.stratified:
            self.replay.insert(state, next_state)
            return
        if is_anchor is None:
            raise ValueError("Stage 1 stratified replay requires is_anchor")
        if is_anchor.ndim != 1 or is_anchor.shape[0] != state.shape[0]:
            raise ValueError("is_anchor must contain one boolean per AMP transition")
        mask = is_anchor.to(device=state.device, dtype=torch.bool)
        if self.anchor_replay is None or self.course_replay is None:
            self._initialize_stratified_replays(mask)
        if self.anchor_replay is not None:
            self.anchor_replay.insert(state[mask], next_state[mask])
        if self.course_replay is not None:
            self.course_replay.insert(state[~mask], next_state[~mask])

    def _sample_policy_batch(self):
        batch_size = self.config.get("amp_batch_size", 512)
        if not self.stratified:
            return self.replay.sample(batch_size)
        half_batch = batch_size // 2
        anchor_state, anchor_next = self.anchor_replay.sample(half_batch)
        course_state, course_next = self.course_replay.sample(half_batch)
        return (
            torch.cat((anchor_state, course_state), dim=0),
            torch.cat((anchor_next, course_next), dim=0),
        )

    def _replay_metrics(self):
        if not self.stratified:
            return {"replay_size": self.replay.size}
        return {
            "anchor_replay_size": 0 if self.anchor_replay is None else self.anchor_replay.size,
            "course_replay_size": 0 if self.course_replay is None else self.course_replay.size,
        }

    def update(self, iteration=None):
        updates_per_iter = int(self.config.get("amp_updates_per_iter", 1))
        metrics = self._replay_metrics()
        if self.stratified and iteration is not None:
            self.policy_iteration = int(iteration)
            if not discriminator_updates_enabled(
                    self.policy_iteration,
                    self.config.get("amp_discriminator_start_iteration", 100)):
                metrics.update({
                    "amp_updates": 0,
                    "amp_update_count": self.iteration,
                    "anchor_replay_empty": 0,
                    "course_replay_empty": 0,
                    "anchor_skipped_updates": 0,
                    "course_skipped_updates": 0,
                })
                return metrics

        if self.stratified:
            anchor_empty = self.anchor_replay is None or self.anchor_replay.size == 0
            course_empty = self.course_replay is None or self.course_replay.size == 0
            if anchor_empty or course_empty:
                metrics.update({
                    "amp_updates": 0,
                    "amp_update_count": self.iteration,
                    "anchor_replay_empty": int(anchor_empty),
                    "course_replay_empty": int(course_empty),
                    "anchor_skipped_updates": updates_per_iter if anchor_empty else 0,
                    "course_skipped_updates": updates_per_iter if course_empty else 0,
                })
                return metrics
        elif self.replay.size == 0:
            return {}

        losses = []
        penalties = []
        all_policy_logits = []
        all_expert_logits = []
        batch_size = self.config.get("amp_batch_size", 512)
        penalty_coefficient = self.config.get("amp_gradient_penalty_coef", 10.0)
        for _ in range(updates_per_iter):
            policy_state, policy_next = self._sample_policy_batch()
            expert_state, expert_next = self.dataset.sample(batch_size, self.device)
            self.normalizer.update(torch.cat((policy_state, policy_next, expert_state, expert_next), dim=0))
            policy_pair = self._pair(policy_state, policy_next).detach()
            expert_pair = self._pair(expert_state, expert_next).detach()
            policy_logits = self.discriminator(policy_pair)
            expert_logits = self.discriminator(expert_pair)
            loss = 0.5 * ((policy_logits + 1).square().mean() +
                          (expert_logits - 1).square().mean())
            penalty = self.discriminator.gradient_penalty(
                expert_pair, coefficient=penalty_coefficient
            )
            self.optimizer.zero_grad()
            (loss + penalty).backward()
            self.optimizer.step()
            self.iteration += 1
            losses.append(loss.detach())
            penalties.append(penalty.detach())
            all_policy_logits.append(policy_logits.detach())
            all_expert_logits.append(expert_logits.detach())

        policy_values = torch.cat(all_policy_logits)
        expert_values = torch.cat(all_expert_logits)
        if not self.stratified:
            return {
                "discriminator_loss": torch.stack(losses).mean().item(),
                "gradient_penalty": torch.stack(penalties).mean().item(),
                "expert_logit": expert_values.mean().item(),
                "policy_logit": policy_values.mean().item(),
            }
        metrics.update({
            "discriminator_loss": torch.stack(losses).mean().item(),
            "gradient_penalty": torch.stack(penalties).mean().item(),
            "expert_logit_mean": expert_values.mean().item(),
            "expert_logit_std": expert_values.std(unbiased=False).item(),
            "policy_logit_update_mean": policy_values.mean().item(),
            "policy_logit_update_std": policy_values.std(unbiased=False).item(),
            "amp_updates": updates_per_iter,
            "amp_update_count": self.iteration,
            "anchor_replay_empty": 0,
            "course_replay_empty": 0,
            "anchor_skipped_updates": 0,
            "course_skipped_updates": 0,
        })
        return metrics

    def _restore_replay(self, replay_state, current):
        if replay_state is None:
            return None
        if current is None:
            current = AMPReplayBuffer(
                int(replay_state["capacity"]),
                int(replay_state["state_dim"]),
                self.device,
            )
        current.load_state_dict(replay_state)
        return current

    def _compatibility(self):
        groups = self.config.get("amp_groups", {})
        weights = self.config.get("amp_group_weights", {})
        return {
            "stage": int(self.stage),
            "amp_reward_coef": float(self.config["amp_reward_coef"]),
            "amp_discriminator_start_iteration": int(
                self.config.get("amp_discriminator_start_iteration", 100)
            ),
            "amp_ramp_end_iteration": int(
                self.config.get("amp_ramp_end_iteration", 499)
            ),
            "amp_curriculum_unlock_iteration": int(
                self.config.get("amp_curriculum_unlock_iteration", 500)
            ),
            "amp_anchor_fraction": float(
                self.config.get("amp_anchor_fraction", 0.15)
            ),
            "amp_updates_per_iter": int(
                self.config.get("amp_updates_per_iter", 1)
            ),
            "amp_batch_size": int(self.config.get("amp_batch_size", 512)),
            "amp_learning_rate": float(
                self.config.get("amp_learning_rate", 1e-4)
            ),
            "amp_replay_rollouts": int(
                self.config.get("amp_replay_rollouts", 2)
            ),
            "amp_gradient_penalty_coef": float(
                self.config.get("amp_gradient_penalty_coef", 10.0)
            ),
            "num_steps_per_env": int(self.config.get("num_steps_per_env", 24)),
            "amp_num_envs": (
                None if self.config.get("amp_num_envs") is None
                else int(self.config["amp_num_envs"])
            ),
            "amp_stratified_replay": bool(self.stratified),
            "amp_groups": tuple(
                (str(name), tuple(str(item) for item in groups[name]))
                for name in sorted(groups)
            ),
            "amp_group_weights": tuple(
                (str(name), float(weights[name])) for name in sorted(weights)
            ),
        }

    def state_dict(self):
        state = {"discriminator": self.discriminator.state_dict(),
                 "optimizer": self.optimizer.state_dict(),
                 "normalizer": self.normalizer.state_dict(),
                 "iteration": self.iteration}
        if self.stratified:
            state["policy_iteration"] = self.policy_iteration
            state["compatibility"] = self._compatibility()
            state["anchor_replay"] = (
                None if self.anchor_replay is None else self.anchor_replay.state_dict()
            )
            state["course_replay"] = (
                None if self.course_replay is None else self.course_replay.state_dict()
            )
        return state

    def load_state_dict(self, state):
        if self.stratified:
            saved_compatibility = state.get("compatibility")
            expected_compatibility = self._compatibility()
            if saved_compatibility is None:
                raise ValueError(
                    "incompatible AMP checkpoint: compatibility metadata missing"
                )
            for field, expected in expected_compatibility.items():
                actual = saved_compatibility.get(field)
                if actual != expected:
                    raise ValueError(
                        f"incompatible AMP checkpoint field {field}: "
                        f"checkpoint {actual!r}, current {expected!r}"
                    )
        self.discriminator.load_state_dict(state["discriminator"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.normalizer.load_state_dict(state["normalizer"])
        if self.stratified:
            self.anchor_replay = self._restore_replay(
                state.get("anchor_replay"), self.anchor_replay
            )
            self.course_replay = self._restore_replay(
                state.get("course_replay"), self.course_replay
            )
        self.iteration = int(state["iteration"])
        self.policy_iteration = int(state.get("policy_iteration", 0))


def restore_amp_checkpoint(session, checkpoint, policy_only=False):
    if policy_only:
        return
    if "amp_state" not in checkpoint:
        raise ValueError("AMP checkpoint state missing; use --amp_policy_only for policy warm start")
    session.load_state_dict(checkpoint["amp_state"])
