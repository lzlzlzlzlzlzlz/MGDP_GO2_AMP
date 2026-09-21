"""AMP reward, normalization, discriminator updates and checkpoint state."""

import math
import torch

from .discriminator import AMPDiscriminator
from .replay import AMPReplayBuffer


def combine_reward(task_reward, disc_output, terrain_class, stage, coef,
                   easy_gate=1.0, hard_gate=0.25):
    if stage not in (1, 2):
        raise ValueError("AMP stage must be 1 or 2")
    if coef == 0:
        return task_reward
    threshold = 4 if stage == 1 else 2
    easy = terrain_class <= threshold
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
        for key, default in (("amp_batch_size", 512), ("amp_replay_capacity", 100000)):
            if config.get(key, default) <= 0:
                raise ValueError(f"{key} must be positive")
        self.dataset = dataset
        self.stage = stage
        self.device = device
        self.config = config
        self.discriminator = AMPDiscriminator().to(device)
        self.optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=config.get("amp_learning_rate", 1e-4))
        self.normalizer = RunningNormalizer(30, device)
        self.replay = AMPReplayBuffer(config.get("amp_replay_capacity", 100000), 30, device)
        self.iteration = 0

    def _pair(self, state, next_state):
        return torch.cat((self.normalizer.normalize(state),
                          self.normalizer.normalize(next_state)), dim=-1)

    def reward(self, state, next_state, task_reward, terrain_class):
        with torch.no_grad():
            logits = self.discriminator(self._pair(state, next_state))
            total = combine_reward(task_reward, logits, terrain_class, self.stage,
                                   self.config["amp_reward_coef"],
                                   self.config.get("amp_easy_gate", 1.0),
                                   self.config.get("amp_hard_gate", 0.25))
        return total, {"task_reward": task_reward.mean().item(),
                       "style_reward": (total - task_reward).mean().item(),
                       "total_reward": total.mean().item(),
                       "policy_logit": logits.mean().item()}

    def record(self, state, next_state):
        self.replay.insert(state, next_state)

    def update(self):
        if self.replay.size == 0:
            return {}
        batch_size = self.config.get("amp_batch_size", 512)
        policy_state, policy_next = self.replay.sample(batch_size)
        expert_state, expert_next = self.dataset.sample(batch_size, self.device)
        self.normalizer.update(torch.cat((policy_state, policy_next, expert_state, expert_next), dim=0))
        policy_pair = self._pair(policy_state, policy_next).detach()
        expert_pair = self._pair(expert_state, expert_next).detach()
        policy_logits = self.discriminator(policy_pair)
        expert_logits = self.discriminator(expert_pair)
        loss = 0.5 * ((policy_logits + 1).square().mean() +
                      (expert_logits - 1).square().mean())
        penalty = self.discriminator.gradient_penalty(expert_pair)
        self.optimizer.zero_grad()
        (loss + penalty).backward()
        self.optimizer.step()
        self.iteration += 1
        return {"discriminator_loss": loss.item(), "gradient_penalty": penalty.item(),
                "expert_logit": expert_logits.mean().item(),
                "policy_logit": policy_logits.mean().item()}

    def state_dict(self):
        return {"discriminator": self.discriminator.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "normalizer": self.normalizer.state_dict(),
                "iteration": self.iteration}

    def load_state_dict(self, state):
        self.discriminator.load_state_dict(state["discriminator"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.normalizer.load_state_dict(state["normalizer"])
        self.iteration = state["iteration"]


def restore_amp_checkpoint(session, checkpoint, policy_only=False):
    if policy_only:
        return
    if "amp_state" not in checkpoint:
        raise ValueError("AMP checkpoint state missing; use --amp_policy_only for policy warm start")
    session.load_state_dict(checkpoint["amp_state"])
