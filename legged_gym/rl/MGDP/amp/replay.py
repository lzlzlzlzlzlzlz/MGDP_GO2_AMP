"""Fixed-size replay for policy motion transitions."""

import torch


class AMPReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, device: str):
        if capacity <= 0 or state_dim <= 0:
            raise ValueError("positive capacity and state_dim required")
        self.capacity = capacity
        self.states = torch.empty(capacity, state_dim, device=device)
        self.next_states = torch.empty_like(self.states)
        self.cursor = 0
        self.size = 0

    def insert(self, state, next_state):
        if state.shape != next_state.shape or state.ndim != 2 or state.shape[1] != self.states.shape[1]:
            raise ValueError("invalid AMP transition dimensions")
        count = state.shape[0]
        if count >= self.capacity:
            state, next_state = state[-self.capacity:], next_state[-self.capacity:]
            count = self.capacity
        positions = (torch.arange(count, device=self.states.device) + self.cursor) % self.capacity
        self.states[positions] = state.detach()
        self.next_states[positions] = next_state.detach()
        self.cursor = (self.cursor + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(self, batch_size):
        if self.size == 0:
            raise ValueError("cannot sample empty AMP replay")
        indices = torch.randint(self.size, (batch_size,), device=self.states.device)
        return self.states[indices], self.next_states[indices]
