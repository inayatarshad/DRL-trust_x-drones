"""Uniform experience replay buffer D."""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int = 1_000_000):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, act, rew, next_obs, done) -> None:
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i] = obs, act, rew
        self.next_obs[i], self.done[i] = next_obs, float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device | str = "cpu"):
        idx = np.random.randint(0, self.size, size=batch_size)
        as_t = lambda x: torch.as_tensor(x[idx], device=device)  # noqa: E731
        return as_t(self.obs), as_t(self.act), as_t(self.rew), as_t(self.next_obs), as_t(self.done)

    def __len__(self) -> int:
        return self.size
