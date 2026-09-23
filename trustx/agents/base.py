"""Common agent interface used by training, evaluation and the explainers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from torch import nn


class Agent(ABC):
    name: str = "agent"
    #: deterministic policy network, obs -> action in [-1, 1]
    actor: nn.Module
    #: off-policy agents learn from a replay buffer; PPO from rollouts
    off_policy: bool = True

    def __init__(self, obs_dim: int, act_dim: int, device: str = "cpu"):
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.device = torch.device(device)

    @torch.no_grad()
    def act(self, obs: np.ndarray, explore: bool = False) -> np.ndarray:
        """Return an action for a single observation (or a batch)."""
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        a = self.actor(x).cpu().numpy()
        if explore:
            a = self._explore(a)
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    def _explore(self, action: np.ndarray) -> np.ndarray:
        return action

    def policy_fn(self):
        """Batched numpy policy, handy for surrogate fitting."""
        return lambda obs: self.act(np.asarray(obs, dtype=np.float32))

    def q_values(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray | None:
        """Per-critic Q estimates, shape (n_critics, batch). None if unavailable."""
        return None

    @abstractmethod
    def update(self, *args, **kwargs) -> dict[str, float]:
        ...

    @abstractmethod
    def state_dict(self) -> dict:
        ...

    @abstractmethod
    def load_state_dict(self, state: dict) -> None:
        ...

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"name": self.name, "obs_dim": self.obs_dim, "act_dim": self.act_dim,
                    "hidden": list(getattr(self, "hidden", (256, 256))), "state": self.state_dict()}, path)

    def load(self, path: str | Path) -> "Agent":
        blob = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(blob["state"])
        return self
