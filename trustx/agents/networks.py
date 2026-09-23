"""Neural network building blocks shared by the agents."""

from __future__ import annotations

import torch
from torch import nn


def mlp(in_dim: int, out_dim: int, hidden: tuple[int, ...] = (256, 256), act=nn.ReLU) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), act()]
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Deterministic policy mu_theta(s) with tanh-bounded output."""

    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        self.net = mlp(obs_dim, act_dim, hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))


class Critic(nn.Module):
    """State-action value function Q_phi(s, a)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        self.net = mlp(obs_dim + act_dim, 1, hidden)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.mul_(1.0 - tau).add_(sp, alpha=tau)
