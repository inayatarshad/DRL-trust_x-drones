"""Proximal Policy Optimization (Schulman et al., 2017) baseline.

Gaussian policy whose mean is a tanh-bounded MLP. The mean network is
exposed as ``actor`` so the explainers treat PPO like the deterministic
agents at evaluation time.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from trustx.agents.base import Agent
from trustx.agents.networks import Actor, mlp


class RolloutBuffer:
    def __init__(self):
        self.obs, self.act, self.logp, self.rew, self.done, self.val = [], [], [], [], [], []

    def add(self, obs, act, logp, rew, done, val):
        self.obs.append(obs)
        self.act.append(act)
        self.logp.append(logp)
        self.rew.append(rew)
        self.done.append(done)
        self.val.append(val)

    def __len__(self):
        return len(self.obs)

    def clear(self):
        self.__init__()


class PPOAgent(Agent):
    name = "ppo"
    off_policy = False

    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256), lr=3e-4, gamma=0.99, lam=0.95,
                 clip=0.2, epochs=10, minibatch=256, entropy_coef=0.0, value_coef=0.5,
                 init_log_std=-0.5, rollout_steps=2048, device="cpu"):
        super().__init__(obs_dim, act_dim, device)
        self.actor = Actor(obs_dim, act_dim, hidden).to(self.device)
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std, device=self.device))
        self.value = mlp(obs_dim, 1, hidden, act=nn.Tanh).to(self.device)
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.value.parameters()) + [self.log_std], lr=lr)
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.minibatch = epochs, minibatch
        self.entropy_coef, self.value_coef = entropy_coef, value_coef
        self.rollout_steps = rollout_steps
        self.rollout = RolloutBuffer()

    def _dist(self, obs: torch.Tensor):
        return torch.distributions.Normal(self.actor(obs), self.log_std.exp())

    @torch.no_grad()
    def sample(self, obs: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Stochastic action for training, with its log-prob and value estimate."""
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        dist = self._dist(o)
        a = dist.sample()
        return a.cpu().numpy(), float(dist.log_prob(a).sum()), float(self.value(o).squeeze(-1))

    def _advantages(self, last_value: float):
        rew = np.asarray(self.rollout.rew, dtype=np.float32)
        done = np.asarray(self.rollout.done, dtype=np.float32)
        val = np.append(np.asarray(self.rollout.val, dtype=np.float32), last_value)
        adv = np.zeros_like(rew)
        gae = 0.0
        for t in reversed(range(len(rew))):
            nonterminal = 1.0 - done[t]
            delta = rew[t] + self.gamma * val[t + 1] * nonterminal - val[t]
            gae = delta + self.gamma * self.lam * nonterminal * gae
            adv[t] = gae
        return adv, adv + val[:-1]

    def update(self, last_obs: np.ndarray, last_done: bool) -> dict[str, float]:
        with torch.no_grad():
            last_val = 0.0 if last_done else float(
                self.value(torch.as_tensor(last_obs, dtype=torch.float32, device=self.device)).squeeze(-1))
        adv, ret = self._advantages(last_val)
        to_t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)  # noqa: E731
        obs, act, old_logp = to_t(self.rollout.obs), to_t(self.rollout.act), to_t(self.rollout.logp)
        adv_t, ret_t = to_t(adv), to_t(ret)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n = len(obs)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        count = 0
        for _ in range(self.epochs):
            for idx in torch.randperm(n, device=self.device).split(self.minibatch):
                dist = self._dist(obs[idx])
                logp = dist.log_prob(act[idx]).sum(-1)
                ratio = (logp - old_logp[idx]).exp()
                surr = torch.min(ratio * adv_t[idx], ratio.clamp(1 - self.clip, 1 + self.clip) * adv_t[idx])
                policy_loss = -surr.mean()
                value_loss = (self.value(obs[idx]).squeeze(-1) - ret_t[idx]).pow(2).mean()
                entropy = dist.entropy().sum(-1).mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.opt.param_groups[0]["params"], 0.5)
                self.opt.step()
                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.item()
                count += 1
        self.rollout.clear()
        return {k: v / max(count, 1) for k, v in stats.items()}

    def state_dict(self) -> dict:
        return {"actor": self.actor.state_dict(), "value": self.value.state_dict(),
                "log_std": self.log_std.detach().cpu()}

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.value.load_state_dict(state["value"])
        with torch.no_grad():
            self.log_std.copy_(state["log_std"])
