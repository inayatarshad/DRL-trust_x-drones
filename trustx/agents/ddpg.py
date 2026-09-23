"""Deep Deterministic Policy Gradient (Lillicrap et al., 2016) baseline."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from trustx.agents.base import Agent
from trustx.agents.networks import Actor, Critic, soft_update


class DDPGAgent(Agent):
    name = "ddpg"

    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256), actor_lr=1e-4, critic_lr=1e-3,
                 gamma=0.99, tau=0.005, exploration_noise=0.1, device="cpu"):
        super().__init__(obs_dim, act_dim, device)
        self.hidden = tuple(hidden)
        self.actor = Actor(obs_dim, act_dim, hidden).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(obs_dim, act_dim, hidden).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma, self.tau = gamma, tau
        self.exploration_noise = exploration_noise

    def _explore(self, action: np.ndarray) -> np.ndarray:
        return action + np.random.normal(0.0, self.exploration_noise, size=action.shape)

    def update(self, buffer, batch_size: int = 256) -> dict[str, float]:
        obs, act, rew, next_obs, done = buffer.sample(batch_size, self.device)
        with torch.no_grad():
            y = rew + self.gamma * (1.0 - done) * self.critic_target(next_obs, self.actor_target(next_obs))
        q = self.critic(obs, act)
        critic_loss = F.mse_loss(q, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = -self.critic(obs, self.actor(obs)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item(), "q_mean": q.mean().item()}

    @torch.no_grad()
    def q_values(self, obs, action):
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        return self.critic(o, a).cpu().numpy()[None]

    def state_dict(self) -> dict:
        return {k: getattr(self, k).state_dict() for k in ("actor", "actor_target", "critic", "critic_target")}

    def load_state_dict(self, state: dict) -> None:
        for k, v in state.items():
            getattr(self, k).load_state_dict(v)
