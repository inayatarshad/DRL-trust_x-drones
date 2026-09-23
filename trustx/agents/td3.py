"""Twin Delayed DDPG (Fujimoto et al., 2018) - Section IV-A, Eqs. (3)-(7)."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from trustx.agents.base import Agent
from trustx.agents.networks import Actor, Critic, soft_update


class TD3Agent(Agent):
    name = "td3"

    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256), actor_lr=3e-4, critic_lr=3e-4,
                 gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_delay=2,
                 exploration_noise=0.1, device="cpu"):
        super().__init__(obs_dim, act_dim, device)
        self.actor = Actor(obs_dim, act_dim, hidden).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic1 = Critic(obs_dim, act_dim, hidden).to(self.device)
        self.critic2 = Critic(obs_dim, act_dim, hidden).to(self.device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=critic_lr)
        self.gamma, self.tau = gamma, tau
        self.policy_noise, self.noise_clip = policy_noise, noise_clip
        self.policy_delay = policy_delay
        self.exploration_noise = exploration_noise
        self.total_updates = 0

    def _explore(self, action: np.ndarray) -> np.ndarray:
        return action + np.random.normal(0.0, self.exploration_noise, size=action.shape)

    def update(self, buffer, batch_size: int = 256) -> dict[str, float]:
        obs, act, rew, next_obs, done = buffer.sample(batch_size, self.device)
        with torch.no_grad():
            # Eqs. (6)-(7): target policy smoothing
            eps = (torch.randn_like(act) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_act = (self.actor_target(next_obs) + eps).clamp(-1.0, 1.0)
            # Eq. (5): clipped double-Q target
            q_next = torch.min(self.critic1_target(next_obs, next_act),
                               self.critic2_target(next_obs, next_act))
            y = rew + self.gamma * (1.0 - done) * q_next
        # Eq. (4): Bellman error for both critics
        q1, q2 = self.critic1(obs, act), self.critic2(obs, act)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        info = {"critic_loss": float(critic_loss), "q_mean": float(q1.mean())}
        self.total_updates += 1
        if self.total_updates % self.policy_delay == 0:
            # Eq. (3): deterministic policy gradient through Q_phi1
            actor_loss = -self.critic1(obs, self.actor(obs)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic1_target, self.critic1, self.tau)
            soft_update(self.critic2_target, self.critic2, self.tau)
            info["actor_loss"] = float(actor_loss)
        return info

    @torch.no_grad()
    def q_values(self, obs, action):
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        return torch.stack([self.critic1(o, a), self.critic2(o, a)]).cpu().numpy()

    def state_dict(self) -> dict:
        return {k: getattr(self, k).state_dict() for k in
                ("actor", "actor_target", "critic1", "critic2", "critic1_target", "critic2_target")}

    def load_state_dict(self, state: dict) -> None:
        for k, v in state.items():
            getattr(self, k).load_state_dict(v)
