"""Training loops for off-policy (TD3/DDPG) and on-policy (PPO) agents."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from trustx.agents import Agent, PPOAgent, ReplayBuffer, build_agent
from trustx.envs import make_env
from trustx.utils.logger import CSVLogger
from trustx.utils.seeding import set_global_seed

LOG_FIELDS = ["episode", "steps", "return", "length", "success", "crash", "wall_time"]


def train(cfg: dict, out_dir: str | Path) -> Agent:
    """Train an agent as described by ``cfg`` and write logs/checkpoints to ``out_dir``."""
    out_dir = Path(out_dir)
    seed = int(cfg.get("seed", 0))
    set_global_seed(seed)
    env = make_env(cfg.get("env"))
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    agent = build_agent(cfg["algo"], obs_dim, act_dim, **cfg.get("agent", {}))
    tcfg = dict(cfg.get("train", {}), _out_dir=str(out_dir))
    with CSVLogger(out_dir / "train_log.csv", LOG_FIELDS, echo_every=tcfg.get("log_every", 10)) as log:
        if isinstance(agent, PPOAgent):
            _train_on_policy(agent, env, tcfg, seed, log)
        else:
            _train_off_policy(agent, env, tcfg, seed, log)
    agent.save(out_dir / "model.pt")
    return agent


def _maybe_checkpoint(agent: Agent, tcfg: dict, ep: int) -> None:
    every, out = int(tcfg.get("checkpoint_every", 0)), tcfg.get("_out_dir")
    if every and out and (ep + 1) % every == 0:
        agent.save(Path(out) / f"model_ep{ep + 1}.pt")


def _train_off_policy(agent: Agent, env, tcfg: dict, seed: int, log: CSVLogger) -> None:
    episodes = int(tcfg.get("episodes", 500))
    warmup = int(tcfg.get("warmup_steps", 5000))
    batch = int(tcfg.get("batch_size", 256))
    buffer = ReplayBuffer(agent.obs_dim, agent.act_dim, int(tcfg.get("buffer_size", 1_000_000)))
    steps, t0 = 0, time.time()
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed * 100_000 + ep)
        ep_ret, done = 0.0, False
        while not done:
            if steps < warmup:
                action = env.action_space.sample()
            else:
                action = agent.act(obs, explore=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            # time-limit truncation is not a true terminal state
            buffer.add(obs, action, reward, next_obs, terminated)
            obs, ep_ret, steps = next_obs, ep_ret + reward, steps + 1
            done = terminated or truncated
            if steps >= warmup:
                agent.update(buffer, batch)
        log.log(_row(ep, steps, ep_ret, env, info, t0))
        _maybe_checkpoint(agent, tcfg, ep)


def _train_on_policy(agent: PPOAgent, env, tcfg: dict, seed: int, log: CSVLogger) -> None:
    episodes = int(tcfg.get("episodes", 500))
    steps, t0 = 0, time.time()
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed * 100_000 + ep)
        ep_ret, done = 0.0, False
        while not done:
            raw, logp, val = agent.sample(obs)
            next_obs, reward, terminated, truncated, info = env.step(np.clip(raw, -1.0, 1.0))
            done = terminated or truncated
            agent.rollout.add(obs, raw, logp, reward, terminated, val)
            obs, ep_ret, steps = next_obs, ep_ret + reward, steps + 1
            if truncated and not terminated:
                # bootstrap through the time limit by folding V(s_T) into the reward
                agent.rollout.rew[-1] += agent.gamma * agent.sample(next_obs)[2]
                agent.rollout.done[-1] = True
            if len(agent.rollout) >= agent.rollout_steps:
                agent.update(obs, done)
        log.log(_row(ep, steps, ep_ret, env, info, t0))
        _maybe_checkpoint(agent, tcfg, ep)


def _row(ep, steps, ep_ret, env, info, t0) -> dict:
    return {"episode": ep, "steps": steps, "return": round(float(ep_ret), 3), "length": env.steps,
            "success": int(info["success"]), "crash": int(info["crash"]),
            "wall_time": round(time.time() - t0, 1)}
