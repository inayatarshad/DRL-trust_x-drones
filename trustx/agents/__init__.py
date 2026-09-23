from trustx.agents.base import Agent
from trustx.agents.ddpg import DDPGAgent
from trustx.agents.ppo import PPOAgent
from trustx.agents.replay_buffer import ReplayBuffer
from trustx.agents.td3 import TD3Agent

AGENTS = {"td3": TD3Agent, "ddpg": DDPGAgent, "ppo": PPOAgent}


def build_agent(name: str, obs_dim: int, act_dim: int, **kwargs) -> Agent:
    try:
        cls = AGENTS[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown agent {name!r}; choose from {sorted(AGENTS)}") from exc
    return cls(obs_dim, act_dim, **kwargs)


def load_agent(path: str, **kwargs) -> Agent:
    import torch

    blob = torch.load(path, map_location="cpu", weights_only=False)
    agent = build_agent(blob["name"], blob["obs_dim"], blob["act_dim"], **kwargs)
    agent.load_state_dict(blob["state"])
    return agent


__all__ = ["Agent", "TD3Agent", "DDPGAgent", "PPOAgent", "ReplayBuffer", "AGENTS", "build_agent", "load_agent"]
