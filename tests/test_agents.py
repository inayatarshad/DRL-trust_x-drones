import numpy as np
import pytest
import torch

from trustx.agents import AGENTS, ReplayBuffer, build_agent, load_agent
from trustx.training import train

OBS, ACT = 18, 3


@pytest.mark.parametrize("name", sorted(AGENTS))
def test_act_is_bounded(name):
    agent = build_agent(name, OBS, ACT, hidden=(32, 32))
    obs = np.random.randn(5, OBS).astype(np.float32)
    a = agent.act(obs, explore=True)
    assert a.shape == (5, ACT)
    assert np.all(np.abs(a) <= 1.0)


def _filled_buffer(n=512):
    buf = ReplayBuffer(OBS, ACT, capacity=1000)
    for _ in range(n):
        buf.add(np.random.randn(OBS), np.random.uniform(-1, 1, ACT), np.random.randn(),
                np.random.randn(OBS), np.random.rand() < 0.05)
    return buf


def test_replay_buffer_wraps():
    buf = ReplayBuffer(OBS, ACT, capacity=10)
    for i in range(25):
        buf.add(np.zeros(OBS), np.zeros(ACT), float(i), np.zeros(OBS), False)
    assert len(buf) == 10
    obs, act, rew, nxt, done = buf.sample(4)
    assert obs.shape == (4, OBS) and rew.min() >= 15


@pytest.mark.parametrize("name", ["td3", "ddpg"])
def test_off_policy_update_changes_actor(name):
    torch.manual_seed(0)
    agent = build_agent(name, OBS, ACT, hidden=(32, 32))
    before = [p.clone() for p in agent.actor.parameters()]
    buf = _filled_buffer()
    for _ in range(4):
        info = agent.update(buf, batch_size=64)
    assert "critic_loss" in info
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.actor.parameters()))


def test_td3_exposes_twin_critics():
    agent = build_agent("td3", OBS, ACT, hidden=(32, 32))
    q = agent.q_values(np.zeros((7, OBS)), np.zeros((7, ACT)))
    assert q.shape == (2, 7)


def test_ppo_gae_on_constant_reward():
    agent = build_agent("ppo", OBS, ACT, hidden=(16, 16), gamma=0.5, lam=1.0)
    for _ in range(3):
        agent.rollout.add(np.zeros(OBS), np.zeros(ACT), 0.0, 1.0, False, 0.0)
    adv, ret = agent._advantages(last_value=0.0)
    np.testing.assert_allclose(ret, [1.75, 1.5, 1.0])


def test_save_load_roundtrip(tmp_path):
    agent = build_agent("td3", OBS, ACT, hidden=(64, 32))
    obs = np.random.randn(3, OBS).astype(np.float32)
    agent.save(tmp_path / "m.pt")
    clone = load_agent(str(tmp_path / "m.pt"))
    np.testing.assert_allclose(agent.act(obs), clone.act(obs))


@pytest.mark.parametrize("algo", ["td3", "ppo"])
def test_short_training_run(tmp_path, algo):
    cfg = {"algo": algo, "seed": 0, "env": {"max_steps": 30},
           "agent": {"hidden": [32, 32]} | ({"rollout_steps": 64, "minibatch": 32} if algo == "ppo" else {}),
           "train": {"episodes": 3, "warmup_steps": 20, "batch_size": 32, "log_every": 0}}
    train(cfg, tmp_path)
    assert (tmp_path / "model.pt").exists()
    assert (tmp_path / "train_log.csv").read_text().count("\n") == 4
