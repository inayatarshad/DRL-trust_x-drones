import numpy as np
import pytest

from trustx.envs import FEATURE_NAMES, SafeFallbackController, make_env


@pytest.fixture
def env():
    return make_env()


def test_observation_shape_and_names(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == (len(FEATURE_NAMES),)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert "clearance" in info


def test_reset_is_deterministic_under_seed(env):
    a, _ = env.reset(seed=123)
    b, _ = env.reset(seed=123)
    np.testing.assert_allclose(a, b)
    assert env.observation_space.shape == a.shape


def test_start_and_goal_are_collision_free(env):
    for seed in range(20):
        env.reset(seed=seed)
        assert env._clearance(env.pos) > env.cfg.uav_radius
        assert env._clearance(env.waypoints[-1]) > env.cfg.uav_radius


def test_step_contract(env):
    env.reset(seed=1)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)


def test_episode_terminates(env):
    env.reset(seed=2)
    for _ in range(env.cfg.max_steps + 1):
        _, _, terminated, truncated, _ = env.step(np.zeros(3))
        if terminated or truncated:
            break
    assert terminated or truncated


def test_flying_into_ground_crashes(env):
    env.reset(seed=3)
    for _ in range(100):
        _, reward, terminated, _, info = env.step(np.array([0.0, 0.0, -1.0]))
        if terminated:
            break
    assert info["crash"]
    assert reward < -50


def test_battery_drains(env):
    env.reset(seed=4)
    env.step(np.array([1.0, 1.0, 0.0]))
    assert env.battery < 1.0


def test_fallback_controller_reaches_goal_sometimes(env):
    ctl = SafeFallbackController()
    successes = 0
    for seed in range(20):
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(ctl(obs))
            done = terminated or truncated
        successes += info["success"]
    assert successes >= 8


def test_multi_waypoint_mission():
    env = make_env({"n_waypoints": 3})
    env.reset(seed=0)
    assert len(env.waypoints) == 3
