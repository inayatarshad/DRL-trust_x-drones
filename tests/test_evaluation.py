import numpy as np
import pytest
import torch

from trustx.agents import build_agent
from trustx.envs import make_env
from trustx.evaluation import (EvalConfig, Method, evaluate_autonomous, evaluate_method, mcnemar, normal_ci,
                               paired_t, run_mission)
from trustx.framework import TrustXExplainer
from trustx.trust.bayesian import KalmanTrustFilter
from trustx.trust.operator import SimulatedOperator, make_operator_pool


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    env = make_env({"max_steps": 40})
    agent = build_agent("td3", 18, 3, hidden=(32, 32))
    explainer, fidelity = TrustXExplainer.build(agent, env, n_samples=600)
    return env, agent, explainer, fidelity


def test_explanation_contents(setup):
    env, agent, explainer, fidelity = setup
    obs, _ = env.reset(seed=0)
    e = explainer.explain(obs, level="detailed")
    text = e.text()
    assert text.startswith("Action:") and "Rule: IF" in text and "Key factors" in text and "Contrast" in text
    assert "Key factors" not in explainer.explain(obs, level="concise").text()
    assert 0 <= e.quality <= 1 and e.latency_ms > 0
    assert fidelity <= 1  # R^2 can be very negative for an untrained, near-constant policy


def test_run_mission_records_trace(setup):
    env, agent, explainer, _ = setup
    m = Method("TRUST-X", agent, explainer)
    r = run_mission(env, m, SimulatedOperator("balanced"), seed=1, trust_filter=KalmanTrustFilter(), record=True)
    assert r.steps == len(r.trace["pos"]) and r.latency_ms
    assert set(r.trace["mode"]) <= {"Cruise", "Obstacle Avoidance", "Final Approach", "Manual Override",
                                    "Emergency Return"}


def test_evaluate_method_is_paired_and_deterministic(setup):
    env, agent, _, _ = setup
    ops = make_operator_pool(2, seed=3)
    a = evaluate_method(env, Method("TD3", agent), ops, 2, EvalConfig())
    b = evaluate_method(env, Method("TD3", agent), ops, 2, EvalConfig())
    assert a["_success_vector"] == b["_success_vector"]
    assert a["trust_delta"] == pytest.approx(b["trust_delta"])
    assert a["missions"] == 4


def test_autonomous_benchmark(setup):
    env, agent, _, _ = setup
    r = evaluate_autonomous(env, agent, [1, 2, 3])
    assert 0 <= r["success_rate"] <= 1 and r["mean_steps"] > 0


def test_normal_ci():
    lo, hi = normal_ci(0.9, 100)
    assert lo == pytest.approx(0.9 - 1.96 * 0.03, abs=1e-6) and hi == pytest.approx(0.9 + 1.96 * 0.03, abs=1e-6)
    assert normal_ci(1.0, 10) == (1.0, 1.0)


def test_mcnemar_detects_paired_improvement():
    rng = np.random.default_rng(0)
    base = (rng.random(400) < 0.6).astype(int)
    better = np.maximum(base, (rng.random(400) < 0.5).astype(int))
    chi2, p = mcnemar(base.tolist(), better.tolist())
    assert p < 0.001
    assert mcnemar([1, 0], [1, 0]) == (0.0, 1.0)


def test_paired_t_one_sided():
    base = [3.0, 2.5, 4.0, 3.2, 2.8]
    _, p = paired_t(base, [b - 1.0 + 0.1 * i for i, b in enumerate(base)])
    assert p < 0.01
