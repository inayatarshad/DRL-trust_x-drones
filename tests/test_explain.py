import numpy as np
import pytest
import torch

from trustx.agents import build_agent
from trustx.envs import FEATURE_NAMES, SafeFallbackController, make_env
from trustx.explain.cem import ContrastiveExplainer
from trustx.explain.maneuvers import MANEUVERS, classify, maneuver_logits
from trustx.explain.shap_lite import ShapLite, exact_shapley
from trustx.explain.surrogate import SurrogateTree, collect_policy_dataset, policy_fidelity
from trustx.explain.uncertainty import UncertaintyEstimator

IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}


def _obs(**kw):
    o = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    o[IDX["goal_dx"]] = 1.0
    for k, v in kw.items():
        o[IDX[k]] = v
    return o


@pytest.mark.parametrize("action,label", [
    ([1.0, 0.0, 0.0], "advance"),
    ([0.7, 0.7, 0.0], "veer_left"),
    ([0.7, -0.7, 0.0], "veer_right"),
    ([0.0, 0.0, 1.0], "climb"),
    ([0.0, 0.0, -1.0], "descend"),
    ([0.05, 0.0, 0.0], "hold"),
])
def test_maneuver_classification(action, label):
    assert classify(_obs(), np.array(action)) == label


def test_maneuver_logits_tensor_matches_numpy():
    o, a = _obs(), np.array([0.3, 0.5, -0.2])
    np_logits = maneuver_logits(o, a)
    t_logits = maneuver_logits(torch.tensor(o), torch.tensor(a, dtype=torch.float64)).numpy()
    np.testing.assert_allclose(np_logits, t_logits, atol=1e-6)
    assert np_logits.shape == (len(MANEUVERS),)


def test_policy_fidelity_bounds():
    x = np.random.randn(100, 3)
    assert policy_fidelity(x, x) == pytest.approx(1.0)
    assert policy_fidelity(x, np.tile(x.mean(0), (100, 1))) == pytest.approx(0.0)


def test_surrogate_learns_fallback_controller():
    env = make_env()
    ctl = SafeFallbackController()
    policy = lambda o: np.stack([ctl(x) for x in np.atleast_2d(o)])  # noqa: E731
    states, actions = collect_policy_dataset(policy, env, 3000)
    tree = SurrogateTree(FEATURE_NAMES).fit(states[:2400], actions[:2400])
    assert tree.fidelity(states[2400:], actions[2400:]) > 0.8
    rule = tree.rule(states[0])
    assert rule.text().startswith("IF ") and rule.maneuver in MANEUVERS
    assert rule.support > 0


def test_shap_lite_matches_exact_shapley():
    rng = np.random.default_rng(0)
    bg = rng.normal(size=(32, 4))
    x = rng.normal(size=4)
    f = lambda X: 2 * X[:, 0] + X[:, 1] * X[:, 2] - X[:, 3] ** 2  # noqa: E731
    exact = exact_shapley(f, x, bg)
    approx = ShapLite(f, bg, list("abcd"), n_samples=3000, seed=1).explain(x).phi
    np.testing.assert_allclose(approx, exact, atol=0.1)


def test_shap_lite_additivity_and_linear_case():
    rng = np.random.default_rng(1)
    bg = rng.normal(size=(64, 6))
    w = np.arange(1, 7, dtype=float)
    f = lambda X: X @ w  # noqa: E731
    x = rng.normal(size=6)
    attr = ShapLite(f, bg, list("abcdef"), n_samples=50).explain(x)
    assert attr.phi.sum() == pytest.approx(attr.prediction - attr.base_value, abs=1e-8)
    assert attr.top(1)[0][0] == attr.feature_names[int(np.argmax(np.abs(attr.phi)))]


class _ClimbWhenClose(torch.nn.Module):
    """Toy policy: advance, but climb when range_front drops below 0.4."""

    def forward(self, s):
        s = torch.atleast_2d(s)
        close = torch.sigmoid(20 * (0.4 - s[:, IDX["range_front"]]))
        a = torch.stack([1 - close, torch.zeros_like(close), close], dim=-1)
        return a


def test_cem_finds_minimal_counterfactual():
    cem = ContrastiveExplainer(_ClimbWhenClose(), steps=200, lr=0.05)
    cf = cem.explain(_obs(range_front=0.8, range_nearest=0.8, battery=1.0), target="climb")
    assert cf.original == "advance"
    assert cf.success
    top = cf.changes()[0]
    assert top[0] == "range_front" and top[1] < 0
    assert "climb instead of advance" in cf.text()


def test_cem_respects_immutable_features():
    cem = ContrastiveExplainer(_ClimbWhenClose(), steps=50)
    cf = cem.explain(_obs(range_front=0.8), target="climb")
    assert cf.delta[IDX["pos_x"]] == 0.0 and cf.delta[IDX["wind_dir"]] == 0.0


def test_uncertainty_in_range():
    agent = build_agent("td3", len(FEATURE_NAMES), 3, hidden=(32, 32))
    u = UncertaintyEstimator(agent)(_obs(visibility=0.3))
    assert 0 <= u.aleatoric < 1 and 0 <= u.epistemic < 1
    assert 0 <= u.confidence <= 1
