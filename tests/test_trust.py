import numpy as np
import pytest

from trustx.trust.bayesian import (GridTrustFilter, KalmanTrustFilter, TrustLikelihood, explanation_level,
                                   needs_override)
from trustx.trust.operator import PROFILES, Observation, SimulatedOperator, make_operator_pool


def test_likelihood_normalised_and_monotone():
    lik = TrustLikelihood()
    tau = np.linspace(0, 1, 11)
    p = lik.probs(tau)
    np.testing.assert_allclose(p.sum(axis=0), 1.0)
    assert np.all(np.diff(p[2]) > 0)   # P(trusted) increases with trust
    assert np.all(np.diff(p[0]) < 0)   # P(disagree) decreases with trust


@pytest.mark.parametrize("cls", [GridTrustFilter, KalmanTrustFilter])
def test_filters_move_with_evidence(cls):
    f = cls()
    for _ in range(10):
        f.update(1)
    high = f.mean
    for _ in range(20):
        f.update(-1)
    assert high > 0.7 and f.mean < 0.45
    assert 0 <= f.var < 0.1


def test_kalman_tracks_grid_filter():
    rng = np.random.default_rng(0)
    g, k = GridTrustFilter(), KalmanTrustFilter()
    for z in rng.choice([-1, 0, 1], size=50, p=[0.2, 0.3, 0.5]):
        g.update(int(z))
        k.update(int(z))
        assert abs(g.mean - k.mean) < 0.1


def test_filter_recovers_true_trust():
    rng = np.random.default_rng(1)
    lik = TrustLikelihood()
    for true in (0.2, 0.8):
        f = KalmanTrustFilter(likelihood=lik)
        for _ in range(200):
            f.update(lik.sample(true, rng))
        assert abs(f.mean - true) < 0.15


def test_granularity_and_override_rules():
    assert explanation_level(0.9) == "concise"
    assert explanation_level(0.6) == "standard"
    assert explanation_level(0.3) == "detailed"
    assert needs_override(0.4, 0.01) and needs_override(0.8, 0.5) and not needs_override(0.8, 0.01)


def _obs(q=0.0, level=None, **kw):
    base = dict(maneuver_changed=False, near_miss=False, risky=False, progress=1.0)
    base.update(kw)
    return Observation(q, level, **base)


def test_explanations_raise_trust():
    explained = SimulatedOperator("balanced", seed=0)
    opaque = SimulatedOperator("balanced", seed=0)
    for _ in range(100):
        explained.review(_obs(0.9, "standard", maneuver_changed=True))
        opaque.review(_obs(0.0, None, maneuver_changed=True))
    assert explained.trust > opaque.trust + 0.2


def test_near_misses_lower_trust():
    op = SimulatedOperator("optimistic", seed=0)
    start = op.trust
    for _ in range(20):
        op.review(_obs(near_miss=True, progress=-1.0))
    assert op.trust < start


def test_low_trust_operator_overrides_when_risky():
    op = SimulatedOperator("skeptical", seed=0)
    op.trust = 0.1
    overrides = [op.review(_obs(risky=True, near_miss=True))[1] for _ in range(50)]
    assert any(overrides)
    op.trust = 0.1
    assert not any(op.review(_obs(risky=False))[1] for _ in range(50) if op.trust < 0.5)


def test_operator_pool_mix():
    pool = make_operator_pool(6)
    assert {o.profile.name for o in pool} == set(PROFILES)
