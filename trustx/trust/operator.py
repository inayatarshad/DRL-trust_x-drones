"""Simulated human operators.

The paper evaluates TRUST-X with 20 human operators. This repository replaces
them with *simulated* operators so the whole loop can run end-to-end without a
user study. Each operator has a latent trust ``tau`` that evolves according to
the mean-reverting dynamics of Eq. (17):

    d tau = alpha (tau_inf - tau) dt + sigma dW + beta * 1[good explanation]

plus penalties for events that erode trust (near misses, and manoeuvre
changes the operator could not anticipate because nothing explained them).
Tags ``Z in {-1, 0, +1}`` are sampled from the Eq. (15) likelihood given the
operator's current trust, and the operator takes manual control when a risky
situation coincides with low trust.

Results obtained with these operators are consequences of the modelling
assumptions below; they are not a substitute for a human-subject study.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from trustx.trust.bayesian import TrustLikelihood


@dataclass
class OperatorProfile:
    name: str
    initial_trust: float
    equilibrium: float        # tau_inf for an opaque system
    reversion: float = 0.05   # alpha
    noise: float = 0.02       # sigma_tau
    explanation_gain: float = 0.03   # beta jump for a good, well-matched explanation
    surprise_penalty: float = 0.04   # unexplained manoeuvre change
    near_miss_penalty: float = 0.06
    override_threshold: float = 0.5
    override_prob: float = 0.6       # chance of grabbing the stick when conditions are met
    need_detail_below: float = 0.5   # wants detailed explanations when trust is below this


PROFILES: dict[str, OperatorProfile] = {
    "skeptical": OperatorProfile("skeptical", initial_trust=0.40, equilibrium=0.45,
                                 surprise_penalty=0.05, near_miss_penalty=0.08, override_prob=0.7),
    "balanced": OperatorProfile("balanced", initial_trust=0.55, equilibrium=0.55),
    "optimistic": OperatorProfile("optimistic", initial_trust=0.70, equilibrium=0.65,
                                  surprise_penalty=0.03, near_miss_penalty=0.05, override_prob=0.5),
}


@dataclass
class Observation:
    """What the operator perceives at a review point."""
    explanation_quality: float     # 0 = black box, 1 = perfectly clear and faithful
    explanation_level: str | None  # "concise" / "standard" / "detailed" or None
    maneuver_changed: bool
    near_miss: bool
    risky: bool                    # hazard ahead (low clearance or high uncertainty)
    progress: float                # metres gained towards the waypoint since last review


class SimulatedOperator:
    def __init__(self, profile: OperatorProfile | str, seed: int = 0,
                 likelihood: TrustLikelihood | None = None, dt: float = 1.0):
        self.profile = PROFILES[profile] if isinstance(profile, str) else profile
        self.rng = np.random.default_rng(seed)
        self.lik = likelihood or TrustLikelihood()
        self.dt = dt
        self.trust = self.profile.initial_trust

    def perceived_quality(self, obs: Observation) -> float:
        """Explanations only help if their granularity matches the operator's need."""
        q = obs.explanation_quality
        if obs.explanation_level is None:
            return q
        wants_detail = self.trust < self.profile.need_detail_below
        if wants_detail and obs.explanation_level == "concise":
            q *= 0.6
        elif not wants_detail and obs.explanation_level == "detailed":
            q *= 0.85  # information overload when the operator already trusts the UAV
        return q

    def review(self, obs: Observation) -> tuple[int, bool]:
        """Update latent trust (Eq. 17), emit a tag (Eq. 15) and decide on an override."""
        p = self.profile
        q = self.perceived_quality(obs)
        d = p.reversion * (p.equilibrium - self.trust) * self.dt
        d += p.noise * np.sqrt(self.dt) * self.rng.normal()
        good_outcome = obs.progress > 0 and not obs.near_miss
        if q > 0.5 and good_outcome:
            d += p.explanation_gain * q
        if obs.maneuver_changed:
            d -= p.surprise_penalty * (1.0 - q)
        if obs.near_miss:
            d -= p.near_miss_penalty * (1.0 - 0.5 * q)
        self.trust = float(np.clip(self.trust + d, 0.0, 1.0))

        tag = self.lik.sample(self.trust, self.rng)
        override = bool(obs.risky and self.trust < p.override_threshold
                        and self.rng.random() < p.override_prob)
        return tag, override


def make_operator_pool(n: int, seed: int = 0) -> list[SimulatedOperator]:
    """A mixed pool of operators with slightly jittered profiles (novice to expert)."""
    rng = np.random.default_rng(seed)
    names = list(PROFILES)
    pool = []
    for i in range(n):
        base = PROFILES[names[i % len(names)]]
        jitter = lambda v, s=0.05: float(np.clip(v + rng.normal(0, s), 0.05, 0.95))  # noqa: E731
        prof = replace(base, initial_trust=jitter(base.initial_trust), equilibrium=jitter(base.equilibrium))
        pool.append(SimulatedOperator(prof, seed=seed * 1000 + i))
    return pool
