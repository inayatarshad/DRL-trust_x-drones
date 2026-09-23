"""Bayesian operator-trust estimation (Section IV-E, Eqs. 15-16).

Operators tag decisions as Trusted (+1), Unclear (0) or Disagree (-1). The
latent trust ``tau in [0, 1]`` generates tags through Eq. (15):

    P(Z=+1 | tau) ~ beta       * sigmoid( alpha1 * (tau - eta1))
    P(Z= 0 | tau) ~ (1 - beta) * N(tau; mu0, sigma0^2)
    P(Z=-1 | tau) ~ beta       * sigmoid(-alpha2 * (tau - eta2))

(normalised over ``z`` for every ``tau``). Two filters estimate ``tau``:

* :class:`GridTrustFilter` - exact recursive Bayes (Eq. 16) on a grid with a
  Gaussian random-walk transition.
* :class:`KalmanTrustFilter` - the real-time Kalman-style variant used in the
  paper, with a moment-matched (assumed-density) measurement update.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TAGS = {"trusted": 1, "unclear": 0, "disagree": -1}


@dataclass
class TrustLikelihood:
    beta: float = 0.7
    alpha1: float = 8.0
    eta1: float = 0.5
    alpha2: float = 8.0
    eta2: float = 0.5
    mu0: float = 0.5
    sigma0: float = 0.2

    def unnormalised(self, tau: np.ndarray) -> np.ndarray:
        """Shape (3, len(tau)) for z = -1, 0, +1."""
        tau = np.asarray(tau, dtype=np.float64)
        sig = lambda x: 1.0 / (1.0 + np.exp(-x))  # noqa: E731
        p_pos = self.beta * sig(self.alpha1 * (tau - self.eta1))
        p_neg = self.beta * sig(-self.alpha2 * (tau - self.eta2))
        p_mid = (1.0 - self.beta) * np.exp(-0.5 * ((tau - self.mu0) / self.sigma0) ** 2)
        return np.stack([p_neg, p_mid, p_pos])

    def probs(self, tau) -> np.ndarray:
        """P(Z = z | tau) for z = -1, 0, +1 (columns sum to one)."""
        u = self.unnormalised(tau)
        return u / u.sum(axis=0, keepdims=True)

    def __call__(self, z: int, tau) -> np.ndarray:
        return self.probs(tau)[z + 1]

    def sample(self, tau: float, rng: np.random.Generator) -> int:
        return int(rng.choice([-1, 0, 1], p=self.probs(np.array([tau]))[:, 0]))


class GridTrustFilter:
    """Exact Bayes filter over a discretised trust grid."""

    def __init__(self, likelihood: TrustLikelihood | None = None, prior_mean: float = 0.5,
                 prior_std: float = 0.2, process_std: float = 0.03, n_grid: int = 201):
        self.lik = likelihood or TrustLikelihood()
        self.grid = np.linspace(0.0, 1.0, n_grid)
        self.process_std = process_std
        belief = np.exp(-0.5 * ((self.grid - prior_mean) / prior_std) ** 2)
        self.belief = belief / belief.sum()
        diff = self.grid[:, None] - self.grid[None, :]
        k = np.exp(-0.5 * (diff / max(process_std, 1e-6)) ** 2)
        self._transition = k / k.sum(axis=0, keepdims=True)

    def predict(self) -> None:
        self.belief = self._transition @ self.belief

    def update(self, z: int) -> None:
        self.predict()
        post = self.belief * self.lik(z, self.grid)
        self.belief = post / post.sum()

    @property
    def mean(self) -> float:
        return float(self.belief @ self.grid)

    @property
    def var(self) -> float:
        return float(self.belief @ (self.grid - self.mean) ** 2)


class KalmanTrustFilter:
    """Gaussian (assumed-density) Kalman filter for real-time use.

    Prediction is the usual random-walk step ``P <- P + Q``. Because Eq. (15)
    is non-Gaussian, the measurement update matches the first two moments of
    ``N(tau; mu, P) * P(z | tau)`` using a small fixed quadrature - the
    moment-matching analogue of the Kalman update, at O(n_points) cost.
    """

    def __init__(self, likelihood: TrustLikelihood | None = None, prior_mean: float = 0.5,
                 prior_var: float = 0.04, process_var: float = 0.03 ** 2, n_points: int = 64):
        self.lik = likelihood or TrustLikelihood()
        self.mu, self.P, self.Q = prior_mean, prior_var, process_var
        self._nodes = np.linspace(-4.0, 4.0, n_points)

    def update(self, z: int) -> None:
        self.P += self.Q
        tau = np.clip(self.mu + np.sqrt(self.P) * self._nodes, 0.0, 1.0)
        w = np.exp(-0.5 * self._nodes ** 2) * self.lik(z, tau)
        w /= w.sum()
        self.mu = float(w @ tau)
        self.P = float(max(w @ (tau - self.mu) ** 2, 1e-6))

    @property
    def mean(self) -> float:
        return self.mu

    @property
    def var(self) -> float:
        return self.P


def explanation_level(tau_hat: float) -> str:
    """Adapt explanation granularity: concise when trust is high, detailed when low."""
    if tau_hat >= 0.7:
        return "concise"
    if tau_hat >= 0.5:
        return "standard"
    return "detailed"


def needs_override(tau_hat: float, var: float, tau_threshold: float = 0.5, var_threshold: float = 0.3) -> bool:
    """Algorithm 1, line 9 - hand authority to the operator when trust is low or uncertain."""
    return tau_hat < tau_threshold or var > var_threshold
