"""SHAP-lite: Monte Carlo Shapley attributions under a latency budget (Section IV-C).

Exact Shapley values (Eq. 11) need ``O(2^|F|)`` model calls. SHAP-lite follows
Eq. (12): for ``M`` random feature permutations and background samples it
measures the marginal change caused by switching feature ``i`` from its
background value to its true value. Evaluating every prefix of a permutation
makes the estimate *efficient*: for every sample the attributions telescope, so

    sum_i phi_i = f(x) - E_z[f(z)]

holds exactly (up to the background sample), which is the SHAP additivity
property. All ``M * (|F| + 1)`` points are evaluated in a single batched call.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

Model = Callable[[np.ndarray], np.ndarray]


@dataclass
class Attribution:
    phi: np.ndarray
    base_value: float
    prediction: float
    feature_names: list[str]

    def top(self, k: int = 3) -> list[tuple[str, float]]:
        order = np.argsort(-np.abs(self.phi))[:k]
        return [(self.feature_names[i], float(self.phi[i])) for i in order]

    def text(self, k: int = 3) -> str:
        return ", ".join(f"{n} ({'+' if v >= 0 else '-'}{abs(v):.2f})" for n, v in self.top(k))


class ShapLite:
    def __init__(self, model: Model, background: np.ndarray, feature_names: list[str],
                 n_samples: int = 100, seed: int = 0):
        self.model = model
        self.background = np.asarray(background, dtype=np.float64)
        self.feature_names = list(feature_names)
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def explain(self, x: np.ndarray) -> Attribution:
        x = np.asarray(x, dtype=np.float64)
        n_feat, m = x.shape[0], self.n_samples
        perms = np.argsort(self.rng.random((m, n_feat)), axis=1)
        z = self.background[self.rng.integers(0, len(self.background), m)]
        # rank[m, j] = position of feature j in permutation m
        rank = np.empty_like(perms)
        rank[np.arange(m)[:, None], perms] = np.arange(n_feat)
        # prefix k (k = 0..F) takes features with rank < k from x, the rest from z
        k = np.arange(n_feat + 1)[None, :, None]
        take_x = rank[:, None, :] < k                       # (m, F+1, F)
        points = np.where(take_x, x[None, None, :], z[:, None, :])
        values = np.asarray(self.model(points.reshape(-1, n_feat)), dtype=np.float64).reshape(m, n_feat + 1)
        marginal = np.diff(values, axis=1)                  # (m, F): gain of adding perms[m, k]
        phi = np.zeros(n_feat)
        np.add.at(phi, perms.ravel(), marginal.ravel())
        phi /= m
        return Attribution(phi, float(values[:, 0].mean()), float(values[:, -1].mean()), self.feature_names)


def exact_shapley(model: Model, x: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Brute-force interventional Shapley values, Eq. (11). Only for small |F| (tests)."""
    x = np.asarray(x, dtype=np.float64)
    background = np.asarray(background, dtype=np.float64)
    n = x.shape[0]

    def v(coalition: tuple[int, ...]) -> float:
        pts = background.copy()
        pts[:, list(coalition)] = x[list(coalition)]
        return float(np.mean(model(pts)))

    phi = np.zeros(n)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for size in range(n):
            w = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            for s in itertools.combinations(others, size):
                phi[i] += w * (v(s + (i,)) - v(s))
    return phi
