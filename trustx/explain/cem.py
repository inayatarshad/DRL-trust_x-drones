"""Contrastive Explanation Mechanism (Sections IV-B and IV-D, Eqs. 13-14).

Answers "why manoeuvre *a* rather than *a'*?" by searching for the smallest,
sparsest state perturbation that makes the policy pick ``a'``::

    min_ds  ||ds||_1 + lambda * (-log pi(a' | s + ds))
    s.t.    s + ds in S_valid,  ||ds||_inf <= eps

We solve it with proximal (ISTA-style) projected gradient descent: a gradient
step on the classification term, soft-thresholding for the L1 term, then
projection onto the box constraints. Only *mutable* features are perturbed -
an operator cannot change the UAV's position or the wind direction, but can
reason about sensor readings, battery level or wind strength.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from trustx.envs.uav_env import FEATURE_NAMES
from trustx.explain.maneuvers import MANEUVERS, maneuver_logits, prototypes

DEFAULT_MUTABLE = ["range_front", "range_left", "range_right", "range_down", "range_nearest",
                   "visibility", "wind_speed", "battery", "goal_dz"]

# (low, high) in normalised observation units; wind variations are limited to
# +-5 m/s (= 5/15 of the normalised wind range) as in the paper.
VALID_BOUNDS = {name: (-1.5, 1.5) for name in FEATURE_NAMES} | {
    "range_front": (0.0, 1.0), "range_left": (0.0, 1.0), "range_right": (0.0, 1.0),
    "range_down": (0.0, 1.0), "range_nearest": (0.0, 1.0),
    "visibility": (0.0, 1.0), "wind_speed": (0.0, 1.0), "battery": (0.0, 1.0),
}
MAX_CHANGE = {"wind_speed": 5.0 / 15.0}

# How to phrase a normalised change in physical units: (scale, unit)
UNITS = {"range_front": (20.0, "m"), "range_left": (20.0, "m"), "range_right": (20.0, "m"),
         "range_down": (20.0, "m"), "range_nearest": (20.0, "m"), "battery": (100.0, "%"),
         "wind_speed": (15.0, "m/s"), "visibility": (100.0, "%"), "goal_dz": (50.0, "m")}


@dataclass
class Counterfactual:
    original: str
    target: str
    delta: np.ndarray
    success: bool
    latency_ms: float
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))

    def changes(self, tol: float = 1e-3) -> list[tuple[str, float]]:
        idx = np.argsort(-np.abs(self.delta))
        return [(self.feature_names[i], float(self.delta[i])) for i in idx if abs(self.delta[i]) > tol]

    def text(self, max_terms: int = 2) -> str:
        if not self.success:
            return (f"No small change to the observed conditions would make the UAV "
                    f"{_verb(self.target)} instead of {_verb(self.original)}.")
        parts = []
        for name, d in self.changes()[:max_terms]:
            scale, unit = UNITS.get(name, (1.0, ""))
            word = "higher" if d > 0 else "lower"
            parts.append(f"{name} were {abs(d) * scale:.1f}{unit} {word}")
        return f"If {' and '.join(parts)}, the UAV would {_verb(self.target)} instead of {_verb(self.original)}."


def _verb(maneuver: str) -> str:
    return maneuver.replace("_", " ")


class ContrastiveExplainer:
    def __init__(self, actor: nn.Module, feature_names: list[str] | None = None,
                 mutable: list[str] | None = None, lam: float = 4.0, eps: float = 0.5,
                 steps: int = 40, lr: float = 0.05, l1: float = 0.02, device: str = "cpu"):
        self.actor = actor
        self.names = list(feature_names or FEATURE_NAMES)
        self.mask = torch.tensor([n in (mutable or DEFAULT_MUTABLE) for n in self.names], dtype=torch.float32)
        self.low = torch.tensor([VALID_BOUNDS.get(n, (-1.5, 1.5))[0] for n in self.names])
        self.high = torch.tensor([VALID_BOUNDS.get(n, (-1.5, 1.5))[1] for n in self.names])
        self.max_change = torch.tensor([min(eps, MAX_CHANGE.get(n, eps)) for n in self.names])
        self.lam, self.steps, self.lr, self.l1 = lam, steps, lr, l1
        self.device = device

    def _logits(self, s: torch.Tensor, protos: np.ndarray) -> torch.Tensor:
        return maneuver_logits(s, self.actor(s), protos=protos)

    def explain(self, state: np.ndarray, target: str | None = None) -> Counterfactual:
        t0 = time.perf_counter()
        s = torch.as_tensor(state, dtype=torch.float32).reshape(1, -1)
        # goal_dx/goal_dy are immutable, so the manoeuvre directions stay fixed
        protos = prototypes(s.numpy())
        with torch.no_grad():
            logits0 = self._logits(s, protos)[0]
        original = MANEUVERS[int(logits0.argmax())]
        if target is None:  # most plausible alternative
            ranked = torch.argsort(logits0, descending=True).tolist()
            target = MANEUVERS[ranked[1]]
        t_idx = MANEUVERS.index(target)

        delta = torch.zeros_like(s, requires_grad=True)
        best = None
        for _ in range(self.steps + 1):
            logits = self._logits(s + delta * self.mask, protos)
            if int(logits[0].argmax()) == t_idx:
                best = delta.detach().clone()
                break
            loss = self.lam * F.cross_entropy(logits, torch.tensor([t_idx]))
            grad, = torch.autograd.grad(loss, delta)
            with torch.no_grad():
                d = delta - self.lr * grad * self.mask
                d = torch.sign(d) * torch.clamp(d.abs() - self.lr * self.l1, min=0.0)   # prox of L1
                d = torch.maximum(torch.minimum(d, self.max_change), -self.max_change)  # ||ds||_inf <= eps
                d = torch.clamp(s + d, self.low, self.high) - s                        # S_valid
                delta.copy_(d * self.mask)
        success = best is not None
        final = (best if success else delta.detach())[0].numpy()
        return Counterfactual(original, target, final, success, (time.perf_counter() - t0) * 1e3, self.names)
