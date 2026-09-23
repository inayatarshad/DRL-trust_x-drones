"""Surrogate decision-tree explanations (Section IV-B).

A shallow regression tree ``f_hat(s) ~ pi(s)`` is distilled from state/action
pairs of the trained policy. Its decision path for the current state is a
human-readable if-then rule, and Eq. (19) measures how faithfully it mimics
the policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_text

from trustx.explain.maneuvers import classify


def policy_fidelity(pi: np.ndarray, f_hat: np.ndarray) -> float:
    """Coefficient of determination between policy and surrogate actions, Eq. (19)."""
    pi, f_hat = np.atleast_2d(pi), np.atleast_2d(f_hat)
    ss_res = np.sum((pi - f_hat) ** 2)
    ss_tot = np.sum((pi - pi.mean(axis=0)) ** 2)
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


@dataclass
class Condition:
    feature: str
    op: str
    threshold: float

    def __str__(self) -> str:
        return f"{self.feature} {self.op} {self.threshold:.2f}"


@dataclass
class Rule:
    conditions: list[Condition]
    action: np.ndarray
    maneuver: str
    support: int

    def text(self, max_conditions: int | None = None) -> str:
        conds = self.conditions if max_conditions is None else self.conditions[-max_conditions:]
        return f"IF {' AND '.join(map(str, conds)) or 'TRUE'} THEN {self.maneuver.upper()}"


def _simplify(conds: list[Condition]) -> list[Condition]:
    """Keep only the tightest bound per (feature, direction)."""
    best: dict[tuple[str, str], Condition] = {}
    for c in conds:
        key = (c.feature, c.op)
        prev = best.get(key)
        if prev is None or (c.op == "<=" and c.threshold < prev.threshold) or (
                c.op == ">" and c.threshold > prev.threshold):
            best[key] = c
    return [c for c in conds if best[(c.feature, c.op)] is c]


class SurrogateTree:
    def __init__(self, feature_names: list[str], max_depth: int = 8, min_samples_leaf: int = 10,
                 random_state: int = 0):
        self.feature_names = list(feature_names)
        self.tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                          random_state=random_state)

    def fit(self, states: np.ndarray, actions: np.ndarray) -> "SurrogateTree":
        self.tree.fit(states, actions)
        return self

    def predict(self, states: np.ndarray) -> np.ndarray:
        return self.tree.predict(np.atleast_2d(states))

    def fidelity(self, states: np.ndarray, actions: np.ndarray) -> float:
        return policy_fidelity(actions, self.predict(states))

    def rule(self, state: np.ndarray) -> Rule:
        t = self.tree.tree_
        x = np.asarray(state, dtype=np.float64).reshape(1, -1)
        conds: list[Condition] = []
        for node in self.tree.decision_path(x).indices:
            if t.children_left[node] == t.children_right[node]:  # leaf
                leaf = node
                break
            f, thr = t.feature[node], t.threshold[node]
            op = "<=" if x[0, f] <= thr else ">"
            conds.append(Condition(self.feature_names[f], op, float(thr)))
        action = t.value[leaf][:, 0]
        return Rule(_simplify(conds), action, str(classify(x[0], action)), int(t.n_node_samples[leaf]))

    def export_text(self, max_depth: int = 4) -> str:
        return export_text(self.tree, feature_names=self.feature_names, max_depth=max_depth, decimals=2)


def collect_policy_dataset(policy, env, n_samples: int = 10_000, noise: float = 0.1,
                           seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Roll out ``policy`` (with a little exploration noise to widen coverage) and
    return visited states with the policy's *noise-free* actions."""
    rng = np.random.default_rng(seed)
    states = []
    ep = 0
    while len(states) < n_samples:
        obs, _ = env.reset(seed=seed * 1_000_003 + ep)
        ep += 1
        done = False
        while not done and len(states) < n_samples:
            states.append(obs)
            a = policy(obs[None])[0] + rng.normal(0.0, noise, size=3)
            obs, _, terminated, truncated, _ = env.step(np.clip(a, -1, 1))
            done = terminated or truncated
    states = np.asarray(states, dtype=np.float32)
    return states, policy(states)
