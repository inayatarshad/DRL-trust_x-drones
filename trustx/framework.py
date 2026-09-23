"""TRUST-X explain-and-adapt layer (Section IV, Algorithm 1).

:class:`TrustXExplainer` bundles the three explanation modules and the
uncertainty estimator around a trained agent and produces, for every
decision, an :class:`Explanation`::

    E_t = (rule, top-3 SHAP-lite features, counterfactual, confidence)

whose verbosity follows the operator-trust estimate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from trustx.agents import Agent
from trustx.envs.uav_env import FEATURE_NAMES
from trustx.explain.cem import ContrastiveExplainer, Counterfactual
from trustx.explain.maneuvers import classify
from trustx.explain.shap_lite import Attribution, ShapLite
from trustx.explain.surrogate import Rule, SurrogateTree, collect_policy_dataset
from trustx.explain.uncertainty import Uncertainty, UncertaintyEstimator


@dataclass
class Explanation:
    maneuver: str
    rule: Rule
    attribution: Attribution | None
    counterfactual: Counterfactual | None
    uncertainty: Uncertainty
    level: str
    local_agreement: float
    latency_ms: float
    quality: float

    def text(self) -> str:
        lines = [f"Action: {self.maneuver.replace('_', ' ').upper()} "
                 f"(confidence {self.uncertainty.confidence:.0%})"]
        max_conds = {"concise": 2, "standard": 4}.get(self.level)
        lines.append(f"Rule: {self.rule.text(max_conds)}")
        if self.attribution is not None and self.level != "concise":
            lines.append(f"Key factors: {self.attribution.text(3)}")
        if self.counterfactual is not None and self.level == "detailed":
            lines.append(f"Contrast: {self.counterfactual.text()}")
        return "\n".join(lines)


class TrustXExplainer:
    def __init__(self, agent: Agent, surrogate: SurrogateTree, background: np.ndarray,
                 use_shap: bool = True, use_cem: bool = True, shap_samples: int = 100, seed: int = 0):
        self.agent = agent
        self.surrogate = surrogate
        self.use_shap, self.use_cem = use_shap, use_cem
        self._shap_target = np.zeros(3)
        self.shap = ShapLite(self._shap_model, background, FEATURE_NAMES, n_samples=shap_samples, seed=seed)
        self.cem = ContrastiveExplainer(agent.actor)
        self.uncertainty = UncertaintyEstimator(agent, seed=seed)

    @classmethod
    def build(cls, agent: Agent, env, n_samples: int = 10_000, max_depth: int = 8, seed: int = 0,
              **kwargs) -> tuple["TrustXExplainer", float]:
        """Distil the surrogate from ``n_samples`` policy states; returns (explainer, held-out fidelity)."""
        states, actions = collect_policy_dataset(agent.policy_fn(), env, n_samples, seed=seed)
        split = int(0.8 * len(states))
        tree = SurrogateTree(FEATURE_NAMES, max_depth=max_depth).fit(states[:split], actions[:split])
        fidelity = tree.fidelity(states[split:], actions[split:])
        bg = states[np.random.default_rng(seed).choice(len(states), size=min(512, len(states)), replace=False)]
        return cls(agent, tree, bg, seed=seed, **kwargs), fidelity

    def _shap_model(self, x: np.ndarray) -> np.ndarray:
        # scalar output: how strongly the surrogate commands the executed direction
        return self.surrogate.predict(x) @ self._shap_target

    def explain(self, obs: np.ndarray, action: np.ndarray | None = None, level: str = "standard") -> Explanation:
        t0 = time.perf_counter()
        obs = np.asarray(obs, dtype=np.float32)
        action = self.agent.act(obs) if action is None else np.asarray(action)
        rule = self.surrogate.rule(obs)
        agreement = float(max(0.0, 1.0 - np.linalg.norm(action - rule.action) / 2.0))

        attribution = None
        if self.use_shap:
            self._shap_target = action / (np.linalg.norm(action) + 1e-8)
            attribution = self.shap.explain(obs)
        counterfactual = self.cem.explain(obs) if self.use_cem else None
        unc = self.uncertainty(obs)
        latency = (time.perf_counter() - t0) * 1e3

        quality = 0.25 + 0.35 * agreement
        quality += 0.2 if attribution is not None else 0.0
        quality += 0.2 if counterfactual is not None and counterfactual.success else 0.0
        return Explanation(str(classify(obs, action)), rule, attribution, counterfactual, unc, level,
                           agreement, latency, float(min(quality, 1.0)))
