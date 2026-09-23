"""Human-in-the-loop mission evaluation (Section V).

Each *method* is a policy plus an optional TRUST-X layer. Every method flies
the same missions (identical environment seeds) in front of the same pool of
simulated operators, so results are paired across methods. One operator flies
a *session* of consecutive missions and keeps their trust between missions.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from trustx.envs import SafeFallbackController
from trustx.envs.uav_env import FEATURE_NAMES
from trustx.explain.maneuvers import classify
from trustx.framework import TrustXExplainer
from trustx.trust.bayesian import KalmanTrustFilter, explanation_level
from trustx.trust.operator import Observation, SimulatedOperator

_I = {n: i for i, n in enumerate(FEATURE_NAMES)}


@dataclass
class Method:
    name: str
    agent: object
    explainer: TrustXExplainer | None = None
    adaptive: bool = True          # adapt explanation granularity to estimated trust
    safe_mode: bool = True         # slow down when estimated trust / confidence is low

    @property
    def explains(self) -> bool:
        return self.explainer is not None


@dataclass
class MissionResult:
    success: bool
    crash: bool
    interventions: int
    steps: int
    trust_before: float
    trust_after: float
    explanation_quality: list[float] = field(default_factory=list)
    latency_ms: list[float] = field(default_factory=list)
    trace: dict[str, list] = field(default_factory=dict)


@dataclass
class EvalConfig:
    review_every: int = 4          # steps between operator reviews (2 s at dt=0.5)
    risk_distance: float = 6.0     # metres; hazards closer than this count as risky
    override_steps: int = 6        # steps the operator keeps manual control
    uncertainty_threshold: float = 0.3
    safe_mode_speed: float = 0.7


def run_mission(env, method: Method, operator: SimulatedOperator, seed: int,
                cfg: EvalConfig | None = None, trust_filter: KalmanTrustFilter | None = None,
                record: bool = False) -> MissionResult:
    cfg = cfg or EvalConfig()
    fallback = SafeFallbackController()
    obs, info = env.reset(seed=seed)
    res = MissionResult(False, False, 0, 0, operator.trust, operator.trust)
    trace: dict[str, list] = {k: [] for k in ("pos", "confidence", "aleatoric", "epistemic", "mode",
                                              "trust", "trust_hat", "latency")} if record else {}
    override_left = 0
    last_maneuver = None
    dist_at_review = np.linalg.norm(info["waypoint"] - info["position"])
    min_clear = np.inf
    safe = False
    done = False
    t = 0
    while not done:
        action = method.agent.act(obs)
        if method.explains and safe:
            action = action * cfg.safe_mode_speed
        manual = override_left > 0
        if manual:
            action = fallback(obs)
            override_left -= 1

        hazard = min(obs[_I["range_front"]], obs[_I["range_nearest"]]) * env.cfg.sensor_range
        if t % cfg.review_every == 0 and not manual:
            maneuver = str(classify(obs, action))
            expl = None
            if method.explains:
                level = explanation_level(trust_filter.mean) if method.adaptive else "standard"
                expl = method.explainer.explain(obs, action, level=level)
                res.explanation_quality.append(expl.quality)
                res.latency_ms.append(expl.latency_ms)
            risky = hazard < cfg.risk_distance
            dist = np.linalg.norm(info["waypoint"] - info["position"])
            seen = Observation(
                explanation_quality=expl.quality if expl else 0.0,
                explanation_level=expl.level if expl else None,
                maneuver_changed=last_maneuver is not None and maneuver != last_maneuver,
                near_miss=min_clear < env.cfg.safety_margin,
                risky=risky,
                progress=float(dist_at_review - dist),
            )
            tag, override = operator.review(seen)
            if method.explains:
                trust_filter.update(tag)
                unc = expl.uncertainty.total
                safe = method.safe_mode and (trust_filter.mean < 0.5 or unc > cfg.uncertainty_threshold)
            if override:
                res.interventions += 1
                override_left = cfg.override_steps
            last_maneuver, dist_at_review, min_clear = maneuver, dist, np.inf
            if record and expl is not None:
                trace["confidence"].append((t, expl.uncertainty.confidence))
                trace["aleatoric"].append((t, expl.uncertainty.aleatoric))
                trace["epistemic"].append((t, expl.uncertainty.epistemic))
                trace["latency"].append((t, expl.latency_ms))
                trace["trust_hat"].append((t, trust_filter.mean))
        if record:
            trace["pos"].append(env.pos.copy())
            trace["trust"].append((t, operator.trust))
            trace["mode"].append(_flight_mode(manual, hazard, cfg, info, env))

        obs, _, terminated, truncated, info = env.step(action)
        min_clear = min(min_clear, info["clearance"])
        done = terminated or truncated
        t += 1

    res.success, res.crash, res.steps = bool(info["success"]), bool(info["crash"]), t
    res.trust_after = operator.trust
    res.trace = trace
    return res


def _flight_mode(manual: bool, hazard: float, cfg: EvalConfig, info: dict, env) -> str:
    if manual:
        return "Manual Override"
    if info["battery"] < 0.2:
        return "Emergency Return"
    if hazard < cfg.risk_distance:
        return "Obstacle Avoidance"
    if np.linalg.norm(info["waypoint"] - info["position"]) < 15.0:
        return "Final Approach"
    return "Cruise"


def evaluate_method(env, method: Method, operators: list[SimulatedOperator], missions_per_operator: int,
                    cfg: EvalConfig | None = None, base_seed: int = 10_000) -> dict:
    """Run every operator's session for one method. Operators are deep-copied so
    each method sees the same operators with the same random streams."""
    per_mission, per_operator = [], []
    for k, op0 in enumerate(operators):
        op = copy.deepcopy(op0)
        tf = KalmanTrustFilter(prior_mean=0.5) if method.explains else None
        start = op.trust
        results = [run_mission(env, method, op, base_seed + k * 1000 + m, cfg, tf)
                   for m in range(missions_per_operator)]
        per_mission += results
        per_operator.append({
            "profile": op.profile.name,
            "trust_start": start,
            "trust_end": op.trust,
            "interventions_per_mission": float(np.mean([r.interventions for r in results])),
            "success_rate": float(np.mean([r.success for r in results])),
        })
    return summarise(method.name, per_mission, per_operator)


def normal_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Normal-approximation binomial confidence interval, Eq. (18)."""
    half = z * np.sqrt(max(p * (1 - p), 0.0) / max(n, 1))
    return float(max(0.0, p - half)), float(min(1.0, p + half))


def summarise(name: str, missions: list[MissionResult], operators: list[dict]) -> dict:
    n = len(missions)
    succ = np.array([m.success for m in missions])
    p = float(succ.mean())
    lat = np.concatenate([m.latency_ms for m in missions]) if any(m.latency_ms for m in missions) else np.array([])
    q = np.concatenate([m.explanation_quality for m in missions]) if lat.size else np.array([])
    trust_delta = np.array([o["trust_end"] - o["trust_start"] for o in operators])
    return {
        "method": name,
        "missions": n,
        "success_rate": p,
        "success_ci": normal_ci(p, n),
        "crash_rate": float(np.mean([m.crash for m in missions])),
        "missions_with_intervention": float(np.mean([m.interventions > 0 for m in missions])),
        "interventions_per_mission": float(np.mean([m.interventions for m in missions])),
        "trust_delta": float(trust_delta.mean()),
        "trust_final": float(np.mean([o["trust_end"] for o in operators])),
        # simulated clarity rating: 1 (opaque) .. 5 (perfectly clear); a black box scores 1 + 4 * 0.25
        "clarity": float(1.0 + 4.0 * (q.mean() if q.size else 0.25)),
        "latency_ms_mean": float(lat.mean()) if lat.size else None,
        "latency_ms_p95": float(np.percentile(lat, 95)) if lat.size else None,
        "_success_vector": succ.astype(int).tolist(),
        "_operators": operators,
    }


def mcnemar(a: list[int], b: list[int]) -> tuple[float, float]:
    """Continuity-corrected McNemar test on paired binary outcomes."""
    a, b = np.asarray(a), np.asarray(b)
    n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    if n01 + n10 == 0:
        return 0.0, 1.0
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return float(chi2), float(stats.chi2.sf(chi2, df=1))


def paired_t(baseline: list[float], treatment: list[float]) -> tuple[float, float]:
    """One-sided paired t-test H1: treatment < baseline (fewer interventions)."""
    d = np.asarray(treatment) - np.asarray(baseline)
    if np.allclose(d, d[0]):
        return 0.0, 1.0
    res = stats.ttest_rel(treatment, baseline, alternative="less")
    return float(res.statistic), float(res.pvalue)


def evaluate_autonomous(env, agent, seeds: list[int]) -> dict:
    """Policy-only benchmark: no operator, no explanations, no overrides."""
    success, crash, steps = [], [], []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(agent.act(obs))
            done = terminated or truncated
        success.append(info["success"])
        crash.append(info["crash"])
        steps.append(env.steps)
    p = float(np.mean(success))
    return {"success_rate": p, "success_ci": normal_ci(p, len(seeds)),
            "crash_rate": float(np.mean(crash)), "mean_steps": float(np.mean(steps))}
