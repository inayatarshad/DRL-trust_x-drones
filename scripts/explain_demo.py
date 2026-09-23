"""Fly one mission and print TRUST-X explanations as an operator would see them.

    python -m scripts.explain_demo --checkpoint checkpoints/td3/model.pt --profile skeptical
"""

from __future__ import annotations

import argparse

import numpy as np

from trustx.agents import load_agent
from trustx.envs import make_env
from trustx.framework import TrustXExplainer
from trustx.trust.bayesian import KalmanTrustFilter, explanation_level
from trustx.trust.operator import PROFILES, Observation, SimulatedOperator
from trustx.utils.config import load_config

TAG_NAMES = {1: "Trusted", 0: "Unclear", -1: "Disagree"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoints/td3/model.pt")
    ap.add_argument("--env-config", default="configs/env.yaml")
    ap.add_argument("--profile", choices=list(PROFILES), default="balanced")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--every", type=int, default=6, help="print an explanation every N steps")
    ap.add_argument("--surrogate-samples", type=int, default=5000)
    args = ap.parse_args(argv)

    env = make_env(load_config(args.env_config))
    agent = load_agent(args.checkpoint)
    explainer, fidelity = TrustXExplainer.build(agent, env, n_samples=args.surrogate_samples)
    print(f"Surrogate tree fidelity (R^2): {fidelity:.3f}\n")
    operator = SimulatedOperator(args.profile, seed=args.seed)
    tf = KalmanTrustFilter()

    obs, info = env.reset(seed=args.seed)
    last, dist0, done, t = None, np.linalg.norm(info["waypoint"] - info["position"]), False, 0
    while not done:
        action = agent.act(obs)
        if t % args.every == 0:
            level = explanation_level(tf.mean)
            e = explainer.explain(obs, action, level)
            dist = np.linalg.norm(info["waypoint"] - info["position"])
            tag, _ = operator.review(Observation(e.quality, level, last is not None and e.maneuver != last,
                                                 info["clearance"] < env.cfg.safety_margin, False, dist0 - dist))
            tf.update(tag)
            print(f"--- t={t * env.cfg.dt:5.1f}s  pos={np.round(info['position'], 1)}  "
                  f"dist={dist:5.1f} m  level={level}  latency={e.latency_ms:.1f} ms")
            print(e.text())
            print(f"Operator tag: {TAG_NAMES[tag]}  ->  estimated trust {tf.mean:.2f} (true {operator.trust:.2f})\n")
            last, dist0 = e.maneuver, dist
        obs, _, term, trunc, info = env.step(action)
        done, t = term or trunc, t + 1
    outcome = "SUCCESS" if info["success"] else "CRASH" if info["crash"] else "TIMEOUT"
    print(f"Mission finished after {t * env.cfg.dt:.1f}s: {outcome}")


if __name__ == "__main__":
    main()
