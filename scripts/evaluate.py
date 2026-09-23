"""Run the human-in-the-loop comparison (Table I) and the ablation study.

    python -m scripts.evaluate --checkpoints checkpoints --out results
    python -m scripts.evaluate --operators 6 --missions 10     # quick run

Black-box baselines (TD3, PPO, DDPG) and TRUST-X (TD3 + explanations + trust
loop) fly identical missions in front of identical simulated operators.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from trustx.agents import load_agent
from trustx.envs import make_env
from trustx.evaluation import EvalConfig, Method, evaluate_autonomous, evaluate_method, mcnemar, paired_t
from trustx.framework import TrustXExplainer
from trustx.trust.operator import make_operator_pool
from trustx.utils.config import load_config
from trustx.utils.seeding import set_global_seed


def build_methods(ckpt: Path, env, surrogate_samples: int, ablations: bool) -> tuple[list[Method], float]:
    agents = {name: load_agent(str(ckpt / name / "model.pt"))
              for name in ("td3", "ppo", "ddpg") if (ckpt / name / "model.pt").exists()}
    if "td3" not in agents:
        raise SystemExit(f"no TD3 checkpoint under {ckpt}; run scripts/train.py first")
    methods = [Method(name.upper(), agent) for name, agent in agents.items()]
    explainer, fidelity = TrustXExplainer.build(agents["td3"], env, n_samples=surrogate_samples)
    methods.append(Method("TRUST-X", agents["td3"], explainer))
    if ablations:
        no_cem = TrustXExplainer(agents["td3"], explainer.surrogate, explainer.shap.background, use_cem=False)
        no_shap = TrustXExplainer(agents["td3"], explainer.surrogate, explainer.shap.background, use_shap=False)
        methods += [
            Method("TRUST-X w/o CEM", agents["td3"], no_cem),
            Method("TRUST-X w/o SHAP-lite", agents["td3"], no_shap),
            Method("TRUST-X w/o adaptive granularity", agents["td3"], explainer, adaptive=False),
            Method("TRUST-X w/o safe mode", agents["td3"], explainer, safe_mode=False),
        ]
    return methods, fidelity


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", default="checkpoints")
    ap.add_argument("--env-config", default="configs/env.yaml")
    ap.add_argument("--out", default="results")
    ap.add_argument("--operators", type=int, default=20, help="paper: 20 operators")
    ap.add_argument("--missions", type=int, default=50, help="missions per operator (paper: 50)")
    ap.add_argument("--surrogate-samples", type=int, default=10_000, help="paper: 10,000 state-action pairs")
    ap.add_argument("--autonomous-missions", type=int, default=500,
                    help="missions for the policy-only benchmark (no operator)")
    ap.add_argument("--no-ablations", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    set_global_seed(args.seed)
    env = make_env(load_config(args.env_config))
    methods, fidelity = build_methods(Path(args.checkpoints), env, args.surrogate_samples, not args.no_ablations)
    print(f"surrogate fidelity (held-out R^2): {fidelity:.3f}")
    operators = make_operator_pool(args.operators, seed=args.seed)

    autonomous = {}
    seeds = list(range(90_000, 90_000 + args.autonomous_missions))
    for m in methods:
        if m.name in ("TD3", "PPO", "DDPG"):
            autonomous[m.name] = evaluate_autonomous(env, m.agent, seeds)
            print(f"autonomous {m.name}: success={autonomous[m.name]['success_rate']:.1%} "
                  f"crash={autonomous[m.name]['crash_rate']:.1%}", flush=True)

    results = []
    for m in methods:
        r = evaluate_method(env, m, operators, args.missions, EvalConfig())
        results.append(r)
        print(f"{m.name:34s} success={r['success_rate']:.1%}  interventions/mission="
              f"{r['interventions_per_mission']:.2f}  trust delta={r['trust_delta']:+.3f}", flush=True)

    trustx = next(r for r in results if r["method"] == "TRUST-X")
    tests = {}
    for r in results:
        if r is trustx:
            continue
        chi2, p_mc = mcnemar(r["_success_vector"], trustx["_success_vector"])
        t, p_t = paired_t([o["interventions_per_mission"] for o in r["_operators"]],
                          [o["interventions_per_mission"] for o in trustx["_operators"]])
        tests[r["method"]] = {"mcnemar_chi2": chi2, "mcnemar_p": p_mc, "paired_t": t, "paired_t_p": p_t}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "setup": {"operators": args.operators, "missions_per_operator": args.missions,
                  "surrogate_samples": args.surrogate_samples, "surrogate_fidelity": fidelity},
        "methods": [{k: v for k, v in r.items() if not k.startswith("_")} | {"operators": r["_operators"]}
                    for r in results],
        "significance_vs_trustx": tests,
        "autonomous": autonomous,
    }
    (out / "evaluation.json").write_text(json.dumps(payload, indent=2, default=float))
    (out / "table1.md").write_text(markdown_table(results, fidelity, tests) + autonomous_table(autonomous))
    print((out / "table1.md").read_text())


def markdown_table(results: list[dict], fidelity: float, tests: dict) -> str:
    def fmt(v, spec):
        return "-" if v is None else format(v, spec)

    lines = ["| Method | Success (%) | 95% CI | Missions w/ intervention (%) | Interventions / mission "
             "| Trust delta (pp) | Clarity (1-5) | Latency mean / p95 (ms) | McNemar p vs TRUST-X |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lo, hi = r["success_ci"]
        lat = (f"{fmt(r['latency_ms_mean'], '.1f')} / {fmt(r['latency_ms_p95'], '.1f')}"
               if r["latency_ms_mean"] is not None else "-")
        p = tests.get(r["method"], {}).get("mcnemar_p")
        lines.append(
            f"| {r['method']} | {100 * r['success_rate']:.1f} | {100 * lo:.1f}-{100 * hi:.1f} "
            f"| {100 * r['missions_with_intervention']:.1f} | {r['interventions_per_mission']:.2f} "
            f"| {100 * r['trust_delta']:+.1f} | {r['clarity']:.2f} | {lat} | {fmt(p, '.3g')} |")
    lines.append("")
    lines.append(f"Surrogate decision-tree fidelity (held-out R^2, Eq. 19): {fidelity:.3f}")
    n = results[0]["missions"]
    lines.append(f"Each method: {n} missions, identical seeds and simulated operators.")
    return "\n".join(lines) + "\n"


def autonomous_table(autonomous: dict) -> str:
    lines = ["", "Policy-only benchmark (no operator in the loop):", "",
             "| Policy | Success (%) | 95% CI | Crash (%) | Mean steps |", "|---|---|---|---|---|"]
    for name, r in autonomous.items():
        lo, hi = r["success_ci"]
        lines.append(f"| {name} | {100 * r['success_rate']:.1f} | {100 * lo:.1f}-{100 * hi:.1f} "
                     f"| {100 * r['crash_rate']:.1f} | {r['mean_steps']:.0f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
