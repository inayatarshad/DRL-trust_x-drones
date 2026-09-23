"""Regenerate the paper's figures from trained checkpoints and training logs.

    python -m scripts.make_figures --checkpoints checkpoints --out results/figures

Figures (numbered as in the paper):
    fig2_explanation.png          rule + SHAP-lite + counterfactual for one decision
    fig3_trajectory.png           TRUST-X flight through the urban environment
    fig4_comparative_trajectories.png
    fig5_learning_curves.png      training return / success per algorithm
    fig6_confidence.png           decision confidence by flight mode + latency
    fig7_uncertainty.png          aleatoric vs epistemic uncertainty
    fig8_trust_evolution.png      operator trust over 50 sessions per profile
"""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from trustx.agents import load_agent  # noqa: E402
from trustx.envs import make_env  # noqa: E402
from trustx.evaluation import EvalConfig, Method, run_mission  # noqa: E402
from trustx.framework import TrustXExplainer  # noqa: E402
from trustx.trust.bayesian import KalmanTrustFilter  # noqa: E402
from trustx.trust.operator import PROFILES, SimulatedOperator  # noqa: E402
from trustx.utils.config import load_config  # noqa: E402

# categorical slots in fixed order (reference palette, light mode)
COLORS = {"TRUST-X": "#2a78d6", "TD3": "#eb6834", "PPO": "#1baf7a", "DDPG": "#eda100"}
MODE_COLORS = {"Cruise": "#2a78d6", "Obstacle Avoidance": "#eb6834", "Final Approach": "#1baf7a",
               "Manual Override": "#e34948", "Emergency Return": "#4a3aa7"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
})


def draw_city(ax, env):
    theta = np.linspace(0, 2 * np.pi, 24)
    for x, y, r, h in env.obstacles:
        xs, ys = x + r * np.cos(theta), y + r * np.sin(theta)
        for z in (0, h):
            ax.plot(xs, ys, z, color="#b9b8b2", linewidth=0.6)
        for k in range(0, 24, 6):
            ax.plot([xs[k]] * 2, [ys[k]] * 2, [0, h], color="#b9b8b2", linewidth=0.6)
    for i, wp in enumerate(env.waypoints):
        ax.scatter(*wp, marker="*", s=180, color="#e34948", edgecolor="white", zorder=5,
                   label="Waypoint" if i == 0 else None)
    ax.set(xlim=(0, env.cfg.size[0]), ylim=(0, env.cfg.size[1]), zlim=(0, env.cfg.size[2]),
           xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
    ax.view_init(elev=28, azim=-60)


def fig2_explanation(explainer, env, out):
    obs, _ = env.reset(seed=4242)
    # fly until something interesting (an obstacle within 10 m) is in view
    for _ in range(200):
        if min(obs[9], obs[13]) < 0.5:
            break
        obs, _, term, trunc, _ = env.step(explainer.agent.act(obs))
        if term or trunc:
            obs, _ = env.reset(seed=4243)
    e = explainer.explain(obs, level="detailed")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), gridspec_kw={"width_ratios": [1.2, 1, 1], "wspace": 0.55})
    ax = axes[0]
    ax.axis("off")
    ax.set_title("Surrogate rule", loc="left", fontweight="bold")
    conds = e.rule.conditions[-6:]
    text = "IF\n" + "\n".join(f"   {c}" for c in conds) + f"\nTHEN {e.maneuver.upper()}"
    ax.text(0, 0.95, text, family="monospace", va="top", fontsize=9, color=INK)
    ax.text(0, 0.02, f"confidence {e.uncertainty.confidence:.0%}  |  latency {e.latency_ms:.1f} ms",
            color=MUTED, fontsize=9)

    ax = axes[1]
    top = e.attribution.top(6)[::-1]
    vals = [v for _, v in top]
    ax.barh([n for n, _ in top], vals, color=["#2a78d6" if v >= 0 else "#eb6834" for v in vals], height=0.6)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_title("SHAP-lite attribution", loc="left", fontweight="bold")
    ax.set_xlabel("contribution to executed command")

    ax = axes[2]
    ch = e.counterfactual.changes()[:6][::-1]
    if ch:
        ax.barh([n for n, _ in ch], [d for _, d in ch], color="#1baf7a", height=0.6)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_title("Minimal counterfactual", loc="left", fontweight="bold")
    ax.set_xlabel("normalised change")
    ax.text(0.0, -0.28, e.counterfactual.text(), transform=ax.transAxes, fontsize=8.5, color=MUTED, wrap=True)
    fig.savefig(out / "fig2_explanation.png")
    plt.close(fig)
    return e


def fig3_4_trajectories(methods, env, out, seed=777):
    op = SimulatedOperator("balanced", seed=0)
    traces = {}
    for m in methods:
        tf = KalmanTrustFilter() if m.explains else None
        r = run_mission(env, m, copy.deepcopy(op), seed, EvalConfig(), tf, record=True)
        traces[m.name] = (np.array(r.trace["pos"]), r)

    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(projection="3d")
    env.reset(seed=seed)
    draw_city(ax, env)
    pos, r = traces["TRUST-X"]
    ax.plot(*pos.T, color=COLORS["TRUST-X"], linewidth=2.5,
            label=f"TRUST-X ({'success' if r.success else 'crash' if r.crash else 'timeout'})")
    ax.scatter(*pos[0], color=INK, s=30, label="Start")
    ax.set_title("UAV trajectory under TRUST-X", loc="left")
    ax.legend(loc="upper left")
    fig.savefig(out / "fig3_trajectory.png")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(projection="3d")
    draw_city(ax, env)
    for name, (pos, r) in traces.items():
        status = "success" if r.success else "crash" if r.crash else "timeout"
        ax.plot(*pos.T, color=COLORS.get(name, MUTED), label=f"{name} ({status})")
    ax.set_title("Comparative trajectories, same mission", loc="left")
    ax.legend(loc="upper left")
    fig.savefig(out / "fig4_comparative_trajectories.png")
    plt.close(fig)


def fig5_learning_curves(ckpt: Path, out: Path, window: int = 25):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    for name in ("td3", "ppo", "ddpg"):
        log = ckpt / name / "train_log.csv"
        if not log.exists():
            continue
        rows = list(csv.DictReader(open(log, encoding="utf-8")))
        ep = np.array([int(r["episode"]) for r in rows])
        ret = np.array([float(r["return"]) for r in rows])
        succ = np.array([int(r["success"]) for r in rows], dtype=float)
        k = np.ones(window) / window
        label = name.upper() + (" (TRUST-X policy)" if name == "td3" else "")
        color = COLORS["TRUST-X"] if name == "td3" else COLORS[name.upper()]
        a1.plot(ep[window - 1:], np.convolve(ret, k, "valid"), color=color, label=label)
        a2.plot(ep[window - 1:], 100 * np.convolve(succ, k, "valid"), color=color, label=label)
    a1.set(xlabel="Training episode", ylabel=f"Return ({window}-episode mean)")
    a1.set_title("Learning curves", loc="left")
    a2.set(xlabel="Training episode", ylabel="Success rate (%)", ylim=(0, 100))
    a2.set_title("Mission success during training", loc="left")
    a1.legend()
    fig.savefig(out / "fig5_learning_curves.png")
    plt.close(fig)


def fig6_7_confidence(trustx: Method, env, out, seed=31):
    op = SimulatedOperator("skeptical", seed=3)
    r = run_mission(env, trustx, op, seed, EvalConfig(review_every=1), KalmanTrustFilter(), record=True)
    t_conf, conf = zip(*r.trace["confidence"])
    modes = [r.trace["mode"][t] for t in t_conf]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    time_s = np.array(t_conf) * env.cfg.dt
    a1.plot(time_s, conf, color=MUTED, linewidth=1.0, zorder=1)
    for mode, color in MODE_COLORS.items():
        idx = [i for i, m in enumerate(modes) if m == mode]
        if idx:
            a1.scatter(time_s[idx], np.array(conf)[idx], color=color, s=16, label=mode, zorder=2)
    a1.set(ylabel="Decision confidence", ylim=(0, 1.05))
    a1.set_title("Real-time decision confidence by flight mode", loc="left")
    a1.legend(ncol=5, loc="lower left", fontsize=8)
    _, lat = zip(*r.trace["latency"])
    a2.plot(time_s, lat, color="#2a78d6")
    a2.axhline(50, color="#e34948", linestyle="--", linewidth=1.2)
    a2.text(time_s[0], 51, "50 ms budget", color="#e34948", fontsize=8, va="bottom")
    a2.set(xlabel="Mission time (s)", ylabel="Explanation latency (ms)")
    fig.savefig(out / "fig6_confidence.png")
    plt.close(fig)

    _, ale = zip(*r.trace["aleatoric"])
    _, epi = zip(*r.trace["epistemic"])
    ale, epi = np.array(ale), np.array(epi)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.fill_between(time_s, 0, ale, color="#2a78d6", alpha=0.35, linewidth=0, label="Aleatoric (data)")
    ax.fill_between(time_s, ale, ale + epi, color="#eb6834", alpha=0.35, linewidth=0, label="Epistemic (model)")
    ax.plot(time_s, np.hypot(ale, epi), color=INK, linewidth=1.5, label="Total")
    ax.axhline(0.3, color="#e34948", linestyle="--", linewidth=1.2, label="Safety threshold 0.3")
    ax.set(xlabel="Mission time (s)", ylabel="Uncertainty")
    ax.set_title("Uncertainty decomposition during a mission", loc="left")
    ax.legend(ncol=4, loc="upper left", fontsize=8)
    fig.savefig(out / "fig7_uncertainty.png")
    plt.close(fig)


def fig8_trust(methods_by_name, env, out, sessions=50):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    for ax, profile in zip(axes, PROFILES):
        for name in ("TD3", "TRUST-X"):
            m = methods_by_name[name]
            op = SimulatedOperator(profile, seed=11)
            tf = KalmanTrustFilter() if m.explains else None
            trust = [op.trust]
            for s in range(sessions):
                run_mission(env, m, op, 50_000 + s, EvalConfig(), tf)
                trust.append(op.trust)
            ax.plot(range(sessions + 1), trust, color=COLORS[name], label="TD3 (black box)" if name == "TD3" else name)
        ax.set_title(f"{profile.capitalize()} operator", loc="left")
        ax.set(xlabel="Mission session", ylim=(0, 1))
    axes[0].set_ylabel("Operator trust")
    axes[0].legend()
    fig.savefig(out / "fig8_trust_evolution.png")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", default="checkpoints")
    ap.add_argument("--env-config", default="configs/env.yaml")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--surrogate-samples", type=int, default=10_000)
    args = ap.parse_args(argv)
    ckpt, out = Path(args.checkpoints), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = make_env(load_config(args.env_config))

    agents = {n.upper(): load_agent(str(ckpt / n / "model.pt"))
              for n in ("td3", "ppo", "ddpg") if (ckpt / n / "model.pt").exists()}
    explainer, fid = TrustXExplainer.build(agents["TD3"], env, n_samples=args.surrogate_samples)
    print(f"surrogate fidelity {fid:.3f}")
    methods = [Method(n, a) for n, a in agents.items()] + [Method("TRUST-X", agents["TD3"], explainer)]
    by_name = {m.name: m for m in methods}

    e = fig2_explanation(explainer, env, out)
    print(e.text())
    fig3_4_trajectories(methods, env, out)
    fig5_learning_curves(ckpt, out)
    fig6_7_confidence(by_name["TRUST-X"], env, out)
    fig8_trust(by_name, env, out)
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
