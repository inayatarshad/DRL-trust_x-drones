# TRUST-X: Real-Time Explainable Decision-Making for RL-Based UAV Control

[![CI](https://github.com/inayatarshad/DRL-trust_x-drones/actions/workflows/ci.yml/badge.svg)](https://github.com/inayatarshad/DRL-trust_x-drones/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Reference implementation of **TRUST-X** ([paper](docs/TRUST-X_paper.pdf)). TRUST-X is an explainability and
trust-calibration layer for deep reinforcement learning (DRL) drone controllers. A TD3 agent learns to fly through a
3D city. For every decision, TRUST-X explains:

* **what** the drone does and under which conditions, as a *surrogate decision-tree rule*
* **why**, as the top-3 *SHAP-lite* feature attributions
* **why not something else**, as a *minimal counterfactual* from the contrastive explanation mechanism
* **how sure** the policy is, as a *decision confidence* with aleatoric/epistemic uncertainty

Explanations are generated in milliseconds. A *Bayesian trust model* reads the operator's feedback
(Trusted / Unclear / Disagree) and adapts how detailed the explanations are, switching to a safe mode when trust
or confidence drops.

```
--- t= 12.0s  pos=[41.3 37.9 22.4]  dist= 58.1 m  level=detailed  latency=9.8 ms
Action: CLIMB (confidence 91%)
Rule: IF range_front <= 0.31 AND range_nearest > 0.12 AND goal_dz <= 0.08 THEN CLIMB
Key factors: range_front (+0.41), goal_dz (-0.12), range_left (+0.07)
Contrast: If range_front were 6.2m higher, the UAV would advance instead of climb.
Operator tag: Trusted  ->  estimated trust 0.61 (true 0.58)
```
<sub>Illustrative console output of `scripts/explain_demo.py`. Run it to see your model's explanations.</sub>

---

## Contents

- [Architecture](#architecture)
- [Quick start](#quick-start)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Results](#results)
- [Repository layout](#repository-layout)
- [Relation to the paper](#relation-to-the-paper)
- [Citation](#citation)

## Architecture

```
 UAV environment ──s_t──▶ TD3 policy ──a_t──▶ environment          (or the operator's override)
                              │
                              ▼
            ┌──────── TRUST-X explainer (< 50 ms) ────────┐
            │ surrogate tree  →  IF-THEN rule  (R² fidelity)│
            │ SHAP-lite       →  top-3 factors              │
            │ contrastive     →  minimal counterfactual     │
            │ uncertainty     →  confidence, aleatoric/epist│
            └───────────────┬───────────────────────────────┘
                            ▼ explanation at the trust-selected level
                        operator ──tag Z_t∈{-1,0,1}──▶ Kalman trust filter ──▶ τ̂_t
                            │                                    │
                            └── override when risk ∧ low trust   └─▶ granularity + safe mode
```

A component-by-component mapping to the paper's equations is in [`docs/architecture.md`](docs/architecture.md).

| Module | Paper | What it does |
|---|---|---|
| [`trustx/envs/uav_env.py`](trustx/envs/uav_env.py) | Sec. V-A | 100×100×50 m city, wind up to 15 m/s, fog, battery, σ=0.05 sensor noise |
| [`trustx/agents/`](trustx/agents) | Eqs. 3–7 | TD3 (main policy), plus DDPG and PPO baselines, in PyTorch |
| [`trustx/explain/surrogate.py`](trustx/explain/surrogate.py) | Sec. IV-B, Eq. 19 | Decision tree distilled from 10k state–action pairs; rules + R² fidelity |
| [`trustx/explain/shap_lite.py`](trustx/explain/shap_lite.py) | Eqs. 11–12 | Monte Carlo permutation Shapley values (M=100), one batched call |
| [`trustx/explain/cem.py`](trustx/explain/cem.py) | Eqs. 13–14 | Sparse, bounded counterfactuals via proximal projected gradient descent |
| [`trustx/explain/uncertainty.py`](trustx/explain/uncertainty.py) | Figs. 6–7 | Aleatoric (sensor noise) vs. epistemic (twin-critic disagreement) |
| [`trustx/trust/bayesian.py`](trustx/trust/bayesian.py) | Eqs. 15–16 | Trust likelihood, exact grid filter and real-time Kalman filter |
| [`trustx/trust/operator.py`](trustx/trust/operator.py) | Eq. 17, Fig. 8 | Simulated skeptical / balanced / optimistic operators |
| [`trustx/evaluation.py`](trustx/evaluation.py) | Alg. 1, Eq. 18 | Human-in-the-loop missions, interventions, McNemar and paired t-tests |

## Quick start

```bash
git clone https://github.com/inayatarshad/DRL-trust_x-drones.git
cd DRL-trust_x-drones
pip install -e ".[dev]"
pytest -q                     # tests for the environment, agents, explainers and trust model
```

Trained checkpoints are included in `checkpoints/`, so you can run the explanation demo right away:

```bash
python -m scripts.explain_demo --profile skeptical
```

Use the library directly:

```python
from trustx.agents import load_agent
from trustx.envs import make_env
from trustx.framework import TrustXExplainer

env = make_env()
agent = load_agent("checkpoints/td3/model.pt")
explainer, fidelity = TrustXExplainer.build(agent, env)   # distils the surrogate tree

obs, _ = env.reset(seed=0)
explanation = explainer.explain(obs, level="detailed")
print(explanation.text())
```

## Reproducing the experiments

```bash
# 1. train the policies (CPU is fine; roughly 20-40 min each)
python -m scripts.train --config configs/td3.yaml
python -m scripts.train --config configs/ddpg.yaml
python -m scripts.train --config configs/ppo.yaml

# 2. Table I + ablations: 20 simulated operators x 50 missions per method
python -m scripts.evaluate --operators 20 --missions 50

# 3. figures
python -m scripts.make_figures
```

Any config value can be overridden from the command line, for example `train.episodes=200 env.n_waypoints=3`.

## Results

RESULTS_PLACEHOLDER

## Repository layout

```
trustx/                 library
  envs/                 UAV environment + safe fallback controller
  agents/               TD3, DDPG, PPO, replay buffer
  explain/              surrogate tree, SHAP-lite, CEM, manoeuvres, uncertainty
  trust/                Bayesian trust filters, simulated operators
  framework.py          TrustXExplainer (explanation E_t)
  evaluation.py         Algorithm 1 mission loop + statistics
  training.py           training loops
scripts/                train / evaluate / make_figures / explain_demo
configs/                YAML configs (environment + per-algorithm)
checkpoints/            trained models and training logs
results/                evaluation tables, JSON and figures
tests/                  pytest suite (run in CI)
docs/                   paper PDF and architecture notes
notebooks/              original figure prototyping notebook
```

## Relation to the paper

This repository is an open, runnable re-implementation of the TRUST-X framework. Some choices differ from the
paper's text, and each one is documented in [`docs/architecture.md`](docs/architecture.md):

* **Simulated operators.** The paper's human-centred metrics came from a user study (20 operators × 50 missions).
  Here, simulated operators follow the paper's own trust model (Eqs. 15 and 17). The trust, intervention and
  clarity numbers therefore show how the closed loop behaves under those assumptions. They are **not** evidence
  about real people.
* **Numbers are measured, not copied.** Every number in the Results section is produced by the code in this
  repository and can be regenerated with the commands above. Expect them to differ from Table I of the paper.
* **Observation design.** Obstacle distances are range-finder readings relative to the heading, and the goal is
  a tanh-scaled direction vector. This makes the navigation task learnable within a CPU budget.
* **Demonstration-seeded warm-up.** TD3 and DDPG fill their replay buffer with noisy demonstrations from a
  hand-written safe controller. Without them, the agents never observe the goal reward and learn only to hover.

## Citation

```bibtex
@inproceedings{akbar2025trustx,
  title  = {{TRUST-X}: Real-Time Explainable Decision-Making Framework for Reinforcement Learning-Based {UAV} Control},
  author = {Akbar, Hanzala and Arshad, Inayat and Rahman, Satwat and Aamir, Fatima and Fatima, Marrium and Khan, Haroon Ur Rashid},
  year   = {2025}
}
```

Released under the [MIT License](LICENSE).
