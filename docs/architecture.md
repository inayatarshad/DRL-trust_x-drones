# TRUST-X architecture

This document maps every component of the paper to the code.

```
                ┌──────────────────────────── UAVNavigationEnv (trustx/envs) ───────────────────────────┐
                │ 100x100x50 m city, wind 0-15 m/s, visibility, battery, sensor noise sigma=0.05         │
                └───────────────┬───────────────────────────────────────────────────────▲───────────────┘
                          obs s_t │                                                       │ a_t (policy or override)
                                  ▼                                                       │
   ┌────────────── TD3 policy mu_theta (trustx/agents/td3.py) ──────────────┐              │
   │ twin critics, target smoothing, delayed actor updates (Eqs. 3-7)        │── a_t ──┐    │
   └─────────────────────────────────────────────────────────────────────────┘         │    │
                                  │                                                  ▼    │
   ┌──────────────────────── TrustXExplainer (trustx/framework.py) ──────────────────────┐  │
   │  SurrogateTree  ─ IF-THEN rule, R^2 fidelity (Eq. 19)      trustx/explain/surrogate  │  │
   │  ShapLite       ─ top-3 Monte Carlo Shapley factors (Eq. 12) trustx/explain/shap_lite│  │
   │  Contrastive    ─ minimal counterfactual via proximal PGD (Eqs. 13-14)   .../cem     │  │
   │  Uncertainty    ─ aleatoric vs epistemic, decision confidence  .../uncertainty      │  │
   └────────────────────────────────────┬─────────────────────────────────────────────────┘  │
                            E_t (level) │                                                     │
                                        ▼                                                     │
   ┌──────────── Operator (trustx/trust/operator.py) ────────────┐   tag Z_t ∈ {-1,0,1}       │
   │ latent trust, Eq. 17 dynamics; tags sampled from Eq. 15      │──────────────┐             │
   │ grabs the stick when risk ∧ low trust  ─────────────────────────── override ───────────────┘
   └──────────────────────────────────────────────────────────────┘              ▼
                                                    ┌─ KalmanTrustFilter (trustx/trust/bayesian.py) ─┐
                                                    │ tau_hat, Var(tau) (Eq. 16)                      │
                                                    │ → explanation granularity (concise/…/detailed)  │
                                                    │ → safe mode when tau_hat < 0.5 or U > 0.3       │
                                                    └─────────────────────────────────────────────────┘
```

| Paper | Code |
|---|---|
| POMDP, observation space (Sec. III, V-A) | `trustx/envs/uav_env.py` |
| TD3 (Eqs. 3-7), PPO and DDPG baselines | `trustx/agents/` |
| Surrogate decision tree, fidelity bound, Eq. 19 | `trustx/explain/surrogate.py` |
| SHAP-lite (Eqs. 11-12) | `trustx/explain/shap_lite.py` |
| Contrastive explanation mechanism (Eqs. 13-14) | `trustx/explain/cem.py` |
| Trust likelihood and Bayesian filter (Eqs. 15-16) | `trustx/trust/bayesian.py` |
| Trust dynamics (Eq. 17), operator profiles (Fig. 8) | `trustx/trust/operator.py` |
| Algorithm 1 (explain & adapt loop) | `trustx/evaluation.py::run_mission` |
| Binomial CI (Eq. 18), McNemar, paired t-test | `trustx/evaluation.py` |
| Uncertainty decomposition (Figs. 6-7) | `trustx/explain/uncertainty.py` |

## Design decisions and deviations

* **Range sensors instead of "five nearest obstacles".** A list of distances
  without bearings is not enough to steer around buildings. The five
  obstacle features are range-finder readings (front, ±45°, down) plus the
  distance to the nearest surface, all relative to the heading towards the
  waypoint.
* **Goal encoding.** The waypoint vector is a unit direction scaled by
  `tanh(distance / 20 m)`, which keeps the signal informative both far away
  and close to arrival.
* **Manoeuvre vocabulary.** Operators reason about *manoeuvres*, so the
  continuous velocity command is mapped to advance / veer left / veer right /
  climb / descend / hold. The contrastive mechanism asks "why *a* rather than
  *a'*?" in that vocabulary, using differentiable soft scores so projected
  gradient descent works on the continuous policy.
* **Kalman filter.** Eq. 15 is not Gaussian, so the real-time filter uses a
  moment-matching (assumed-density) update. It is tested against the exact
  grid filter.
* **Simulated operators.** The human study is replaced by simulated
  operators whose trust follows Eq. 17 and whose tags follow Eq. 15. The
  human-centred metrics (trust delta, interventions, clarity) therefore
  reflect the assumptions in `trustx/trust/operator.py`. They show how the
  loop behaves; they are not evidence about real people.
