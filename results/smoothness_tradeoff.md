# Explainability vs. performance: TD3 smoothness sweep

TD3 trained with CAPS-style temporal + spatial smoothness regularisation
(`agent.smooth_temporal = agent.smooth_spatial = λ`, 800 episodes, seed 0).
Success/crash: 200 held-out missions, no operator in the loop. Jitter: mean
absolute step-to-step change of the commanded velocity. Fidelity: held-out R²
(Eq. 19) of a depth-8 surrogate tree distilled from 8,000 policy states.

| Policy | λ | Checkpoint | Success (%) | Crash (%) | Jitter | Surrogate R² |
|---|---|---|---|---|---|---|
| TD3 (released) | 0.0 | final | 80.5 | 7.5 | 0.37 | 0.38 |
| TD3 | 0.1 | ep 600 | 42.0 | 12.5 | 0.13 | 0.57 |
| TD3 | 0.1 | final | 23.0 | 16.0 | 0.11 | 0.75 |
| TD3 | 0.2 | ep 700 | 15.0 | 31.0 | 0.08 | 0.80 |
| TD3 | 0.2 | final | 20.0 | 29.0 | 0.08 | 0.72 |
| TD3 | 0.5 | final | 14.0 | 42.5 | 0.06 | 0.82 |
| PPO (released) | – | final | 82.0 | 7.0 | 0.12 | 0.85 |

Jitter for the two released policies was measured on noise-free rollouts; for the
sweep rows, on rollouts with 0.1 exploration noise, so treat small jitter differences
between the two groups with care.

Takeaways

* Smoothness makes TD3 far easier to explain (R² 0.38 → 0.8) but, with this
  training budget, severely hurts mission success. That is a real
  explainability/performance trade-off.
* PPO's Gaussian policy is naturally smoother and reaches both high success
  and high fidelity. The TRUST-X layer is policy-agnostic, so the evaluation
  reports TRUST-X on both a TD3 and a PPO backbone.
