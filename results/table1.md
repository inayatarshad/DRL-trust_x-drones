| Method | Success (%) | 95% CI | Missions w/ intervention (%) | Interventions / mission | Trust delta (pp) | Clarity (1-5) | Latency mean / p95 (ms) | McNemar p vs TRUST-X |
|---|---|---|---|---|---|---|---|---|
| TD3 | 89.0 | 87.1-90.9 | 87.3 | 2.42 | -40.7 | 2.00 | - | 2.03e-15 |
| PPO | 88.4 | 86.4-90.4 | 71.9 | 1.79 | -30.0 | 2.00 | - | 1.85e-11 |
| DDPG | 7.1 | 5.5-8.7 | 90.0 | 3.79 | -43.5 | 2.00 | - | 2.58e-151 |
| TRUST-X | 77.7 | 75.1-80.3 | 6.4 | 0.14 | +27.1 | 4.57 | 12.2 / 43.5 | - |
| TRUST-X (PPO backbone) | 81.5 | 79.1-83.9 | 2.7 | 0.05 | +34.8 | 4.81 | 6.8 / 13.8 | 0.0212 |
| TRUST-X w/o CEM | 80.4 | 77.9-82.9 | 24.7 | 0.70 | +15.0 | 3.87 | 2.6 / 3.9 | 0.00432 |
| TRUST-X w/o SHAP-lite | 80.5 | 78.0-83.0 | 29.0 | 0.77 | +15.8 | 3.78 | 11.1 / 44.3 | 0.004 |
| TRUST-X w/o adaptive granularity | 77.1 | 74.5-79.7 | 5.6 | 0.12 | +29.4 | 4.57 | 11.9 / 45.4 | 0.519 |
| TRUST-X w/o safe mode | 81.1 | 78.7-83.5 | 8.5 | 0.18 | +26.5 | 4.52 | 11.0 / 41.9 | 0.00259 |

Surrogate decision-tree fidelity (held-out R^2, Eq. 19): TD3 policy 0.380, PPO policy 0.854
Each method: 1000 missions, identical seeds and simulated operators.

Policy-only benchmark (no operator in the loop):

| Policy | Success (%) | 95% CI | Crash (%) | Mean steps |
|---|---|---|---|---|
| TD3 | 83.2 | 79.9-86.5 | 5.6 | 77 |
| PPO | 82.8 | 79.5-86.1 | 5.8 | 73 |
| DDPG | 2.0 | 0.8-3.2 | 29.2 | 190 |
