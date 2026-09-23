"""Decision confidence and uncertainty decomposition (Figs. 6-7).

* **Aleatoric** (data) uncertainty: spread of the policy's action when the
  observation is re-sampled under the environment's sensor-noise model. It is
  high in fog, near obstacles and in gusty wind.
* **Epistemic** (model) uncertainty: disagreement between TD3's twin critics
  on the chosen action, normalised by the magnitude of the value estimate.
  Critics trained on the same data disagree most in rarely visited states.

Both components are squashed to ``[0, 1)`` and combined as
``total = sqrt(aleatoric^2 + epistemic^2)``; ``confidence = 1 - min(total, 1)``.
The paper's safety threshold of 0.3 on total uncertainty can then trigger a
safe-mode or an operator hand-over.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trustx.envs.uav_env import FEATURE_NAMES

_NOISY = np.array([n.startswith("range_") for n in FEATURE_NAMES])


@dataclass
class Uncertainty:
    aleatoric: float
    epistemic: float

    @property
    def total(self) -> float:
        return float(np.hypot(self.aleatoric, self.epistemic))

    @property
    def confidence(self) -> float:
        return float(1.0 - min(self.total, 1.0))


class UncertaintyEstimator:
    def __init__(self, agent, noise_std: float = 0.05, n_samples: int = 16, aleatoric_scale: float = 4.0,
                 epistemic_scale: float = 2.0, seed: int = 0):
        self.agent = agent
        self.noise_std = noise_std
        self.n_samples = n_samples
        self.a_scale, self.e_scale = aleatoric_scale, epistemic_scale
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs: np.ndarray, visibility: float | None = None) -> Uncertainty:
        obs = np.asarray(obs, dtype=np.float32)
        vis = float(obs[FEATURE_NAMES.index("visibility")]) if visibility is None else visibility
        sigma = self.noise_std * (1.0 + (1.0 - vis))
        noisy = np.repeat(obs[None], self.n_samples, axis=0)
        noisy[:, _NOISY] += self.rng.normal(0.0, sigma, size=(self.n_samples, _NOISY.sum()))
        noisy[:, _NOISY] = np.clip(noisy[:, _NOISY], 0.0, 1.0)
        actions = self.agent.act(noisy)
        aleatoric = float(np.tanh(self.a_scale * actions.std(axis=0).mean()))

        epistemic = 0.0
        q = self.agent.q_values(obs[None], self.agent.act(obs[None]))
        if q is not None and q.shape[0] > 1:
            spread = float(q[:, 0].std())
            epistemic = float(np.tanh(self.e_scale * spread / (abs(float(q[:, 0].mean())) + 10.0)))
        return Uncertainty(aleatoric, epistemic)
