"""Map continuous velocity commands onto human-readable manoeuvres.

Operators reason about *manoeuvres* ("climb", "veer left") rather than raw
velocity vectors, so the explainers and the contrastive mechanism work in
this discrete vocabulary. Directions are expressed relative to the heading
towards the active waypoint, which is part of the observation.
"""

from __future__ import annotations

import numpy as np
import torch

from trustx.envs.uav_env import FEATURE_NAMES

MANEUVERS: list[str] = ["advance", "veer_left", "veer_right", "climb", "descend", "hold"]
HOLD_SPEED = 0.25
_G = [FEATURE_NAMES.index(n) for n in ("goal_dx", "goal_dy")]


def prototypes(obs: np.ndarray) -> np.ndarray:
    """Unit direction for each non-hold manoeuvre, shape (..., 5, 3)."""
    g = np.asarray(obs, dtype=np.float64)[..., _G]
    h = g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-8)
    left = np.stack([-h[..., 1], h[..., 0]], axis=-1)
    zeros = np.zeros(h.shape[:-1] + (1,))
    s = 1.0 / np.sqrt(2.0)
    fwd = np.concatenate([h, zeros], -1)
    vl = np.concatenate([s * (h + left), zeros], -1)
    vr = np.concatenate([s * (h - left), zeros], -1)
    up = np.broadcast_to(np.array([0.0, 0.0, 1.0]), fwd.shape)
    return np.stack([fwd, vl, vr, up, -up], axis=-2)


def maneuver_logits(obs, action, temperature: float = 8.0, protos: np.ndarray | None = None):
    """Soft scores over MANEUVERS. Differentiable in ``action`` when it is a tensor.

    ``protos`` may be passed to reuse precomputed manoeuvre directions.
    """
    if protos is None:
        protos = prototypes(obs.detach().cpu().numpy() if torch.is_tensor(obs) else obs)
    if torch.is_tensor(action):
        p = torch.as_tensor(protos, dtype=action.dtype, device=action.device)
        norm = action.norm(dim=-1, keepdim=True)
        cos = (p * action.unsqueeze(-2)).sum(-1) / (norm + 1e-8)
        hold = temperature * (3.0 - 2.0 * norm / HOLD_SPEED)
        return torch.cat([temperature * cos, hold], dim=-1)
    a = np.asarray(action, dtype=np.float64)
    norm = np.linalg.norm(a, axis=-1, keepdims=True)
    cos = np.einsum("...kd,...d->...k", protos, a) / (norm + 1e-8)
    return np.concatenate([temperature * cos, temperature * (3.0 - 2.0 * norm / HOLD_SPEED)], axis=-1)


def classify(obs: np.ndarray, action: np.ndarray) -> np.ndarray | str:
    """Manoeuvre label(s) for a state/action pair (or a batch)."""
    idx = np.argmax(maneuver_logits(obs, action), axis=-1)
    if np.ndim(idx) == 0:
        return MANEUVERS[int(idx)]
    return np.array(MANEUVERS)[idx]
