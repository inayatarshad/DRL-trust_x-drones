"""Hand-written controllers used as the operator's manual override."""

from __future__ import annotations

import numpy as np

from trustx.envs.uav_env import FEATURE_NAMES

_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}


class SafeFallbackController:
    """Conservative "operator takes the stick" controller.

    Flies towards the waypoint at reduced speed and climbs whenever the
    forward-looking range finders report something close.
    """

    def __init__(self, speed: float = 0.6, climb_threshold: float = 0.35, clearance: float = 0.15):
        self.speed = speed
        self.climb_threshold = climb_threshold
        self.clearance = clearance

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        goal = obs[[_IDX["goal_dx"], _IDX["goal_dy"], _IDX["goal_dz"]]].astype(np.float64)
        direction = goal / (np.linalg.norm(goal) + 1e-8)
        action = self.speed * direction
        front = min(obs[_IDX["range_front"]], obs[_IDX["range_left"]], obs[_IDX["range_right"]])
        if front < self.climb_threshold:
            action[:2] *= max(front / self.climb_threshold - 0.3, 0.0)
            action[2] = 1.0
        elif min(obs[_IDX["range_down"]], obs[_IDX["range_nearest"]]) < self.clearance:
            action[2] = 0.6
        return np.clip(action, -1.0, 1.0).astype(np.float32)
