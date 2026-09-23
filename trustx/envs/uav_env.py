"""3D urban UAV navigation environment (Section V-A of the paper).

The UAV flies inside a ``100 x 100 x 50 m`` volume populated with cylindrical
buildings. It must reach one or more waypoints while avoiding obstacles under
wind disturbances, reduced visibility and a finite battery.

Observation (18-D, roughly normalised to ``[-1, 1]``)::

    pos_x, pos_y, pos_z            UAV position
    goal_dx, goal_dy, goal_dz      vector to the active waypoint
    vel_x, vel_y, vel_z            UAV velocity
    range_front, range_left,       horizontal range finders (0 = contact,
    range_right                    1 = nothing within sensor range), aimed
                                   along / 45 deg either side of the heading
                                   towards the goal
    range_down                     distance to the roof or ground below
    range_nearest                  3-D distance to the closest surface
    visibility                     0 (fog) .. 1 (clear)
    wind_speed, wind_dir           wind magnitude and direction
    battery                        state of charge, 0 .. 1

Action (3-D, ``[-1, 1]``): commanded velocity in the world frame, scaled by
``max_speed``. The vehicle tracks the command with a first-order lag and is
pushed by the wind.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

FEATURE_NAMES: list[str] = [
    "pos_x", "pos_y", "pos_z",
    "goal_dx", "goal_dy", "goal_dz",
    "vel_x", "vel_y", "vel_z",
    "range_front", "range_left", "range_right", "range_down", "range_nearest",
    "visibility", "wind_speed", "wind_dir", "battery",
]

RAY_ANGLES = np.deg2rad([0.0, 45.0, -45.0])


@dataclass
class UAVConfig:
    size: tuple[float, float, float] = (100.0, 100.0, 50.0)
    dt: float = 0.5
    max_steps: int = 250
    max_speed: float = 5.0
    response: float = 0.6          # first-order tracking gain per step
    uav_radius: float = 0.5
    sensor_range: float = 20.0
    safety_margin: float = 3.0     # proximity penalty radius (paper: -0.5)
    goal_radius: float = 3.0
    n_obstacles: tuple[int, int] = (10, 20)
    obstacle_radius: tuple[float, float] = (2.0, 6.0)
    obstacle_height: tuple[float, float] = (15.0, 45.0)
    wind_speed: tuple[float, float] = (0.0, 15.0)
    wind_drift: float = 0.08       # fraction of wind velocity imparted on the UAV
    visibility: tuple[float, float] = (0.3, 1.0)
    sensor_noise: float = 0.05     # paper: sigma = 0.05
    n_waypoints: int = 1
    battery_drain: float = 0.0015  # per step at hover, scaled with effort
    # reward shaping
    progress_weight: float = 1.0
    proximity_penalty: float = -0.5
    step_penalty: float = -0.05
    goal_reward: float = 100.0
    crash_penalty: float = -100.0
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict | None) -> "UAVConfig":
        d = dict(d or {})
        known = {k: d.pop(k) for k in list(d) if k in cls.__dataclass_fields__}
        for k in ("size", "n_obstacles", "obstacle_radius", "obstacle_height", "wind_speed", "visibility"):
            if k in known:
                known[k] = tuple(known[k])
        return cls(**known, extra=d)


class UAVNavigationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: UAVConfig | dict | None = None):
        super().__init__()
        self.cfg = config if isinstance(config, UAVConfig) else UAVConfig.from_dict(config)
        self.observation_space = spaces.Box(-1.5, 1.5, shape=(len(FEATURE_NAMES),), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.feature_names = list(FEATURE_NAMES)
        self._size = np.asarray(self.cfg.size, dtype=np.float64)

    # ------------------------------------------------------------------ setup
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        c, rng = self.cfg, self.np_random
        n = int(rng.integers(c.n_obstacles[0], c.n_obstacles[1] + 1))
        self.obstacles = np.column_stack([
            rng.uniform(10, self._size[0] - 10, n),
            rng.uniform(10, self._size[1] - 10, n),
            rng.uniform(*c.obstacle_radius, n),
            rng.uniform(*c.obstacle_height, n),
        ])  # columns: x, y, radius, height

        self.pos = self._free_point(np.array([5.0, 5.0, 8.0]), np.array([20.0, 20.0, 15.0]))
        self.waypoints = [
            self._free_point(np.array([self._size[0] - 25, self._size[1] - 25, 5.0]),
                             np.array([self._size[0] - 5, self._size[1] - 5, 35.0]))
        ]
        for _ in range(c.n_waypoints - 1):
            self.waypoints.insert(-1, self._free_point(np.array([15.0, 15.0, 5.0]),
                                                       self._size - np.array([15.0, 15.0, 10.0])))
        self.wp_index = 0
        self.vel = np.zeros(3)
        self.battery = 1.0
        self.visibility = float(rng.uniform(*c.visibility))
        speed = float(rng.uniform(*c.wind_speed))
        self.wind_dir = float(rng.uniform(-np.pi, np.pi))
        self.wind = speed * np.array([np.cos(self.wind_dir), np.sin(self.wind_dir), 0.0])
        self.steps = 0
        self.min_clearance = np.inf
        self.path = [self.pos.copy()]
        self._prev_dist = self._goal_dist()
        return self._observe(), self._info()

    def _free_point(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        for _ in range(200):
            p = self.np_random.uniform(lo, hi)
            if self._clearance(p) > self.cfg.safety_margin + 1.0:
                return p
        # fall back to flying above everything
        p = self.np_random.uniform(lo, hi)
        p[2] = min(self._size[2] - 2.0, self.cfg.obstacle_height[1] + 2.0)
        return p

    # --------------------------------------------------------------- geometry
    def _obstacle_distances(self, p: np.ndarray) -> np.ndarray:
        ox, oy, r, h = self.obstacles.T
        dxy = np.maximum(np.hypot(p[0] - ox, p[1] - oy) - r, 0.0)
        dz = np.maximum(p[2] - h, 0.0)
        return np.hypot(dxy, dz)

    def _clearance(self, p: np.ndarray) -> float:
        d_obs = self._obstacle_distances(p).min() if len(self.obstacles) else np.inf
        return float(min(d_obs, p[2]))  # the ground is an obstacle too

    def _heading(self) -> float:
        g = self.waypoints[self.wp_index] - self.pos
        return float(np.arctan2(g[1], g[0]))

    def _ranges(self) -> np.ndarray:
        """Three horizontal ray casts, a downward ray and the nearest distance."""
        c = self.cfg
        out = np.full(len(RAY_ANGLES), c.sensor_range)
        tall = self.obstacles[self.obstacles[:, 3] > self.pos[2]]
        if len(tall):
            centre = tall[:, :2] - self.pos[:2]
            radius = tall[:, 2]
            for k, ang in enumerate(RAY_ANGLES + self._heading()):
                u = np.array([np.cos(ang), np.sin(ang)])
                proj = centre @ u
                perp2 = np.einsum("ij,ij->i", centre, centre) - proj ** 2
                hit = (perp2 <= radius ** 2) & (proj > 0)
                if hit.any():
                    t = proj[hit] - np.sqrt(np.maximum(radius[hit] ** 2 - perp2[hit], 0.0))
                    out[k] = min(out[k], max(float(t.min()), 0.0))
        ox, oy, r, h = self.obstacles.T
        below = (np.hypot(self.pos[0] - ox, self.pos[1] - oy) <= r) & (h <= self.pos[2])
        down = self.pos[2] - (h[below].max() if below.any() else 0.0)
        extra = np.array([down, self._clearance(self.pos)])
        return np.concatenate([out, np.minimum(extra, c.sensor_range)]) / c.sensor_range

    # ------------------------------------------------------------ observation
    def _goal_dist(self) -> float:
        return float(np.linalg.norm(self.waypoints[self.wp_index] - self.pos))

    def _observe(self) -> np.ndarray:
        c, rng = self.cfg, self.np_random
        half = self._size / 2.0
        goal = (self.waypoints[self.wp_index] - self.pos) / self._size
        noise_scale = c.sensor_noise * (1.0 + (1.0 - self.visibility))
        ranges = np.clip(self._ranges() + rng.normal(0.0, noise_scale, 5), 0.0, 1.0)
        obs = np.concatenate([
            (self.pos - half) / half,
            goal,
            self.vel / c.max_speed,
            ranges,
            [self.visibility,
             np.linalg.norm(self.wind) / max(c.wind_speed[1], 1e-6),
             self.wind_dir / np.pi,
             self.battery],
        ])
        obs[:3] += rng.normal(0.0, c.sensor_noise * 0.1, 3)  # GPS noise
        return obs.astype(np.float32)

    def _info(self) -> dict:
        return {
            "position": self.pos.copy(),
            "waypoint": self.waypoints[self.wp_index].copy(),
            "waypoint_index": self.wp_index,
            "clearance": self._clearance(self.pos),
            "battery": self.battery,
            "success": False,
            "crash": False,
        }

    # ------------------------------------------------------------------- step
    def step(self, action):
        c = self.cfg
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        cmd = a * c.max_speed
        self.vel = self.vel + c.response * (cmd - self.vel)
        drift = c.wind_drift * self.wind
        self.pos = self.pos + (self.vel + drift) * c.dt
        self.steps += 1
        effort = 1.0 + 0.5 * np.linalg.norm(self.vel) / c.max_speed + 0.3 * np.linalg.norm(self.wind) / 15.0
        self.battery = max(0.0, self.battery - c.battery_drain * effort)
        self.path.append(self.pos.copy())

        clearance = self._clearance(self.pos)
        self.min_clearance = min(self.min_clearance, clearance)
        dist = self._goal_dist()
        reward = c.progress_weight * (self._prev_dist - dist) + c.step_penalty
        if clearance < c.safety_margin:
            reward += c.proximity_penalty
        self._prev_dist = dist

        out_of_bounds = bool(np.any(self.pos < [0, 0, 0]) or np.any(self.pos > self._size))
        crash = clearance <= c.uav_radius or out_of_bounds
        terminated = False
        success = False
        if crash:
            reward += c.crash_penalty
            terminated = True
        elif dist < c.goal_radius:
            if self.wp_index == len(self.waypoints) - 1:
                reward += c.goal_reward
                terminated = success = True
            else:
                reward += c.goal_reward / len(self.waypoints)
                self.wp_index += 1
                self._prev_dist = self._goal_dist()
        elif self.battery <= 0.0:
            terminated = True
        truncated = not terminated and self.steps >= c.max_steps

        info = self._info()
        info.update(success=success, crash=crash)
        return self._observe(), float(reward), terminated, truncated, info


def make_env(config: dict | None = None, seed: int | None = None) -> UAVNavigationEnv:
    env = UAVNavigationEnv(config)
    if seed is not None:
        env.reset(seed=seed)
    return env
