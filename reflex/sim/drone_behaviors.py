"""Predefined flight behaviors of a tiny-whoop drone, for generating training data.

Every behavior is a trajectory: position (world frame, m) and heading (rad) as a function of
time. Parameters (distance, height, speed, side, ...) are drawn from a random generator, so
one behavior name with many seeds gives many different flights.

Frame (robot at the origin, standing, facing +x): +x forward, +y left, +z up. The camera sits
about 0.5 m above the floor and sees roughly +-35 deg sideways, from the floor at about
0.5 m up to about 1.4 m height at 2.5 m distance. Distances are kept indoor-sized (0.7-3.5 m),
speeds tiny-whoop-sized (0.3-3 m/s).
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

CAM_Z = 0.52            # camera height with the robot standing (mast mount)
HALF_FOV = math.radians(33)


def in_view_point(rng, x_range=(1.2, 3.0), margin=0.85) -> np.ndarray:
    """Random point the standing robot's camera sees."""
    x = rng.uniform(*x_range)
    y = rng.uniform(-1, 1) * margin * x * math.tan(HALF_FOV)
    z = rng.uniform(0.3, min(1.6, CAM_Z + x * math.tan(math.radians(20))))
    return np.array([x, y, z])


def _ease(a: float) -> float:
    a = min(max(a, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * a)


class Trajectory:
    name = "trajectory"
    duration = 10.0

    def __init__(self):
        self._hover_wobble = None

    def pos(self, t: float) -> np.ndarray:
        raise NotImplementedError

    def at(self, t: float) -> tuple[np.ndarray, float]:
        """Position and heading at time t (heading follows the direction of travel)."""
        p = self.pos(t)
        q = self.pos(t + 0.05)
        d = q[:2] - p[:2]
        if np.linalg.norm(d) > 1e-3:
            self._yaw = math.atan2(d[1], d[0])
        return p + self._wobble(t), getattr(self, "_yaw", math.pi)

    def _wobble(self, t: float) -> np.ndarray:
        """Small hover drift of a real tiny whoop (a few cm)."""
        if self._hover_wobble is None:
            return np.zeros(3)
        a, f, ph = self._hover_wobble
        return a * np.sin(2 * math.pi * f * t + ph)

    def _set_wobble(self, rng, amp=0.02):
        self._hover_wobble = (rng.uniform(0.3, 1.0, 3) * amp, rng.uniform(0.3, 1.5, 3), rng.uniform(0, 2 * math.pi, 3))


class Keyframes(Trajectory):
    """Eased moves between keyframes [(time, position)], held after the last one."""

    def __init__(self, keys: list[tuple[float, np.ndarray]]):
        super().__init__()
        self.keys = [(float(t), np.asarray(p, float)) for t, p in keys]
        self.duration = self.keys[-1][0]

    def pos(self, t):
        k = self.keys
        if t <= k[0][0]:
            return k[0][1].copy()
        for (t0, p0), (t1, p1) in zip(k, k[1:]):
            if t <= t1:
                return p0 + _ease((t - t0) / max(t1 - t0, 1e-9)) * (p1 - p0)
        return k[-1][1].copy()


class Function(Trajectory):
    def __init__(self, f: Callable[[float], np.ndarray], duration: float):
        super().__init__()
        self.f, self.duration = f, duration

    def pos(self, t):
        return np.asarray(self.f(min(t, self.duration)), float)


def _travel(a, b, speed) -> float:
    return float(np.linalg.norm(np.asarray(b) - np.asarray(a))) / speed


# ------------------------------------------------------------------------ the behaviors

def hover(rng):
    """Hovers at one spot in view, drifting a few cm."""
    p = in_view_point(rng)
    tr = Keyframes([(0, p), (rng.uniform(6, 10), p)])
    tr._set_wobble(rng, amp=rng.uniform(0.01, 0.04))
    return tr


def cross(rng):
    """Flies straight across the view, entering and leaving at the sides."""
    x, z = rng.uniform(1.0, 3.2), rng.uniform(0.35, 1.4)
    side = rng.choice([-1, 1])
    y_out = (x * math.tan(HALF_FOV) + 0.4) * side
    a, b = np.array([x + rng.uniform(-0.5, 0.5), y_out, z]), np.array([x + rng.uniform(-0.5, 0.5), -y_out, z + rng.uniform(-0.2, 0.2)])
    t = _travel(a, b, rng.uniform(0.5, 2.5))
    return Keyframes([(0, a), (t, b), (t + 0.5, b)])


def approach(rng):
    """Comes straight at the robot from far away, stops close, backs off again."""
    far = in_view_point(rng, x_range=(2.8, 3.5), margin=0.5)
    near = np.array([rng.uniform(0.7, 1.1), far[1] * 0.25, rng.uniform(0.45, 0.8)])
    v = rng.uniform(0.5, 1.5)
    t1 = _travel(far, near, v)
    hold = rng.uniform(0.5, 2.0)
    return Keyframes([(0, far), (t1, near), (t1 + hold, near), (2 * t1 + hold, far)])


def orbit(rng):
    """Flies an arc around the robot at constant distance, back and forth."""
    r, z = rng.uniform(1.3, 2.6), rng.uniform(0.4, 1.2)
    span = math.radians(rng.uniform(20, 32))
    w = rng.uniform(0.25, 0.6) * 2 * math.pi / 4      # rad/s of the sweep phase
    ph = rng.uniform(0, 2 * math.pi)
    return Function(lambda t: np.array([r * math.cos(span * math.sin(w * t + ph)),
                                        r * math.sin(span * math.sin(w * t + ph)), z]), rng.uniform(8, 12))


def figure8(rng):
    """Figure eight in a vertical plane facing the robot."""
    x, z0 = rng.uniform(1.4, 2.8), rng.uniform(0.6, 1.0)
    ay, az = rng.uniform(0.3, 0.6) * x * math.tan(HALF_FOV), rng.uniform(0.15, 0.3)
    w = 2 * math.pi / rng.uniform(4, 7)
    return Function(lambda t: np.array([x, ay * math.sin(w * t), z0 + az * math.sin(2 * w * t)]), rng.uniform(8, 12))


def zigzag(rng):
    """Zigzags left and right while coming closer."""
    x0, x1, z = rng.uniform(2.6, 3.4), rng.uniform(1.0, 1.5), rng.uniform(0.4, 1.1)
    n = int(rng.integers(3, 6))
    keys, t = [], 0.0
    v = rng.uniform(0.8, 2.0)
    prev = None
    for k in range(n + 1):
        x = x0 + (x1 - x0) * k / n
        p = np.array([x, (1 if k % 2 else -1) * 0.7 * x * math.tan(HALF_FOV), z])
        t += 0 if prev is None else _travel(prev, p, v)
        keys.append((t, p))
        prev = p
    return Keyframes(keys + [(t + 0.5, prev)])


def rise_drop(rng):
    """Goes straight up and down at one spot (take-off / landing like)."""
    p = in_view_point(rng, x_range=(1.3, 2.8))
    lo, hi = p.copy(), p.copy()
    lo[2], hi[2] = 0.05, rng.uniform(1.0, 1.5)
    v = rng.uniform(0.4, 1.0)
    t1 = _travel(lo, hi, v)
    return Keyframes([(0, lo), (0.8, lo), (0.8 + t1, hi), (1.8 + t1, hi), (1.8 + 2 * t1, lo), (2.5 + 2 * t1, lo)])


def dart(rng):
    """Hovers, then shoots to another spot fast, several times."""
    keys, t = [], 0.0
    p = in_view_point(rng)
    for _ in range(int(rng.integers(3, 6))):
        keys.append((t, p))
        t += rng.uniform(0.6, 1.5)                       # hover
        keys.append((t, p))
        q = in_view_point(rng)
        t += _travel(p, q, rng.uniform(2.0, 3.0))
        p = q
    keys.append((t, p))
    tr = Keyframes(keys + [(t + 0.8, p)])
    tr._set_wobble(rng)
    return tr


def leave_return(rng):
    """Flies out of view at one side, waits outside, comes back in from the other side."""
    x, z = rng.uniform(1.2, 2.8), rng.uniform(0.4, 1.2)
    side = rng.choice([-1, 1])
    inside = np.array([x, rng.uniform(-0.3, 0.3) * x, z])
    out_a = np.array([x, side * (x * math.tan(HALF_FOV) + 0.5), z])
    out_b = np.array([x, -out_a[1], z + rng.uniform(-0.2, 0.2)])
    v = rng.uniform(0.6, 1.8)
    t1 = 1.0 + _travel(inside, out_a, v)
    t2 = t1 + rng.uniform(1.0, 2.5)                       # out of view
    t3 = t2 + 0.1                                          # "teleport" to the other side, unseen
    t4 = t3 + _travel(out_b, inside, v)
    return Keyframes([(0, inside), (1.0, inside), (t1, out_a), (t2, out_a), (t3, out_b), (t4, inside), (t4 + 1.5, inside)])


def waypoints(rng):
    """Random smooth flight between points in view, sometimes pausing."""
    keys, t = [], 0.0
    p = in_view_point(rng)
    while t < 10:
        keys.append((t, p))
        if rng.random() < 0.3:
            t += rng.uniform(0.5, 1.5)
            keys.append((t, p))
        q = in_view_point(rng)
        t += _travel(p, q, rng.uniform(0.6, 1.6))
        p = q
    keys.append((t, p))
    return Keyframes(keys)


BEHAVIORS: dict[str, Callable[[np.random.Generator], Trajectory]] = {
    "hover": hover, "cross": cross, "approach": approach, "orbit": orbit, "figure8": figure8,
    "zigzag": zigzag, "rise_drop": rise_drop, "dart": dart, "leave_return": leave_return,
    "waypoints": waypoints,
}


def make(name: str, rng: np.random.Generator) -> Trajectory:
    tr = BEHAVIORS[name](rng)
    tr.name = name
    return tr
