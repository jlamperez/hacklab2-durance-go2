"""Predefined flight behaviors of a tiny-whoop drone, for generating training data.

Every behavior is a trajectory: position (world frame, m) and heading (rad) as a function of
time. Parameters (distance, height, speed, side, ...) are drawn from a random generator, so
one behavior name with many seeds gives many different flights.

Frame (robot at the origin, standing, facing +x): +x forward, +y left, +z up. The camera is
mounted like the real one (drone_recording.CAMERA_MOUNT): on a small tripod 10 cm above the
head and 10 cm behind its front, about 0.5 m above the floor, level, 76 deg field of view
(+-38 deg). The drone stays 1-5 m away, mostly close (sample_distance), at tiny-whoop speeds
(0.3-3 m/s). The paths are what a pilot aims for; `Flight` turns one into the twitchy,
tilting, overshooting motion of a real tiny whoop.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

CAM_X = 0.23            # camera position ahead of the robot's center (m)
CAM_Z = 0.50            # camera height with the robot standing (tripod mount, measured in the model)
HALF_FOV = math.radians(38)   # half of the real camera's 76 deg (square sensor: same both ways)
NEAR, FAR = 1.0, 5.0    # distance range of the drone (m), see sample_distance


def sample_distance(rng, lo: float = NEAR, hi: float = FAR) -> float:
    """Forward distance in [lo, hi], weighted towards close: Beta(1.3, 3), so with the
    defaults the median is about 2 m, a third of the flights is closer than 1.7 m and
    fewer than one in ten is beyond 3.5 m."""
    return lo + (hi - lo) * float(rng.beta(1.3, 3.0))


def in_view_point(rng, x_range: tuple[float, float] | None = None, margin=0.85) -> np.ndarray:
    """Random point the standing robot's camera sees. Distance from sample_distance unless
    `x_range` is given."""
    x = rng.uniform(*x_range) if x_range else sample_distance(rng)
    y = rng.uniform(-1, 1) * margin * (x - CAM_X) * math.tan(HALF_FOV)
    z = rng.uniform(0.3, min(2.0, CAM_Z + margin * (x - CAM_X) * math.tan(HALF_FOV)))
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
    x, z = sample_distance(rng), rng.uniform(0.35, 1.5)
    side = rng.choice([-1, 1])
    y_out = ((x - CAM_X) * math.tan(HALF_FOV) + 0.4) * side
    a, b = np.array([x + rng.uniform(-0.5, 0.5), y_out, z]), np.array([x + rng.uniform(-0.5, 0.5), -y_out, z + rng.uniform(-0.2, 0.2)])
    t = _travel(a, b, rng.uniform(0.5, 2.5))
    return Keyframes([(0, a), (t, b), (t + 0.5, b)])


def approach(rng):
    """Comes straight at the robot from far away, stops close, backs off again."""
    far = in_view_point(rng, x_range=(3.0, FAR), margin=0.5)
    near = np.array([rng.uniform(NEAR, 1.4), far[1] * 0.25, rng.uniform(0.4, 0.8)])
    v = rng.uniform(0.5, 1.5)
    t1 = _travel(far, near, v)
    hold = rng.uniform(0.5, 2.0)
    return Keyframes([(0, far), (t1, near), (t1 + hold, near), (2 * t1 + hold, far)])


def orbit(rng):
    """Flies an arc around the robot (centered on the camera) at constant distance, back and forth."""
    r, z = sample_distance(rng), rng.uniform(0.4, 1.2)
    span = math.radians(rng.uniform(20, 32))
    w = rng.uniform(0.25, 0.6) * 2 * math.pi / 4      # rad/s of the sweep phase
    ph = rng.uniform(0, 2 * math.pi)
    return Function(lambda t: np.array([CAM_X + r * math.cos(span * math.sin(w * t + ph)),
                                        r * math.sin(span * math.sin(w * t + ph)), z]), rng.uniform(8, 12))


def figure8(rng):
    """Figure eight in a vertical plane facing the robot."""
    x, z0 = sample_distance(rng, lo=1.2), rng.uniform(0.6, 1.0)
    ay, az = rng.uniform(0.3, 0.6) * (x - CAM_X) * math.tan(HALF_FOV), rng.uniform(0.15, 0.3)
    w = 2 * math.pi / rng.uniform(4, 7)
    return Function(lambda t: np.array([x, ay * math.sin(w * t), z0 + az * math.sin(2 * w * t)]), rng.uniform(8, 12))


def zigzag(rng):
    """Zigzags left and right while coming closer."""
    x0, x1, z = rng.uniform(3.0, FAR), rng.uniform(NEAR, 1.5), rng.uniform(0.4, 1.1)
    n = int(rng.integers(3, 6))
    keys, t = [], 0.0
    v = rng.uniform(0.8, 2.0)
    prev = None
    for k in range(n + 1):
        x = x0 + (x1 - x0) * k / n
        p = np.array([x, (1 if k % 2 else -1) * 0.7 * (x - CAM_X) * math.tan(HALF_FOV), z])
        t += 0 if prev is None else _travel(prev, p, v)
        keys.append((t, p))
        prev = p
    return Keyframes(keys + [(t + 0.5, prev)])


def rise_drop(rng):
    """Goes straight up and down at one spot (take-off / landing like)."""
    p = in_view_point(rng)
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
    x, z = sample_distance(rng), rng.uniform(0.4, 1.2)
    side = rng.choice([-1, 1])
    inside = np.array([x, rng.uniform(-0.3, 0.3) * x, z])
    out_a = np.array([x, side * ((x - CAM_X) * math.tan(HALF_FOV) + 0.5), z])
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


def none(rng):
    """No drone: it stays parked far outside the view (negatives: empty scene, moving head)."""
    p = np.array([-40.0, 0.0, 1.0])     # behind the robot (the head turns at most 35 deg)
    return Keyframes([(0, p), (rng.uniform(6, 10), p)])


BEHAVIORS: dict[str, Callable[[np.random.Generator], Trajectory]] = {
    "hover": hover, "cross": cross, "approach": approach, "orbit": orbit, "figure8": figure8,
    "zigzag": zigzag, "rise_drop": rise_drop, "dart": dart, "leave_return": leave_return,
    "waypoints": waypoints, "none": none,
}


def make(name: str, rng: np.random.Generator) -> Trajectory:
    tr = BEHAVIORS[name](rng)
    tr.name = name
    return tr


# ------------------------------------------------------------------- flight dynamics

G = 9.81
MAX_TILT = math.radians(40)


class Flight:
    """A twitchy tiny whoop flying a behavior's path, instead of gliding along it.

    The path is only where the pilot wants the drone to be. The drone chases it like a
    springy controller with too little damping and only part of the speed fed forward:
    it lags behind on fast moves and overshoots when the path stops. On top come a slow drift
    of where it holds position (5-10 cm, under 1 Hz: a whoop never quite stays put), fast
    controller jitter (filtered noise, small in position but visible as a nervous tilt),
    bobbing in height, and now and then a short jerk (a correction, a bump of air). It tilts into its acceleration like a
    real quadcopter (up to 40 deg) and its heading wobbles. All parameters are drawn per
    flight; they are plausible for a 65 mm whoop indoors, not measured on the team's drone.

        flight = Flight(path, rng, sim.dt)
        pos, quat = flight.step()          # once per simulation step
    """

    def __init__(self, path: Trajectory, rng: np.random.Generator, dt: float, calm: bool = False):
        self.path, self.rng, self.dt, self.t = path, rng, dt, 0.0
        k = 0.4 if calm else 1.0                                  # calm: for tests and comparisons
        self.omega = rng.uniform(5.0, 9.0)                        # rad/s: how hard it chases the path
        self.zeta = rng.uniform(0.3, 0.55)                        # damping: < 1, so it overshoots
        self.ff = rng.uniform(0.3, 0.7)                           # share of the path speed fed forward
        self.tau = rng.uniform(0.04, 0.10)                        # s: jitter correlation time
        self.sigma = k * rng.uniform(0.4, 1.0) * np.array([1.0, 1.0, 0.6])   # m/s^2 jitter
        self.drift_tau = rng.uniform(1.0, 2.5)                    # s
        self.drift_sigma = k * rng.uniform(0.04, 0.10) * np.array([1.0, 1.0, 0.5])   # m
        self.bob = (k * rng.uniform(0.01, 0.03), rng.uniform(1.0, 3.0), rng.uniform(0, 2 * math.pi))
        self.jerk_rate, self.jerk_dv = k * rng.uniform(0.2, 1.0), rng.uniform(0.15, 0.5)   # per s, m/s
        self.yaw_wobble = (k * rng.uniform(0.05, 0.2), rng.uniform(0.5, 2.0), rng.uniform(0, 2 * math.pi))
        p0, self.yaw = path.at(0.0)
        self.p, self.v, self.noise = p0.copy(), np.zeros(3), np.zeros(3)
        self.drift = self.drift_sigma * rng.standard_normal(3)
        self.a = np.zeros(3)

    def step(self) -> tuple[np.ndarray, np.ndarray]:
        """Advance one dt; returns position (m, world) and attitude quaternion (w, x, y, z)."""
        dt, t, rng = self.dt, self.t, self.rng
        ref, yaw = self.path.at(t)
        v_ref = (self.path.at(t + dt)[0] - self.path.at(max(t - dt, 0.0))[0]) / (2 * dt) if t > 0 else np.zeros(3)
        self.noise += -self.noise / self.tau * dt + self.sigma * math.sqrt(2 * dt / self.tau) * rng.standard_normal(3)
        self.drift += (-self.drift / self.drift_tau * dt
                       + self.drift_sigma * math.sqrt(2 * dt / self.drift_tau) * rng.standard_normal(3))
        ref = ref + self.drift
        if rng.random() < self.jerk_rate * dt:
            d = rng.standard_normal(3) * [1.0, 1.0, 0.5]
            self.v += self.jerk_dv * d / max(np.linalg.norm(d), 1e-9)
        w, z = self.omega, self.zeta
        self.a = w * w * (ref - self.p) + 2 * z * w * (self.ff * v_ref - self.v) + self.noise
        self.v += self.a * dt
        self.p += self.v * dt
        if self.p[2] < 0.03:                                       # it does not fly through the floor
            self.p[2], self.v[2] = 0.03, max(self.v[2], 0.0)
        self.t += dt
        amp, f, ph = self.bob
        pos = self.p + [0.0, 0.0, amp * math.sin(2 * math.pi * f * self.t + ph)]
        ya, yf, yph = self.yaw_wobble
        self.yaw = yaw + ya * math.sin(2 * math.pi * yf * self.t + yph)
        return pos, self.attitude()

    def attitude(self) -> np.ndarray:
        """Body z along the thrust (acceleration + gravity), capped at MAX_TILT, then the heading."""
        thrust = self.a + [0.0, 0.0, G]
        h = np.hypot(thrust[0], thrust[1])
        cap = max(thrust[2], 1.0) * math.tan(MAX_TILT)
        if h > cap:
            thrust[:2] *= cap / h
        zb = thrust / np.linalg.norm(thrust)
        xh = np.array([math.cos(self.yaw), math.sin(self.yaw), 0.0])
        yb = np.cross(zb, xh)
        yb /= np.linalg.norm(yb)
        xb = np.cross(yb, zb)
        r = np.column_stack([xb, yb, zb])
        qw = math.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
        if qw < 1e-6:                                              # 180 deg turns: not reachable at <= 40 deg tilt
            return np.array([0.0, 0.0, 0.0, 1.0])
        return np.array([qw, (r[2, 1] - r[1, 2]) / (4 * qw), (r[0, 2] - r[2, 0]) / (4 * qw), (r[1, 0] - r[0, 1]) / (4 * qw)])

    def tilt_deg(self) -> float:
        thrust = self.a + [0.0, 0.0, G]
        return math.degrees(math.atan2(np.hypot(thrust[0], thrust[1]), max(thrust[2], 1e-9)))
