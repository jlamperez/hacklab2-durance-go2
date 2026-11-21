"""Synthetic river recordings: a bouncing ball, plus distractors, as GenX320-like events.

Purpose: train and deploy a first model before any real recording exists, and pretrain
before fine-tuning on real data. It is a cartoon of the sensor, not an emulator: events
appear where a shape's edge moves (leading edge one polarity, trailing edge the other),
with random dropout, background noise and hot pixels. Real recordings decide.

Writes the same `.bin` format as the camera, plus `<name>.gt.npz` (t_us, x, visible)
with the ball's true position, so labels do not depend on the centroid heuristic:

    data/river_synth/<session>/ball/*.bin   ball (sometimes with a distractor)
    data/river_synth/<session>/none/*.bin   only distractors: walker, waving hand,
                                            camera pan, flicker
    data/river_synth/<session>/none/empty_000.bin   noise and hot pixels only, for the
                                            labeller's noise calibration

Sensor noise (background rate, hot pixels) is drawn once per session, as on a real sensor.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import EVENT_DTYPE, write_river_bin


@dataclass
class SynthConfig:
    sensor: int = 320
    duration_s: float = 20.0
    dt_us: int = 2000
    # background events/s over the sensor. First real GenX320 take (noise filter 0.5):
    # ~700 ev/s, so the upper end is a deliberately noisy room
    noise_hz: tuple[float, float] = (3e2, 1e4)
    hot_pixels: tuple[int, int] = (0, 8)
    keep_prob: tuple[float, float] = (0.4, 0.9)    # fraction of edge pixels that fire per step
    min_speed: float = 30.0                         # px/s; a slower ball is invisible -> none


class _Events:
    def __init__(self):
        self.parts: list[np.ndarray] = []

    def add(self, x, y, p, t):
        if len(x):
            e = np.empty(len(x), dtype=EVENT_DTYPE)
            e["x"], e["y"], e["p"], e["t"] = x, y, p, t
            self.parts.append(e)

    def done(self) -> np.ndarray:
        if not self.parts:
            return np.zeros(0, dtype=EVENT_DTYPE)
        ev = np.concatenate(self.parts)
        return ev[np.argsort(ev["t"], kind="stable")]


def _edge_events(out: _Events, mask_fn, box, t, dt, lead_p, keep, rng, sensor):
    """Events where mask_fn(old) and mask_fn(new) differ inside box (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = (int(np.clip(v, 0, sensor)) for v in box)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    old, new = mask_fn(xx, yy)
    for m, p in ((new & ~old, lead_p), (old & ~new, 1 - lead_p)):
        ys, xs = np.nonzero(m & (rng.random(m.shape) < keep))
        out.add(xs + x0, ys + y0, np.full(len(xs), p, np.uint8), t + rng.integers(0, dt, len(xs)))


def _ball(out, cfg, rng, T):
    """Bouncing ball. Returns ground truth (t_us, x, visible) per step."""
    s, dt = cfg.sensor, cfg.dt_us * 1e-6
    r = rng.uniform(6, 22)
    floor = rng.uniform(0.6, 0.95) * s
    g, e = rng.uniform(400, 1500), rng.uniform(0.55, 0.85)
    lead_p = int(rng.integers(0, 2))          # bright or dark ball against the background
    keep = rng.uniform(*cfg.keep_prob)
    x, y = rng.uniform(r, s - r), rng.uniform(0.2 * s, floor - r)
    vx, vy = rng.choice([-1, 1]) * rng.uniform(30, 250), rng.uniform(-300, 100)
    away = 0.0                                 # seconds left out of view
    ts, xs, vis = [], [], []
    for k in range(T):
        t = k * cfg.dt_us
        if away > 0:
            away -= dt
            if away <= 0:                      # comes back in from a side
                x = r if rng.random() < 0.5 else s - r
                y, vx = rng.uniform(0.2 * s, floor - r), np.sign(s / 2 - x) * rng.uniform(40, 250)
                vy = rng.uniform(-400, 0)
            ts.append(t); xs.append(x); vis.append(False)
            continue
        on_floor = y >= floor - r - 0.5
        if on_floor and abs(vy) < 60 and rng.random() < 0.6 * dt:   # someone kicks it again
            vy, vx = -rng.uniform(300, 900), rng.choice([-1, 1]) * rng.uniform(30, 250)
        nx, ny, nvy = x + vx * dt, y + vy * dt, vy + g * dt
        if ny > floor - r:
            ny, nvy = floor - r, -abs(nvy) * e if abs(nvy) > 40 else 0.0
            vx *= 1 - 0.8 * dt                 # rolling friction
        if nx < r or nx > s - r:
            if rng.random() < 0.3:             # leaves the picture
                away = rng.uniform(0.3, 2.0)
                ts.append(t); xs.append(x); vis.append(False)
                continue
            vx, nx = -vx, np.clip(nx, r, s - r)
        cx0, cy0 = x, y

        def disk(xx, yy, cx0=cx0, cy0=cy0, cx1=nx, cy1=ny):
            return ((xx - cx0) ** 2 + (yy - cy0) ** 2 <= r * r,
                    (xx - cx1) ** 2 + (yy - cy1) ** 2 <= r * r)

        _edge_events(out, disk, (min(x, nx) - r - 1, min(y, ny) - r - 1,
                                 max(x, nx) + r + 2, max(y, ny) + r + 2),
                     t, cfg.dt_us, lead_p, keep, rng, s)
        speed = np.hypot(nx - x, ny - y) / dt
        x, y, vy = nx, ny, nvy
        ts.append(t); xs.append(x); vis.append(speed >= cfg.min_speed)
    return {"t_us": np.array(ts, np.int64), "x": np.array(xs, np.float32),
            "visible": np.array(vis, bool)}


def _rect_track(out, cfg, rng, T, w, h, pos_fn):
    """A rectangle whose top-left corner follows pos_fn(time_s) -> (x, y)."""
    s, lead_p, keep = cfg.sensor, int(rng.integers(0, 2)), rng.uniform(*cfg.keep_prob)
    px, py = pos_fn(0.0)
    for k in range(T):
        t = k * cfg.dt_us
        nx, ny = pos_fn((k + 1) * cfg.dt_us * 1e-6)

        def rect(xx, yy, ax=px, ay=py, bx=nx, by=ny):
            return ((xx >= ax) & (xx < ax + w) & (yy >= ay) & (yy < ay + h),
                    (xx >= bx) & (xx < bx + w) & (yy >= by) & (yy < by + h))

        _edge_events(out, rect, (min(px, nx) - 1, min(py, ny) - 1, max(px, nx) + w + 2,
                                 max(py, ny) + h + 2), t, cfg.dt_us, lead_p, keep, rng, s)
        px, py = nx, ny


def _walker(out, cfg, rng, T):
    s = cfg.sensor
    w, h = rng.uniform(40, 110), rng.uniform(150, 300)
    v = rng.choice([-1, 1]) * rng.uniform(30, 140)
    x0, y = (-w if v > 0 else s), rng.uniform(0, s - h * 0.6)
    period = s + 2 * w
    _rect_track(out, cfg, rng, T, w, h, lambda ts: ((x0 + v * ts + w) % period - w, y))


def _hand(out, cfg, rng, T):
    s = cfg.sensor
    w, h = rng.uniform(25, 50), rng.uniform(40, 80)
    cx, y = rng.uniform(0.2, 0.8) * s, rng.uniform(0, 0.6) * s
    amp, f = rng.uniform(15, 60), rng.uniform(1, 3)
    on, off = rng.uniform(1, 4), rng.uniform(0.5, 3)        # waves, then rests
    _rect_track(out, cfg, rng, T, w, h,
                lambda ts: (cx + (amp * np.sin(2 * np.pi * f * ts) if ts % (on + off) < on else 0.0), y))


def _pan(out, cfg, rng, T):
    """Camera turning: vertical edges of the background sweep across the image in bursts."""
    s = cfg.sensor
    n = int(rng.integers(4, 12))
    xs0 = rng.uniform(0, s, n)
    heights = [(rng.uniform(0, 0.5) * s, rng.uniform(0.3, 1.0) * s) for _ in range(n)]
    v = rng.choice([-1, 1]) * rng.uniform(80, 400)
    burst, pause = rng.uniform(0.3, 1.0), rng.uniform(0.5, 3.0)

    def shift(ts):
        cycles, phase = divmod(ts, burst + pause)
        return v * (cycles * burst + min(phase, burst))

    for x0, (y0, hh) in zip(xs0, heights):
        _rect_track(out, cfg, rng, T, 3, hh, lambda ts, x0=x0, y0=y0: ((x0 + shift(ts)) % s, y0))


def _flicker(out, cfg, rng, T):
    s = cfg.sensor
    cx, cy, size = rng.uniform(0, s - 20), rng.uniform(0, s - 20), int(rng.integers(5, 20))
    rate = rng.uniform(2e3, 2e4)
    n = rng.poisson(rate * T * cfg.dt_us * 1e-6)
    out.add((cx + rng.integers(0, size, n)).astype(np.uint16), (cy + rng.integers(0, size, n)).astype(np.uint16),
            rng.integers(0, 2, n).astype(np.uint8), rng.integers(0, T * cfg.dt_us, n))


DISTRACTORS = {"walker": _walker, "hand": _hand, "pan": _pan, "flicker": _flicker}


@dataclass
class SensorNoise:
    """Background rate (events/s over the sensor) and hot pixels (x, y, events/s)."""
    rate_hz: float
    hot: list

    @classmethod
    def sample(cls, cfg: SynthConfig, rng: np.random.Generator) -> "SensorNoise":
        n_hot = int(rng.integers(*cfg.hot_pixels)) if cfg.hot_pixels[1] > 0 else 0
        return cls(float(rng.uniform(*cfg.noise_hz)),
                   [(int(rng.integers(0, cfg.sensor)), int(rng.integers(0, cfg.sensor)),
                     float(rng.uniform(50, 300))) for _ in range(n_hot)])


def _noise(out, cfg, rng, T, noise: SensorNoise | None = None):
    s, dur = cfg.sensor, T * cfg.dt_us
    noise = noise or SensorNoise.sample(cfg, rng)
    n = rng.poisson(noise.rate_hz * dur * 1e-6)
    out.add(rng.integers(0, s, n), rng.integers(0, s, n), rng.integers(0, 2, n).astype(np.uint8),
            rng.integers(0, dur, n))
    for hx, hy, rate in noise.hot:
        k = rng.poisson(rate * dur * 1e-6)
        out.add(np.full(k, hx), np.full(k, hy), rng.integers(0, 2, k).astype(np.uint8),
                rng.integers(0, dur, k))


def generate(kind: str, cfg: SynthConfig, rng: np.random.Generator,
             noise: SensorNoise | None = None):
    """Events and ground truth (None unless kind is 'ball') for one recording.
    kind: ball, none (distractors only) or empty (sensor noise only)."""
    T = int(cfg.duration_s * 1e6 / cfg.dt_us)
    out = _Events()
    gt = None
    if kind == "ball":
        gt = _ball(out, cfg, rng, T)
        n_dis = int(rng.random() < 0.4)
    elif kind == "empty":
        n_dis = 0
    else:
        n_dis = int(rng.integers(1, 3))
    for name in rng.choice(list(DISTRACTORS), size=n_dis, replace=False):
        DISTRACTORS[name](out, cfg, rng, T)
    _noise(out, cfg, rng, T, noise)
    ev = out.done()
    keep = (ev["x"] < cfg.sensor) & (ev["y"] < cfg.sensor)
    return ev[keep], gt


def write_dataset(root: str | Path, sessions: int, per_kind: int, cfg: SynthConfig, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for si in range(sessions):
        noise = SensorNoise.sample(cfg, rng)
        jobs = [("empty", "none", "empty_000")] + [(k, k, f"{k}_{i:03d}") for k in ("ball", "none")
                                                     for i in range(per_kind)]
        for kind, folder, name in jobs:
            path = Path(root) / f"synth{si:02d}" / folder / f"{name}.bin"
            ev, gt = generate(kind, cfg, rng, noise)
            write_river_bin(path, ev)
            if gt is not None:
                np.savez(path.with_suffix(".gt.npz"), **gt)
        print(f"session synth{si:02d} written")


def main() -> None:
    ap = argparse.ArgumentParser(description="Write synthetic river recordings")
    ap.add_argument("--out", default="data/river_synth")
    ap.add_argument("--sessions", type=int, default=5)
    ap.add_argument("--per-kind", type=int, default=6, help="recordings per kind per session")
    ap.add_argument("--duration", type=float, default=20.0, help="seconds per recording")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    write_dataset(a.out, a.sessions, a.per_kind, SynthConfig(duration_s=a.duration), a.seed)


if __name__ == "__main__":
    main()
