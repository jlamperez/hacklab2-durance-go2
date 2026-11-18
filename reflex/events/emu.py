"""Event-camera emulator: intensity frames -> events (t, x, y, p).

Reduced ESIM/v2e model: every pixel remembers the log-intensity at which it last
fired, and emits an ON/OFF event each time the current log-intensity moves away
from that reference by more than a contrast threshold.

This is an emulator, not the real sensor. It exists so that data and pipeline are
ready before the hackathon. Parameters are randomized per episode
(`EmuConfig.randomized`) so the network does not overfit to one sensor model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# t in microseconds; p: 1 = ON (brighter), 0 = OFF (darker)
EVENT_DTYPE = np.dtype([("t", "<i8"), ("x", "<u2"), ("y", "<u2"), ("p", "u1")])


def empty_events() -> np.ndarray:
    return np.zeros(0, dtype=EVENT_DTYPE)


@dataclass
class EmuConfig:
    c_on: float = 0.20            # ON contrast threshold (log-intensity)
    c_off: float = 0.20           # OFF contrast threshold
    sigma_c: float = 0.03         # per-pixel threshold mismatch (relative)
    noise_rate_hz: float = 0.5    # background activity per pixel
    hot_pixel_frac: float = 1e-4  # fraction of hot pixels
    hot_rate_hz: float = 50.0     # firing rate of a hot pixel
    log_eps: float = 0.05         # avoids log(0) and flattens shadows
    max_events_per_px: int = 8    # per-pixel, per-step cap (saturation)
    gain: float = 1.0             # global illumination (multiplies intensity)

    @classmethod
    def randomized(cls, rng: np.random.Generator) -> "EmuConfig":
        """Strong per-episode domain randomization."""
        return cls(
            c_on=float(rng.uniform(0.12, 0.35)),
            c_off=float(rng.uniform(0.12, 0.35)),
            sigma_c=float(rng.uniform(0.01, 0.06)),
            noise_rate_hz=float(10 ** rng.uniform(-1.3, 0.2)),   # 0.05 .. 1.6 Hz
            hot_pixel_frac=float(10 ** rng.uniform(-5.0, -3.3)),
            hot_rate_hz=float(rng.uniform(20, 200)),
            gain=float(rng.uniform(0.5, 1.5)),
        )


class EventEmulator:
    def __init__(self, height: int, width: int, cfg: EmuConfig | None = None,
                 rng: np.random.Generator | None = None):
        self.h, self.w = height, width
        self.cfg = cfg or EmuConfig()
        self.rng = rng or np.random.default_rng(0)
        mism = 1.0 + self.cfg.sigma_c * self.rng.standard_normal((2, height, width))
        mism = np.clip(mism, 0.5, 1.5)
        self.c_on = (self.cfg.c_on * mism[0]).astype(np.float32)
        self.c_off = (self.cfg.c_off * mism[1]).astype(np.float32)
        self.hot = self.rng.random((height, width)) < self.cfg.hot_pixel_frac
        self.ref: np.ndarray | None = None
        self.t_prev_us = 0

    def _log(self, frame: np.ndarray) -> np.ndarray:
        f = np.clip(frame.astype(np.float32) * self.cfg.gain, 0.0, 1.0)
        return np.log(f + self.cfg.log_eps)

    def reset(self, frame: np.ndarray, t_us: int) -> None:
        """Set the reference. `frame`: grayscale in [0, 1], shape (H, W)."""
        self.ref = self._log(frame)
        self.t_prev_us = int(t_us)

    def step(self, frame: np.ndarray, t_us: int) -> np.ndarray:
        """Return the events that happened between the previous frame and this one."""
        if self.ref is None:
            self.reset(frame, t_us)
            return empty_events()
        t0, t1 = self.t_prev_us, int(t_us)
        dt_s = max(t1 - t0, 1) * 1e-6
        diff = self._log(frame) - self.ref

        cap = self.cfg.max_events_per_px
        n_on = np.clip(np.floor(diff / self.c_on), 0, cap).astype(np.int32)
        n_off = np.clip(np.floor(-diff / self.c_off), 0, cap).astype(np.int32)
        self.ref += n_on * self.c_on - n_off * self.c_off

        chunks = [self._emit(n_on, 1, t0, t1), self._emit(n_off, 0, t0, t1)]

        # Noise: background activity plus hot pixels, random polarity.
        rate = np.where(self.hot, self.cfg.hot_rate_hz, self.cfg.noise_rate_hz)
        n_noise = self.rng.poisson(rate * dt_s).astype(np.int32)
        if n_noise.any():
            ev = self._emit(n_noise, 1, t0, t1)
            ev["p"] = self.rng.integers(0, 2, size=len(ev), dtype=np.uint8)
            chunks.append(ev)

        self.t_prev_us = t1
        events = np.concatenate(chunks)
        return events[np.argsort(events["t"], kind="stable")]

    def _emit(self, counts: np.ndarray, polarity: int, t0: int, t1: int) -> np.ndarray:
        ys, xs = np.nonzero(counts)
        if len(ys) == 0:
            return empty_events()
        n = counts[ys, xs]
        ev = np.zeros(int(n.sum()), dtype=EVENT_DTYPE)
        ev["x"] = np.repeat(xs, n)
        ev["y"] = np.repeat(ys, n)
        ev["p"] = polarity
        # The k events of one pixel are spread across the interval.
        idx_in_px = np.arange(len(ev)) - np.repeat(np.cumsum(n) - n, n)
        frac = (idx_in_px + 1) / np.repeat(n + 1, n)
        ev["t"] = t0 + (frac * (t1 - t0)).astype(np.int64)
        return ev
