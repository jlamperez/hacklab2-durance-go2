"""No-learning ball detector: the densest compact cluster of recent events.

Baseline and fallback for the follow-the-ball behavior. It answers "is there one compact
moving thing, and where", from the events of the last few tens of milliseconds:

  - events are counted on a coarse grid, the busiest neighbourhood is the candidate
  - the candidate counts as found only if enough events fall near it AND they are a large
    share of all events: when the camera itself turns, events are spread over the whole
    image and the share drops, so ego-motion is rejected
  - the position is the mean of the events near the candidate

It cannot tell a ball from a hand, a drone or a person: anything compact that moves is a
"ball". That is the job of the learned detector (SNN); this one shows where it is needed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrackerConfig:
    width: int = 320
    height: int = 320
    cell_px: int = 16           # grid for finding the candidate
    radius_px: float = 40.0     # events within this distance of the candidate belong to it
    min_events: int = 20        # in the window; noise is ~1 event per 30 ms on a still scene
    min_share: float = 0.5      # share of all events in the window that must belong to the blob


@dataclass(frozen=True)
class BallObservation:
    found: bool
    x_frac: float = 0.5         # 0 = left edge of the image, 1 = right edge
    y_frac: float = 0.5         # 0 = top, 1 = bottom
    n_blob: int = 0
    n_total: int = 0

    @property
    def share(self) -> float:
        return self.n_blob / self.n_total if self.n_total else 0.0


class CentroidTracker:
    def __init__(self, cfg: TrackerConfig | None = None):
        self.cfg = cfg or TrackerConfig()

    def observe(self, x: np.ndarray, y: np.ndarray) -> BallObservation:
        """Events of one window (pixel coordinates) -> observation."""
        c = self.cfg
        n = len(x)
        if n < c.min_events:
            return BallObservation(False, n_total=n)
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        gw, gh = -(-c.width // c.cell_px), -(-c.height // c.cell_px)
        gx = np.clip((x // c.cell_px).astype(np.intp), 0, gw - 1)
        gy = np.clip((y // c.cell_px).astype(np.intp), 0, gh - 1)
        grid = np.bincount(gy * gw + gx, minlength=gw * gh).reshape(gh, gw).astype(np.float32)
        # 3x3 box sum, so a blob on a cell border still wins
        padded = np.pad(grid, 1)
        smooth = sum(padded[dy:dy + gh, dx:dx + gw] for dy in range(3) for dx in range(3))
        cy, cx = np.unravel_index(int(np.argmax(smooth)), smooth.shape)
        px, py = (cx + 0.5) * c.cell_px, (cy + 0.5) * c.cell_px
        near = (x - px) ** 2 + (y - py) ** 2 <= c.radius_px ** 2
        n_blob = int(near.sum())
        if n_blob < c.min_events or n_blob < c.min_share * n:
            return BallObservation(False, n_blob=n_blob, n_total=n)
        return BallObservation(True, float(x[near].mean() / c.width), float(y[near].mean() / c.height),
                               n_blob, n)
