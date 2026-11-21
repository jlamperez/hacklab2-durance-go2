"""Where the target is in the image: the contract between perception and the follow controller.

Anything that can say where the ball (or the drone) is produces a `Target`:

  CentroidTracker   no learning: the centre of the recent events (below). Baseline and
                    stand-in until the SNN that outputs coordinates is ready.
  the SNN           the network trained to output the ball's coordinates, on a PC or on the
                    Kria. Its adapter converts the network's output to a Target and
                    publishes it on /reflex/target; nothing downstream changes.

Coordinates are normalised so they do not depend on the resolution or the downsampling:

  u in [-1, 1]   -1 left edge, +1 right edge of the image, as the camera sees it
  v in [-1, 1]   -1 top edge,  +1 bottom edge. For a ball on the floor, larger v = closer.
  confidence     0 means "nothing seen"; u and v are then meaningless.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Target:
    u: float
    v: float
    confidence: float
    t_us: int = 0        # time of the newest event the estimate is based on

    @property
    def found(self) -> bool:
        return self.confidence > 0.0


def no_target(t_us: int = 0) -> Target:
    return Target(0.0, 0.0, 0.0, t_us)


@dataclass
class CentroidConfig:
    min_events: float = 30.0    # events near the peak (above the background) below which nothing is
                                # seen; set by `calibrate`
    recent_bins: int = 10       # newest time steps of the window that count (10 = all of a 50 ms window)
    zone_top: float = 0.3       # ignore image rows above this fraction (people's upper bodies,
                                # the room). 0.0 to use the whole image, e.g. for a drone
    radius: int = 6             # cells around the peak that belong to the target
    min_share: float = 0.4      # the target must hold this share of all events above the floor:
                                # less means the whole scene moves (the dog's own motion)
    cell_floor: float = 2.0     # events per cell treated as sensor noise when computing the share


class CentroidTracker:
    """Target = centre of the densest cluster of recent events, if it stands out.

    One moving object on a still camera gives a compact cloud of events; sensor noise
    gives isolated events. So: subtract the still-scene background (hot pixels), find
    the densest 3x3 cell patch, and take the weighted centre around it. The dog's own
    motion gives events everywhere: the share test rejects that. Of two moving objects,
    the one with more events wins.
    """

    def __init__(self, cfg: CentroidConfig | None = None):
        self.cfg = cfg or CentroidConfig()
        self.background: np.ndarray | None = None   # mean events per cell on a still scene
        self.last: dict = {}     # details of the last estimate, for viewers

    def _zone(self, window: np.ndarray) -> tuple[np.ndarray, int]:
        t, _, h, _ = window.shape
        top = int(h * self.cfg.zone_top)
        counts = window[max(0, t - self.cfg.recent_bins):, :, top:, :].astype(np.float32).sum(axis=(0, 1))
        if self.background is not None and self.background.shape == counts.shape:
            counts = np.maximum(counts - self.background, 0.0)
        return counts, top

    @staticmethod
    def _box3(a: np.ndarray) -> np.ndarray:
        p = np.pad(a, 1)
        return sum(p[i:i + a.shape[0], j:j + a.shape[1]] for i in range(3) for j in range(3))

    def calibrate(self, idle_windows: np.ndarray, margin: float = 1.3, percentile: float = 99.0) -> float:
        """Learn the still scene: a background map (hot pixels, flicker) and `min_events` above it."""
        self.background = None
        zones = np.stack([self._zone(w)[0] for w in idle_windows])
        self.background = zones.mean(axis=0)
        peaks = [float(self._box3(np.maximum(z - self.background, 0.0)).max()) for z in zones]
        self.cfg.min_events = max(float(np.percentile(peaks, percentile)) * margin, 5.0)
        return self.cfg.min_events

    def predict(self, window: np.ndarray, t_us: int = 0) -> Target:
        c = self.cfg
        _, _, h, w = window.shape
        counts, top = self._zone(window)
        dense = self._box3(counts)
        py, px = np.unravel_index(int(dense.argmax()), dense.shape)
        peak = float(dense[py, px])
        self.last = {"peak": round(peak, 1), "min_events": round(float(c.min_events), 1)}
        if peak < c.min_events:
            return no_target(t_us)
        y0, y1 = max(0, py - c.radius), min(counts.shape[0], py + c.radius + 1)
        x0, x1 = max(0, px - c.radius), min(counts.shape[1], px + c.radius + 1)
        patch = counts[y0:y1, x0:x1]
        above = np.maximum(counts - c.cell_floor, 0.0)
        share = float(np.maximum(patch - c.cell_floor, 0.0).sum() / max(above.sum(), 1e-6))
        self.last["share"] = round(share, 3)
        if share < c.min_share:
            return no_target(t_us)
        rows, cols = np.mgrid[y0:y1, x0:x1]
        mass = float(patch.sum())
        cy = float((rows * patch).sum() / mass) + top + 0.5
        cx = float((cols * patch).sum() / mass) + 0.5
        amount = min(1.0, 0.5 * peak / c.min_events)
        clean = min(1.0, share / c.min_share - 0.5)
        return Target(2.0 * cx / w - 1.0, 2.0 * cy / h - 1.0, amount * clean, t_us)
