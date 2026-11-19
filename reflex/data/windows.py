"""Turn an event stream into the network's input windows.

One window = the last `window_ms` of events, split into `n_bins` time steps and
summed over `downsample` x `downsample` pixel blocks: shape (T, 2, H/ds, W/ds).
The same function feeds training, the closed loop and (on Saturday) the real camera.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from reflex.events import repr as ev_repr


@dataclass(frozen=True)
class WindowConfig:
    window_ms: float = 50.0
    n_bins: int = 10          # SNN time steps per window
    downsample: int = 4
    height: int = 180         # sensor resolution the window is cut from
    width: int = 320

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (self.n_bins, 2, self.height // self.downsample, self.width // self.downsample)

    def to_dict(self) -> dict:
        return asdict(self)


def make_window(events: np.ndarray, t_end_us: int, cfg: WindowConfig) -> np.ndarray:
    """Window ending at `t_end_us`. `events` must be sorted by time. Returns uint8 counts."""
    t0, t1 = int(t_end_us - cfg.window_ms * 1000), int(t_end_us)
    n_bins, _, h2, w2 = cfg.shape
    lo, hi = np.searchsorted(events["t"], [t0, t1])
    ev = events[lo:hi]
    # Count straight into the small grid. Same result as to_bins() followed by
    # downsample(), without building the full-resolution array (matters on the Kria).
    ds = cfg.downsample
    y, x = ev["y"].astype(np.intp) // ds, ev["x"].astype(np.intp) // ds
    keep = (y < h2) & (x < w2)
    b = ((ev["t"] - t0) * n_bins // max(t1 - t0, 1)).astype(np.intp)
    flat = ((b * 2 + ev["p"]) * h2 + y) * w2 + x
    counts = np.bincount(flat[keep], minlength=n_bins * 2 * h2 * w2)
    return np.minimum(counts, 255).astype(np.uint8).reshape(n_bins, 2, h2, w2)
