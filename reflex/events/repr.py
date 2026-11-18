"""Event representations: from the (t, x, y, p) list to tensors and images."""
from __future__ import annotations

import numpy as np


def to_frame(events: np.ndarray, height: int, width: int) -> np.ndarray:
    """Per-polarity count. Returns (2, H, W): channel 0 = OFF, channel 1 = ON."""
    frame = np.zeros((2, height, width), dtype=np.float32)
    if len(events):
        np.add.at(frame, (events["p"].astype(np.intp), events["y"], events["x"]), 1.0)
    return frame


def to_bins(events: np.ndarray, height: int, width: int, t_start_us: int,
            t_end_us: int, n_bins: int) -> np.ndarray:
    """Split the window [t_start, t_end) into `n_bins` steps. Returns (T, 2, H, W).

    This is the SNN input: one network time step per bin.
    """
    out = np.zeros((n_bins, 2, height, width), dtype=np.float32)
    if len(events) == 0:
        return out
    ev = events[(events["t"] >= t_start_us) & (events["t"] < t_end_us)]
    b = ((ev["t"] - t_start_us) * n_bins // max(t_end_us - t_start_us, 1)).astype(np.intp)
    np.add.at(out, (b, ev["p"].astype(np.intp), ev["y"], ev["x"]), 1.0)
    return out


def downsample(frame: np.ndarray, factor: int) -> np.ndarray:
    """Sum `factor` x `factor` blocks over the last two axes."""
    if factor == 1:
        return frame
    *lead, h, w = frame.shape
    h2, w2 = h // factor, w // factor
    f = frame[..., : h2 * factor, : w2 * factor]
    return f.reshape(*lead, h2, factor, w2, factor).sum(axis=(-3, -1))


def render(frame: np.ndarray, saturation: float = 3.0) -> np.ndarray:
    """BGR uint8 image of a (2, H, W) frame: ON light blue, OFF red, dark background."""
    off = np.clip(frame[0] / saturation, 0, 1)[..., None]
    on = np.clip(frame[1] / saturation, 0, 1)[..., None]
    bg = np.array([28, 22, 20], dtype=np.float32)        # BGR
    c_on = np.array([255, 200, 90], dtype=np.float32)
    c_off = np.array([70, 70, 255], dtype=np.float32)
    img = bg * (1 - np.maximum(on, off)) + c_on * on + c_off * off * (1 - on)
    return np.clip(img, 0, 255).astype(np.uint8)
