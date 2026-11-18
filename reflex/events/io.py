"""Event I/O. This is the seam with the real camera.

Single internal format: a structured array of `EVENT_DTYPE` (t in µs, x, y, p).
The adapter for the Durance camera is added here on Saturday; the rest of the
code must not care where events come from.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .emu import EVENT_DTYPE

_FIELDS = ("t", "x", "y", "p")


def save_npz(path: str | Path, events: np.ndarray, height: int, width: int, **meta) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, t=events["t"], x=events["x"], y=events["y"], p=events["p"],
                        height=height, width=width, **meta)


def load_npz(path: str | Path) -> tuple[np.ndarray, dict]:
    d = np.load(path, allow_pickle=False)
    ev = np.zeros(len(d["t"]), dtype=EVENT_DTYPE)
    for k in _FIELDS:
        ev[k] = d[k]
    meta = {k: (d[k].item() if d[k].ndim == 0 else d[k]) for k in d.files if k not in _FIELDS}
    return ev, meta
