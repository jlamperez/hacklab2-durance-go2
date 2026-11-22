"""Readout of the target-zone SNN's output (models/zones3, reflex/river/): which zone wins.

The board (spleat-server) sends, per 10 ms frame, one spike count per output neuron
(`none, left, center, right` for 3 zones). The network was trained to be read like this
(reflex/river/model.py, `readout_counts`): sum each neuron's spikes over the last `readout`
frames (20 = 200 ms) and take the neuron with the most. Ties go to `none` (index 0), as with
torch's argmax in training.

`margin`: a zone wins only if its count is at least `none`'s + margin. 0 is the rule the
network was trained and validated with. A margin above 0 trades false triggers for misses;
it is not validated.

Kept free of ROS and sockets so the FPGA client and a ROS node can share it.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ZoneDecision:
    label: str             # winning class name
    index: int             # its index (0 = none)
    counts: np.ndarray     # spikes per class over the window
    frames: int            # frames in the window so far (< readout while filling)


class ZonesReadout:
    def __init__(self, names: list[str], readout: int = 20, margin: float = 0.0):
        if not names or readout < 1:
            raise ValueError("need class names and readout >= 1")
        self.names, self.readout, self.margin = list(names), readout, margin
        self.window: deque[np.ndarray] = deque(maxlen=readout)

    @classmethod
    def from_model_dir(cls, model_dir: Path, margin: float = 0.0) -> "ZonesReadout":
        """Class names and readout length from the training run's meta.json."""
        meta = json.loads((Path(model_dir) / "meta.json").read_text())
        return cls(meta["names"], int(meta["window"]["readout"]), margin)

    def reset(self) -> None:
        self.window.clear()

    def push(self, values) -> ZoneDecision:
        """Add one frame's output (one value per class, any shape that flattens to that)."""
        v = np.asarray(values, dtype=np.float64).reshape(-1)
        if v.size != len(self.names):
            raise ValueError(f"{v.size} values, expected {len(self.names)} ({', '.join(self.names)})")
        self.window.append(v)
        counts = np.sum(self.window, axis=0)
        zone = int(np.argmax(counts[1:])) + 1 if len(counts) > 1 else 0
        index = zone if counts[zone] >= counts[0] + max(self.margin, 1e-9) else 0
        return ZoneDecision(self.names[index], index, counts, len(self.window))

    def format(self, d: ZoneDecision) -> str:
        spikes = " ".join(f"{n} {int(c)}" for n, c in zip(self.names, d.counts))
        filling = "" if d.frames >= self.readout else f" (window {d.frames}/{self.readout})"
        return f"-> {d.label.upper():6s} [{spikes}]{filling}"
