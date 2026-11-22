"""Readout of a target-zone SNN's output: which zone wins.

The board (spleat-server) sends, per camera frame, one spike count per output neuron. Two
networks are read this way:

- models/zones3 (reflex/river/): `none, left, center, right`, 10 ms frames. Trained to be
  read like this (reflex/river/model.py, `readout_counts`): sum each neuron's spikes over the
  last `readout` frames (20 = 200 ms) and take the neuron with the most.
- models/ball_v3 ball_zones (a teammate's train_ball_zones.py): `upper_left ... lower_right,
  none` (2 rows x 3 columns, none LAST), 5 ms frames. Trained with `time_steps` (4) steps per
  frame and the state reset every frame; its rule (`ball_zones_readout.json`) is argmax of the
  spike counts, no spikes -> none. The board runs one step per frame with the state carried
  over, so reading the last `time_steps` frames is the closest match, not the same thing.

Ties go to `none`, wherever it sits in the list.

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
    index: int             # its index (none_index = none)
    counts: np.ndarray     # spikes per class over the window
    frames: int            # frames in the window so far (< readout while filling)


class ZonesReadout:
    def __init__(self, names: list[str], readout: int = 20, margin: float = 0.0):
        if not names or readout < 1:
            raise ValueError("need class names and readout >= 1")
        if "none" not in names:
            raise ValueError(f"no 'none' class in {names}")
        self.names, self.readout, self.margin = list(names), readout, margin
        self.none_index = self.names.index("none")
        self.window: deque[np.ndarray] = deque(maxlen=readout)

    @classmethod
    def from_model_dir(cls, model_dir: Path, margin: float = 0.0, readout: int = 0) -> "ZonesReadout":
        """Class names and readout length from the model folder; `readout` > 0 overrides it.

        meta.json (reflex/river training runs: names, window.readout) or exactly one
        *_readout.json (riverops exports such as models/ball_v3: classes, time_steps).
        """
        model_dir = Path(model_dir)
        if (model_dir / "meta.json").is_file():
            meta = json.loads((model_dir / "meta.json").read_text())
            names, n = meta["names"], int(meta["window"]["readout"])
        else:
            found = sorted(model_dir.glob("*_readout.json"))
            if len(found) != 1:
                raise FileNotFoundError(f"{model_dir}: no meta.json and {len(found)} *_readout.json")
            meta = json.loads(found[0].read_text())
            names, n = meta["classes"], int(meta["time_steps"])
        return cls(names, readout if readout > 0 else n, margin)

    def reset(self) -> None:
        self.window.clear()

    def push(self, values) -> ZoneDecision:
        """Add one frame's output (one value per class, any shape that flattens to that)."""
        v = np.asarray(values, dtype=np.float64).reshape(-1)
        if v.size != len(self.names):
            raise ValueError(f"{v.size} values, expected {len(self.names)} ({', '.join(self.names)})")
        self.window.append(v)
        counts = np.sum(self.window, axis=0)
        zones = [i for i in range(len(counts)) if i != self.none_index]
        zone = max(zones, key=lambda i: counts[i]) if zones else self.none_index   # first of equals
        none = self.none_index
        index = zone if counts[zone] >= counts[none] + max(self.margin, 1e-9) else none
        return ZoneDecision(self.names[index], index, counts, len(self.window))

    def format(self, d: ZoneDecision) -> str:
        spikes = " ".join(f"{n} {int(c)}" for n, c in zip(self.names, d.counts))
        filling = "" if d.frames >= self.readout else f" (window {d.frames}/{self.readout})"
        return f"-> {d.label.upper():6s} [{spikes}]{filling}"


def reflex_label(name: str) -> str:
    """Class name -> the behavior node's label (none / left / center / right), by column.

    `upper_left` and `lower_left` -> left, `*_middle` -> center, `*_right` -> right. Columns are
    image columns; with the camera looking forward and not mirrored, image left is the dog's
    left (as for models/zones3, reflex/river/data.py). Not checked for ball_v3's training data.
    """
    for key, label in (("left", "left"), ("middle", "center"), ("center", "center"), ("right", "right")):
        if name == key or name.endswith("_" + key):
            return label
    return "none"
