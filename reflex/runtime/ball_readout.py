"""Where the ball is, from the output of the ball U-Net.

The network (Training/train_ball_size.py, Training/train_ball_small.py) is meant to run on
the FPGA and returns the spikes of its head layer, [T, 9, 80, 80], or their sum over T.
This is the off-chip readout: a numpy port of `decode` in train_ball_size.py, so it runs
on the Kria without PyTorch. Its constants come from <prefix>_readout.json, written by the
export next to the model.

Head channels, on an 80x80 grid of 4x4 px cells:
  heatmap   the peak cell is the coarse position
  x_offset  votes for where the ball sits inside that cell along x, one channel per bin
  y_offset  the same along y

Not checked against the real FPGA output yet: whether the server returns spikes per step
or counts, and in which layout.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BallReadout:
    time_steps: int
    grid: int
    cell_px: float
    heatmap_channel: int
    x_offset: tuple[int, int]       # channel range [start, stop)
    y_offset: tuple[int, int]
    heatmap_logit_scale: float
    offset_logit_scale: float
    no_ball_logit: float
    ball_probability_threshold: float = 0.5

    @property
    def sensor(self) -> float:
        """Side of the square input frame, in pixels (320)."""
        return self.grid * self.cell_px

    @classmethod
    def load(cls, path: str | Path) -> BallReadout:
        r = json.loads(Path(path).read_text())
        ch = r["channels"]
        return cls(
            time_steps=int(r["time_steps"]),
            grid=int(r["grid"]),
            cell_px=float(r["cell_px"]),
            heatmap_channel=int(ch["heatmap"]),
            x_offset=tuple(ch["x_offset"]),
            y_offset=tuple(ch["y_offset"]),
            heatmap_logit_scale=float(r["heatmap_logit_scale"]),
            offset_logit_scale=float(r["offset_logit_scale"]),
            no_ball_logit=float(r["no_ball_logit"]),
            ball_probability_threshold=float(r.get("ball_probability_threshold", 0.5)),
        )


@dataclass(frozen=True)
class Ball:
    x: float        # input-frame pixels, from the left
    y: float        # input-frame pixels, from the top
    p_ball: float   # probability that a ball is in the frame
    found: bool     # p_ball >= the readout's threshold


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


def _offset(counts: np.ndarray, channels: tuple[int, int], row: int, col: int,
            readout: BallReadout) -> float:
    """Expected position inside the cell, in pixels from its top-left corner."""
    votes = counts[channels[0]:channels[1], row, col]
    probs = _softmax(votes * (readout.offset_logit_scale / readout.time_steps))
    centers = (np.arange(len(votes)) + 0.5) * (readout.cell_px / len(votes))
    return float(probs @ centers)


def decode(output: np.ndarray, readout: BallReadout) -> Ball:
    """Ball position and probability from head spikes [T, C, H, W] or counts [C, H, W]."""
    counts = np.asarray(output, dtype=np.float64)
    if counts.ndim == 4:
        counts = counts.sum(axis=0)
    heat = counts[readout.heatmap_channel]
    logits = heat.ravel() * (readout.heatmap_logit_scale / readout.time_steps)
    peak = logits.max()
    log_sum = peak + np.log(np.exp(logits - peak).sum())
    p_ball = float(0.5 * (1.0 + np.tanh((log_sum - readout.no_ball_logit) / 2)))
    row, col = divmod(int(heat.argmax()), heat.shape[1])
    return Ball(
        x=col * readout.cell_px + _offset(counts, readout.x_offset, row, col, readout),
        y=row * readout.cell_px + _offset(counts, readout.y_offset, row, col, readout),
        p_ball=p_ball,
        found=p_ball >= readout.ball_probability_threshold,
    )


def to_camera(x: float, y: float, width: int, height: int, sensor: float = 320.0) -> tuple[float, float]:
    """Input-frame pixels -> camera pixels, when the whole camera frame was fitted into the
    square input like `fit_crop` in the training scripts (centered, scaled by its larger side).
    The identity for a 320x320 camera."""
    side = float(max(width, height))
    scale = side / sensor
    return x * scale + (width - side) / 2, y * scale + (height - side) / 2
