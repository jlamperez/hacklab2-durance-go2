"""Detectors: turn one event window into a decision.

Two interchangeable implementations with the same `predict(window, t_us)`:

  SNNDetector     the trained spiking network
  SimpleDetector  no learning: a burst of events in the lower part of the image is a threat

The simple detector is the fallback if the SNN cannot be deployed, and the baseline
for the pitch: it reacts to anything that moves, the SNN should not.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LABELS = ("none", "left", "center", "right")


@dataclass(frozen=True)
class Decision:
    label: str          # none / left / center / right, in the dog frame
    confidence: float   # 0..1
    t_us: int = 0       # time of the newest event the decision is based on

    @property
    def is_threat(self) -> bool:
        return self.label != "none"


@dataclass
class SimpleConfig:
    threshold: float = 60.0     # events in the busiest third of the zone; set by `calibrate`
    zone_top: float = 0.35      # zone = image rows below this fraction (where the floor is)
    recent_bins: int = 4        # only the newest time steps count: a burst, not a total


class SimpleDetector:
    """Threat = burst of events in the lower zone; side = which third of it is busiest."""

    def __init__(self, cfg: SimpleConfig | None = None):
        self.cfg = cfg or SimpleConfig()
        self.last: dict = {}     # details of the last prediction, for viewers

    def _thirds(self, window: np.ndarray) -> np.ndarray:
        t, _, h, w = window.shape
        zone = window[t - self.cfg.recent_bins:, :, int(h * self.cfg.zone_top):, :]
        cols = zone.astype(np.float32).sum(axis=(0, 1, 2))          # events per column
        return np.array([c.sum() for c in np.array_split(cols, 3)])  # left, center, right

    def calibrate(self, idle_windows: np.ndarray, margin: float = 1.5, percentile: float = 100.0) -> float:
        """Set the threshold above the background seen while nothing happens.

        On Saturday the idle windows come from the real dog standing still. `percentile`
        below 100 ignores the busiest windows, for live calibration where something may
        move for a moment (someone walking past, a webcam adjusting its exposure).
        """
        peaks = [float(self._thirds(w).max()) for w in idle_windows]
        self.cfg.threshold = max(float(np.percentile(peaks, percentile)) * margin, 5.0)
        return self.cfg.threshold

    def predict(self, window: np.ndarray, t_us: int = 0) -> Decision:
        thirds = self._thirds(window)
        k = int(thirds.argmax())
        score = float(thirds[k] / self.cfg.threshold)
        # Events per third of the zone, as a fraction of the threshold.
        self.last = {"scores": dict(zip(LABELS[1:], (thirds / self.cfg.threshold).round(3).tolist())),
                     "threshold": float(self.cfg.threshold)}
        if score < 1.0:
            return Decision("none", 1.0 - 0.5 * score, t_us)
        return Decision(LABELS[1 + k], min(1.0, 0.5 * score), t_us)


@dataclass
class BallConfig:
    model_dir: str = "models/ball"       # ball_model.json, ball_parameters.json, ball_readout.json
    p_ball: float | None = None          # threshold on P(ball); None = the readout's (0.5)
    device: str = "cpu"


# Window the ball network was trained on: one binary frame of the last 1 ms of events
# (Training/train_ball_size.py: ACC_TIME = 1000 us, polarity discarded).
BALL_WINDOW_MS = 1.0


class BallDetector:
    """Durance-style ball detector: the riverops ball network, run as on the FPGA.

    Input: the newest events as one binary frame, centred in the network's square input
    (a 320x180 camera gets 70 empty rows above and below, as the training script fits
    its own recordings). Output: the 9-channel 80x80 head decoded with ball_readout.json
    into (x, y, P(ball)). Decision: a ball is a threat if P(ball) passes the threshold;
    its side is the third of the image it is in (image left = dog's left).
    It detects balls, not approaching ones: a ball rolling past also counts.
    """

    def __init__(self, cfg: BallConfig | None = None):
        from reflex.runtime.river_net import RiverNet
        self.cfg = cfg or BallConfig()
        d = Path(self.cfg.model_dir)
        self.net = RiverNet(d / "ball_model.json", d / "ball_parameters.json", self.cfg.device)
        self.readout = json.loads((d / "ball_readout.json").read_text())
        r = self.readout
        c, self.side, side_w = self.net.input_shape
        if c != 1 or self.side != side_w:
            raise ValueError(f"expected a 1-channel square input, got {self.net.input_shape}")
        if tuple(self.net.output_shape[1:]) != (r["grid"], r["grid"]):
            raise ValueError(f"output {self.net.output_shape} does not match readout grid {r['grid']}")
        self.threshold = r["ball_probability_threshold"] if self.cfg.p_ball is None else self.cfg.p_ball
        self.last: dict = {}

    def frame(self, window: np.ndarray) -> tuple[np.ndarray, int, int]:
        """Binary (1, S, S) frame from a (T, 2, H, W) window, and where the image sits in it."""
        img = window.sum(axis=(0, 1)) > 0
        h, w = img.shape
        if h > self.side or w > self.side:
            raise ValueError(f"window {h}x{w} is larger than the {self.side}x{self.side} input")
        y0, x0 = (self.side - h) // 2, (self.side - w) // 2
        out = np.zeros((1, self.side, self.side), dtype=np.float32)
        out[0, y0:y0 + h, x0:x0 + w] = img
        return out, x0, y0

    def decode(self, counts: np.ndarray) -> tuple[float, float, float]:
        """(x, y, P(ball)) in input pixels, as `decode` in Training/train_ball_size.py."""
        r = self.readout
        t, cell, bins = r["time_steps"], r["cell_px"], r["offset_bins"]
        heat = counts[r["channels"]["heatmap"]].ravel() * (r["heatmap_logit_scale"] / t)
        m = heat.max()
        p_ball = 1.0 / (1.0 + np.exp(-(m + np.log(np.exp(heat - m).sum()) - r["no_ball_logit"])))
        row, col = divmod(int(heat.argmax()), r["grid"])
        centers = (np.arange(bins) + 0.5) * cell / bins

        def offset(key):
            a, b = r["channels"][key]
            z = counts[a:b, row, col] * (r["offset_logit_scale"] / t)
            p = np.exp(z - z.max())
            return float((p / p.sum() * centers).sum())

        return col * cell + offset("x_offset"), row * cell + offset("y_offset"), float(p_ball)

    def predict(self, window: np.ndarray, t_us: int = 0) -> Decision:
        frame, x0, y0 = self.frame(window)
        counts = self.net.run(frame, self.readout["time_steps"])
        x, y, p = self.decode(counts)
        w = window.shape[-1]
        x_img, y_img = x - x0, y - y0
        side = LABELS[1 + min(2, max(0, int(3 * x_img / w)))]
        self.last = {"p_ball": round(p, 4), "x": round(x_img, 1), "y": round(y_img, 1),
                     "side": side, "threshold": self.threshold,
                     "heat_spikes": int(counts[self.readout["channels"]["heatmap"]].sum())}
        if p < self.threshold:
            return Decision("none", 1.0 - p, t_us)
        return Decision(side, p, t_us)


class SNNDetector:
    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        import torch

        from reflex.models.snn import ReflexSNN, SNNConfig
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = ckpt["model_cfg"]
        self.labels = tuple(ckpt["labels"])
        self.window_cfg = ckpt["window"]
        self.model = ReflexSNN(SNNConfig.from_dict(cfg))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval().to(device)
        self.device, self._torch = device, torch
        self.last: dict = {}     # details of the last prediction, for viewers

    def predict(self, window: np.ndarray, t_us: int = 0) -> Decision:
        torch = self._torch
        with torch.no_grad():
            logits, rates = self.model(torch.from_numpy(window)[None].to(self.device), return_stats=True)
            prob = torch.softmax(logits[0], dim=0).cpu().numpy()
        k = int(prob.argmax())
        # firing_rate: fraction of neurons that fire per time step, one value per layer.
        self.last = {"probs": dict(zip(self.labels, prob.round(4).tolist())), "firing_rate": rates}
        return Decision(self.labels[k], float(prob[k]), t_us)
