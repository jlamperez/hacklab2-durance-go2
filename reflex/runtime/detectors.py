"""Detectors: turn one event window into a decision.

Two interchangeable implementations with the same `predict(window, t_us)`:

  SNNDetector     the trained spiking network
  SimpleDetector  no learning: a burst of events in the lower part of the image is a threat

The simple detector is the fallback if the SNN cannot be deployed, and the baseline
for the pitch: it reacts to anything that moves, the SNN should not.
"""
from __future__ import annotations

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
        if score < 1.0:
            return Decision("none", 1.0 - 0.5 * score, t_us)
        return Decision(LABELS[1 + k], min(1.0, 0.5 * score), t_us)


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

    def predict(self, window: np.ndarray, t_us: int = 0) -> Decision:
        torch = self._torch
        with torch.no_grad():
            logits = self.model(torch.from_numpy(window)[None].to(self.device))
            prob = torch.softmax(logits[0], dim=0).cpu().numpy()
        k = int(prob.argmax())
        return Decision(self.labels[k], float(prob[k]), t_us)
