"""Conv-only spiking network for RIVER, built from riverops layers.

Same building block and neuron settings as Durance's Motion Buddy notebook
(`SpikingConvBN2d`, LIF with fixed leak, hard reset). Stride-2 convolutions shrink
the 80x80 input to 5x5; the last convolution covers the whole 5x5 and gives one
output channel per class at 1x1, the output shape the default spleat-server
config uses (`outputs: channels C, height 1, width 1`).

Changing anything here changes the architecture, which needs a new bitstream from
riversdk.dev. Retraining with the same architecture only changes weights.txt.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    channels: tuple[int, ...] = (16, 32, 32, 64)   # one stride-2 conv each
    decay: float = 0.93                            # membrane kept per 10 ms step (notebook value)
    v_threshold: float = 1.0
    reset_mode: str = "HARD"

    def to_dict(self) -> dict:
        return asdict(self)


def _out_size(n: int, k: int, s: int, p: int) -> int:
    return (n + 2 * p - k) // s + 1


class TargetSNN(nn.Module):
    """Input (B, T, 2, S, S). Output: spikes per frame and class, (B, T, n_classes)."""

    def __init__(self, n_classes: int, input_size: int = 80, cfg: ModelConfig | None = None):
        super().__init__()
        from riverops.nn import SpikingConvBN2d

        self.cfg = cfg or ModelConfig()
        lif = {"initial_tau": 1.0 / (1.0 - self.cfg.decay), "learnable_leak": False,
               "v_threshold": self.cfg.v_threshold, "reset_mode": self.cfg.reset_mode}
        layers, c_in, size = [], 2, input_size
        for i, c in enumerate(self.cfg.channels):
            k, p = (5, 2) if i == 0 else (3, 1)
            layers.append(SpikingConvBN2d(c_in, c, kernel_size=k, stride=2, padding=p, bias=False,
                                          first_layer=(i == 0), **lif))
            c_in, size = c, _out_size(size, k, 2, p)
        layers.append(SpikingConvBN2d(c_in, n_classes, kernel_size=size, bias=False, **lif))
        self.net = nn.Sequential(*layers)
        self.n_classes, self.input_size, self.final_kernel = n_classes, input_size, size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(start_dim=2)


def spike_targets(labels: torch.Tensor, n_classes: int, steps: int,
                  soft_neighbours: bool = True) -> torch.Tensor:
    """Target spike counts, as in the notebook: 80 % of the steps for the true class, 20 % for
    the others. With soft_neighbours, the zones next to the true zone get 50 % (Ziegler et al.
    2024 use 1 / 0.5 / 0); `none` (class 0) has no neighbours."""
    t = torch.full((len(labels), n_classes), 0.2 * steps, device=labels.device)
    rows = torch.arange(len(labels), device=labels.device)
    if soft_neighbours:
        for d in (-1, 1):
            nb = labels + d
            ok = (labels > 0) & (nb >= 1) & (nb < n_classes)
            t[rows[ok], nb[ok]] = 0.5 * steps
    t[rows, labels] = 0.8 * steps
    return t


def readout_counts(spikes: torch.Tensor, readout: int) -> torch.Tensor:
    """Spike count per class over the last `readout` frames: what the ROS node accumulates."""
    return spikes[:, -readout:].sum(dim=1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
