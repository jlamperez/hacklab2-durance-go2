"""Small convolutional spiking network in plain PyTorch.

Deliberately minimal so it can be ported to Durance's RIVERLib, whose supported
layers we have not seen yet. Every choice below is an ASSUMPTION to confirm at the
opening, and each is a config field rather than hard-coded:

  - neuron: leaky integrate-and-fire, reset by subtraction, fixed threshold
  - layers: convolution, linear; no normalization layers, no pooling inside
  - input: binary spikes (an event either happened in that cell and time step or not)
  - readout: last-layer spikes averaged over a coarse grid (keeps left/right), then a
    non-spiking leaky integrator averaged over time

Trained with surrogate gradients: the spike is a hard step in the forward pass and
a smooth function in the backward pass.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass
class SNNConfig:
    in_channels: int = 2
    channels: tuple[int, ...] = (16, 32, 64)
    kernel: int = 3
    readout_grid: tuple[int, int] = (3, 4)   # last spikes are averaged over this coarse grid
    n_classes: int = 4
    beta: float = 0.9            # membrane leak per time step
    threshold: float = 1.0
    surrogate_slope: float = 5.0
    binary_input: bool = True    # clip event counts to {0, 1}

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SNNConfig":
        d = dict(d)
        for key in ("channels", "readout_grid"):
            if key in d:
                d[key] = tuple(d[key])
        return cls(**d)


class _SpikeFn(torch.autograd.Function):
    """Heaviside forward, fast-sigmoid derivative backward."""

    @staticmethod
    def forward(ctx, v, slope):
        ctx.save_for_backward(v)
        ctx.slope = slope
        return (v > 0).to(v.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (v,) = ctx.saved_tensors
        return grad_out / (1.0 + ctx.slope * v.abs()) ** 2, None


class LIF(nn.Module):
    """Leaky integrate-and-fire layer with subtractive reset. State is passed explicitly."""

    def __init__(self, beta: float, threshold: float, slope: float):
        super().__init__()
        self.beta, self.threshold, self.slope = beta, threshold, slope

    def forward(self, current: torch.Tensor, mem: torch.Tensor | None):
        mem = current if mem is None else self.beta * mem + current
        spikes = _SpikeFn.apply(mem - self.threshold, self.slope)
        return spikes, mem - spikes * self.threshold


class ReflexSNN(nn.Module):
    def __init__(self, cfg: SNNConfig | None = None):
        super().__init__()
        self.cfg = cfg or SNNConfig()
        c = self.cfg
        chans = (c.in_channels, *c.channels)
        self.convs = nn.ModuleList(
            nn.Conv2d(chans[i], chans[i + 1], c.kernel, stride=2, padding=c.kernel // 2)
            for i in range(len(c.channels)))
        self.lifs = nn.ModuleList(LIF(c.beta, c.threshold, c.surrogate_slope) for _ in c.channels)
        self.pool = nn.AdaptiveAvgPool2d(c.readout_grid)
        self.head = nn.Linear(c.channels[-1] * c.readout_grid[0] * c.readout_grid[1], c.n_classes)

    def forward(self, x: torch.Tensor, return_stats: bool = False):
        """x: (B, T, 2, H, W) event counts. Returns logits (B, n_classes).

        With `return_stats`, also returns the mean firing rate of each spiking layer
        (fraction of neurons that fire per time step): lower means sparser.
        """
        c = self.cfg
        x = x.float()
        if c.binary_input:
            x = (x > 0).float()
        mems = [None] * len(self.convs)
        readout, out_sum = None, 0.0
        rates = [0.0] * len(self.convs)
        n_steps = x.shape[1]
        for t in range(n_steps):
            h = x[:, t]
            for i, (conv, lif) in enumerate(zip(self.convs, self.lifs)):
                h, mems[i] = lif(conv(h), mems[i])
                if return_stats:
                    rates[i] = rates[i] + h.mean().detach()
            cur = self.head(self.pool(h).flatten(1))
            readout = cur if readout is None else c.beta * readout + cur
            out_sum = out_sum + readout
        logits = out_sum / n_steps
        if return_stats:
            return logits, [float(r / n_steps) for r in rates]
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
