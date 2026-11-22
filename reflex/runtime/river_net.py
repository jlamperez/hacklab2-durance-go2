"""Run a riverops export (River Model Graph + fixed-point parameters) in plain PyTorch.

This is a software stand-in for the network on SPLEAT, the FPGA accelerator: it reads
the same two files the board gets (`*_model.json`, `*_parameters.json`) and computes in
the same integers, so its output can later be compared with the board's.

Arithmetic, from riverops 0.2.0 (`riverops/nn/spiking_conv.py`, fixed-point forward):
  weights, biases  int8, Q1.7    (value = int / 128)
  potentials       int16, Q4.12  (value = int / 4096); threshold and leak in the same format
  per time step and neuron, everything in units of 1/4096:
    current   = 32 * (sum(input * w) + b)          exact: (w/128 + b/128) * 4096
    leak      = round(v * leak / 4096)
    v         = clamp(clamp(v - leak) + current)
    spike     = v >= threshold
    v         = 0 where it spiked                   HARD reset
Assumptions that the two files do not record (see models/ball/PROVENANCE.md): the reset
mode (HARD, as the ball training script uses) and how the hardware rounds the leak
(here round-half-to-even, as torch.round in riverops; a hardware shift would floor).

Graph semantics (checked by `reflex.deploy.spleat_bundle.graph_shapes`): a node's input
is the sum of the spikes of its sources, upsampled (nearest) by `upscaling`, then
convolved. The network state starts at zero for every frame, as in training.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from reflex.deploy.spleat_bundle import (POTENTIAL_BITS, POTENTIAL_FRAC, WEIGHT_FRAC,
                                         check_parameters, graph_shapes)

_PMIN, _PMAX = -(1 << (POTENTIAL_BITS - 1)), (1 << (POTENTIAL_BITS - 1)) - 1
_CURRENT_SCALE = 1 << (POTENTIAL_FRAC - WEIGHT_FRAC)     # 32: Q1.7 products -> Q4.12


class RiverNet:
    def __init__(self, model_path: str | Path, params_path: str | Path, device: str = "cpu"):
        model = json.loads(Path(model_path).read_text())
        params = json.loads(Path(params_path).read_text())
        shapes = graph_shapes(model)
        check_parameters(model, params, shapes)
        self.device = torch.device(device)
        self.input_shape = shapes["input"]["out"]                  # (C, H, W)
        self.output = model["outputs"][0]
        self.output_shape = shapes[self.output]["out"]
        self.layers = []
        for i, node in enumerate(model["nodes"]):
            name = node["node_index"]
            c_out, (c_in, _, _) = node["channels"], shapes[name]["in"]
            k = node["kernel_size"]
            w = torch.tensor(params[f"weights_{i}"], dtype=torch.float32).reshape(c_out, c_in, k, k)
            b = torch.tensor(params[f"bias_{i}"], dtype=torch.float32)
            pw0, pw1, ph0, ph1 = node["padding"]
            self.layers.append({
                "name": name, "sources": shapes[name]["sources"], "up": node["upscaling"],
                "stride": node["stride"], "pad": (pw0, pw1, ph0, ph1),
                "w": w.to(self.device), "b": b.to(self.device),
                "threshold": float(params[f"threshold_{i}"]), "leak": float(params[f"leak_{i}"]),
            })

    @torch.no_grad()
    def run(self, frames: np.ndarray | torch.Tensor, time_steps: int) -> np.ndarray:
        """Spike counts of the output node summed over `time_steps`, shape (C, H, W).

        `frames`: (C, H, W) fed at every step, or (T, C, H, W) one per step.
        """
        return self.run_steps(frames, time_steps).sum(axis=0)

    @torch.no_grad()
    def run_steps(self, frames: np.ndarray | torch.Tensor, time_steps: int) -> np.ndarray:
        """Output spikes of every step, shape (T, C, H, W), state kept from step to step.

        With one frame per step this is what spleat-server does on the board: one step per
        frame, neuron state carried over (docs/fpga-zones3-2026-11-22.md).
        """
        x = torch.as_tensor(frames, dtype=torch.float32, device=self.device)
        if x.dim() == 3:
            x = x.expand(time_steps, *x.shape)
        if tuple(x.shape[1:]) != tuple(self.input_shape) or x.shape[0] != time_steps:
            raise ValueError(f"frames {tuple(x.shape)}, expected ({time_steps}, {self.input_shape})")
        x = x[:, None]                                   # (T, 1, C, H, W): batch of one
        spikes = {"input": x}
        for layer in self.layers:
            inp = sum(spikes[s] for s in layer["sources"])             # (T, 1, C, H, W)
            t, n = inp.shape[:2]
            flat = inp.flatten(0, 1)
            if layer["up"] > 1:
                flat = F.interpolate(flat, scale_factor=layer["up"], mode="nearest")
            flat = F.pad(flat, layer["pad"])
            current = F.conv2d(flat, layer["w"], layer["b"], stride=layer["stride"])
            current = (current * _CURRENT_SCALE).clamp(_PMIN, _PMAX).unflatten(0, (t, n))
            v = torch.zeros_like(current[0])
            out = []
            for step in range(t):
                leak = torch.round(v * layer["leak"] / (1 << POTENTIAL_FRAC))
                v = ((v - leak).clamp(_PMIN, _PMAX) + current[step]).clamp(_PMIN, _PMAX)
                s = (v >= layer["threshold"]).to(v.dtype)
                v = v * (1.0 - s)
                out.append(s)
            spikes[layer["name"]] = torch.stack(out)
        return spikes[self.output][:, 0].cpu().numpy()
