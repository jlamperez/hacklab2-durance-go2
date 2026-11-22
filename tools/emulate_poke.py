"""What the board should answer to tools/spleat_poke.py, computed on the laptop.

Feeds the same synthetic frames as spleat_poke.py (same pattern, fraction and seed, both
polarity channels equal, value 1.0) to a reflex.river checkpoint, frame after frame with the
neuron state kept (as spleat-server does), and prints the output spikes per frame: in float
and in riverops' fixed-point emulation with BatchNorm folded and the per-layer rescale, which
is what weights.txt holds. Compare with spleat_poke.py's `values` right after
`systemctl restart spleat-server` (the server keeps neuron state across connections).

    pixi run --manifest-path tools/river/pixi.toml python tools/emulate_poke.py \
        models/zones3/best.pt --pattern dense --fraction 0.05 --count 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "tools"), str(ROOT)]

from riverops.nn import FixedPointFormat, QuantConfig  # noqa: E402
from spleat_poke import make_frame  # noqa: E402

from reflex.river.model import ModelConfig, TargetSNN  # noqa: E402
from reflex.river.train import fold_for_board  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--pattern", choices=("zeros", "pixel", "blob", "dense", "ones"), default="dense")
    ap.add_argument("--fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--count", type=int, default=10)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    n_classes = int(meta["n_classes"])
    size = meta.get("frame", {}).get("sensor", 320) // meta.get("frame", {}).get("downsample", 4)
    cfg = None
    if "model" in meta:
        cfg = ModelConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in meta["model"].items()})
    model = TargetSNN(n_classes, size, cfg)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    qcfg = QuantConfig(weight_format=FixedPointFormat(total_bits=8, frac_bits=7),
                       potential_format=FixedPointFormat(total_bits=16, frac_bits=12),
                       enable_weight_quant=True, enable_potential_quant=True)
    fixed, _ = fold_for_board(model, qcfg)

    frame = torch.tensor(make_frame(args.pattern, size, args.fraction, args.seed, 2), dtype=torch.float32)
    x = frame[None, None].repeat(1, args.count, 1, 1, 1)
    print(f"pattern {args.pattern}: {int(frame.count_nonzero())} active pixels of {frame.numel()}, "
          f"{args.count} frames, {n_classes} outputs")
    with torch.no_grad():
        for name, net in (("float", model), ("fixed point (as weights.txt)", fixed)):
            out = net(x)[0].int().tolist()
            print(f"{name}:")
            for k, v in enumerate(out):
                print(f"  frame {k}: values {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
