"""What the board should answer to tools/spleat_poke.py, for any riverops export.

Like tools/emulate_poke.py, but reads the two files the board gets (`*_model.json`,
`*_parameters.json`) through reflex.runtime.river_net.RiverNet, so it works for networks
not trained with reflex.river (models/ball_v3/ball_zones_*). Feeds the same synthetic frame
as spleat_poke.py (same pattern, fraction and seed, value 1.0 in both polarity channels),
one step per frame with the neuron state kept, as spleat-server does.

spleat_poke.py sends the frame as CHW bytes; the FPGA reads them as HWC
(docs/fpga-zones3-2026-11-22.md, section 12). Both readings are printed. Compare with
spleat_poke.py's `values` right after `systemctl restart spleat-server` (the server keeps
neuron state across connections).

    pixi run python tools/emulate_river_poke.py models/ball_v3/ball_zones_model.json \
        models/ball_v3/ball_zones_parameters.json --pattern dense --fraction 0.05 --count 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "tools"), str(ROOT)]

from spleat_poke import make_frame  # noqa: E402

from reflex.runtime.river_net import RiverNet  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("parameters", type=Path)
    ap.add_argument("--pattern", choices=("zeros", "pixel", "blob", "dense", "ones"), default="dense")
    ap.add_argument("--fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--count", type=int, default=10)
    args = ap.parse_args()

    net = RiverNet(args.model, args.parameters)
    c, h, w = net.input_shape
    if h != w:
        raise SystemExit(f"input {c}x{h}x{w} is not square; spleat_poke.py sends square frames")
    chw = make_frame(args.pattern, h, args.fraction, args.seed, c)       # what spleat_poke sends
    hwc = chw.reshape(-1).reshape(h, w, c).transpose(2, 0, 1)            # those bytes read as HWC
    print(f"pattern {args.pattern}: {int((chw != 0).sum())} active values of {chw.size}, "
          f"{args.count} frames, output {'x'.join(map(str, net.output_shape))}")
    for name, frame in (("bytes read as CHW", chw), ("bytes read as HWC (board)", hwc)):
        out = net.run_steps(frame.copy(), args.count)
        print(f"== {name}")
        for k, v in enumerate(out):
            print(f"frame {k}: values {[int(a) for a in v.reshape(-1)]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
