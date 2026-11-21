"""Compare the SNN with the no-learning detector on held-out windows.

    pixi run eval                      # data/synth, out/models/snn.pt

Both see exactly the same validation windows (same episode split as training). The
simple detector's threshold is calibrated on idle windows from the training split,
the way it will be calibrated on the real dog standing still.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from reflex.runtime.detectors import LABELS, SimpleDetector, SNNDetector
from reflex.train import load_shards, split_by_episode


def score(pred: np.ndarray, y: np.ndarray, kind: np.ndarray, kinds: list[str]) -> dict:
    neg = y == 0
    out = {
        "accuracy": float((pred == y).mean()),
        "missed_threat": float((pred[~neg] == 0).mean()),
        "wrong_side": float(((pred[~neg] != 0) & (pred[~neg] != y[~neg])).mean()),
        "false_alarm_all": float((pred[neg] != 0).mean()),
    }
    for i, k in enumerate(kinds):
        m = neg & (kind == i)
        if m.any():
            out[f"false_alarm_{k}"] = float((pred[m] != 0).mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/synth")
    ap.add_argument("--model", default="out/models/snn.pt")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    data, meta = load_shards(args.data)
    kinds = meta["kinds"]
    tr, va = split_by_episode(data["episode"], args.val_frac, args.seed)
    x, y, kind = data["x"][va], data["y"][va], data["kind"][va]

    simple = SimpleDetector()
    idle = data["x"][tr & (data["kind"] == kinds.index("idle"))][:300]
    thr = simple.calibrate(idle)
    results = {"simple": score(np.array([LABELS.index(simple.predict(w).label) for w in x]), y, kind, kinds)}

    if Path(args.model).exists():
        snn = SNNDetector(args.model, device=args.device)
        results["snn"] = score(np.array([LABELS.index(snn.predict(w).label) for w in x]), y, kind, kinds)

    names = list(results)
    print(f"{len(y)} held-out SIMULATED windows; simple-detector threshold {thr:.0f} events\n")
    print(f"{'metric':28s}" + "".join(f"{n:>10s}" for n in names))
    for key in results[names[0]]:
        print(f"{key:28s}" + "".join(f"{results[n][key] * 100:9.1f}%" for n in names))
    Path("out").mkdir(exist_ok=True)
    Path("out/eval.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
