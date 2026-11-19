"""Generate the synthetic dataset: simulated episodes cut into labelled windows.

    pixi run gen-data                          # 2000 episodes into data/synth
    pixi run gen-data --episodes 200 --out data/tiny

Writes one .npz shard per worker with:
  x        uint8 (N, T, 2, H, W)   event counts per time step
  y        int64 (N,)              index into LABELS
  episode  int64 (N,)              episode id, used to split train/validation
  kind     int64 (N,)              index into KINDS (which negative it came from)
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from reflex.data.episodes import KINDS, run_episode
from reflex.data.windows import WindowConfig, make_window
from reflex.sim.scene import LABELS, Go2Sim

STRIDE_MS = 20          # spacing between candidate windows
MAX_POS, MAX_NEG = 12, 8  # windows kept per episode


def _worker(args) -> dict:
    worker_id, episode_ids, seed, out_dir, wcfg = args
    rng = np.random.default_rng([seed, worker_id])
    sim = Go2Sim(seed=seed * 1000 + worker_id, eye_hw=(wcfg.height, wcfg.width))
    xs, ys, eps, kinds = [], [], [], []
    for ep_id in episode_ids:
        ep = run_episode(sim, rng)
        ends = np.arange(int(wcfg.window_ms * 1000) + 10_000, int(ep.t_us[-1]), STRIDE_MS * 1000)
        labels = np.array([LABELS.index(ep.label_at(int(t))) for t in ends])
        pos, neg = np.nonzero(labels > 0)[0], np.nonzero(labels == 0)[0]
        keep = np.concatenate([
            rng.choice(pos, min(len(pos), MAX_POS), replace=False),
            rng.choice(neg, min(len(neg), MAX_NEG), replace=False),
        ]).astype(int)
        for k in keep:
            xs.append(make_window(ep.events, int(ends[k]), wcfg))
            ys.append(labels[k])
            eps.append(ep_id)
            kinds.append(KINDS.index(ep.kind))
    path = Path(out_dir) / f"shard_{worker_id:03d}.npz"
    np.savez_compressed(path, x=np.stack(xs), y=np.array(ys), episode=np.array(eps), kind=np.array(kinds))
    return {"shard": path.name, "windows": len(ys)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wcfg = WindowConfig()
    chunks = np.array_split(np.arange(args.episodes), args.workers)
    jobs = [(i, c.tolist(), args.seed, str(out), wcfg) for i, c in enumerate(chunks) if len(c)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        results = pool.map(_worker, jobs)
    total = sum(r["windows"] for r in results)
    (out / "meta.json").write_text(json.dumps({
        "labels": LABELS, "kinds": KINDS, "window": wcfg.to_dict(), "episodes": args.episodes,
        "windows": total, "seed": args.seed, "source": "MuJoCo + event emulator (synthetic)",
    }, indent=2))
    print(f"{args.episodes} episodes -> {total} windows in {time.time() - t0:.0f} s  ({out})")


if __name__ == "__main__":
    main()
