"""Generate the simulated drone training set, split over several machines.

Everyone runs the same plan; each machine takes every N-th flight, so nobody duplicates work:

    python -m reflex.sim.drone_dataset --plan                       # what will be generated
    python -m reflex.sim.drone_dataset --worker 0 --workers 4       # on machine 1
    python -m reflex.sim.drone_dataset --worker 1 --workers 4       # on machine 2, ...

Then copy everyone's data/sim_drone_ds/ into one folder (file names never collide).
Interrupted? Run the same command again: finished flights are skipped.

The mix (per drone behavior of reflex/sim/drone_behaviors.py, seeds = flights):
  still   robot motionless: the clean case, most of the data
  follow  head turns towards the drone: the whole scene makes events, as in the demo
  sweep   head moves on its own
plus "none" (no drone) with the head still and sweeping, as negatives.

Every flight gets a fixed seed and a split (train / validation / test, by seed, about
80/10/10), written to data/sim_drone_ds/splits.csv by every worker (the same file on all).
Simulation only: see reflex/sim/drone_recording.py for camera, labels and their limits.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from types import SimpleNamespace

from reflex.sim.drone_behaviors import BEHAVIORS
from reflex.sim.drone_recording import REPO, record

MIX = {"still": 20, "follow": 5, "sweep": 2}      # flights per drone behavior and head mode
NONE_MIX = {"still": 5, "sweep": 15}              # flights without a drone
SEED0 = 10_000                                    # away from the seeds used for trying things out


def plan() -> list[tuple[str, str, int, str]]:
    """(behavior, head, seed, split) for every flight, the same on every machine."""
    jobs = []
    for b in BEHAVIORS:
        mix = NONE_MIX if b == "none" else MIX
        for head, n in mix.items():
            for k in range(n):
                jobs.append((b, head, k))
    out = []
    for i, (b, head, k) in enumerate(jobs):
        split = "test" if i % 10 == 9 else "validation" if i % 10 == 4 else "train"
        out.append((b, head, SEED0 + i, split))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", type=int, default=0, help="this machine's number, 0 .. workers-1")
    ap.add_argument("--workers", type=int, default=1, help="how many machines share the plan")
    ap.add_argument("--plan", action="store_true", help="print the plan and exit")
    ap.add_argument("--data", default=str(REPO / "data" / "sim_drone_ds"))
    ap.add_argument("--video", action="store_true", help="also write a video per flight (slower)")
    args = ap.parse_args()
    jobs = plan()
    if args.plan:
        counts = {}
        for b, head, _, split in jobs:
            counts[(b, head)] = counts.get((b, head), 0) + 1
        for (b, head), n in counts.items():
            print(f"{b:13s} {head:7s} {n:3d} flights")
        sp = {s: sum(1 for j in jobs if j[3] == s) for s in ("train", "validation", "test")}
        print(f"total {len(jobs)} flights (about {len(jobs) * 9 / 60:.0f} min of simulated time); "
              f"split {sp}")
        return
    if not 0 <= args.worker < args.workers:
        raise SystemExit("--worker must be 0 .. workers-1")

    from reflex.events.h5 import write_bin, write_h5
    data = Path(args.data)
    data.mkdir(parents=True, exist_ok=True)
    with open(data / "splits.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "behavior", "head", "seed", "split"])
        for b, head, seed, split in jobs:
            w.writerow([f"{b}_{head}_{seed}.h5", b, head, seed, split])
    rec_args = SimpleNamespace(duration=0.0, render_hz=500.0, frame_ms=40.0, no_video=not args.video,
                               out=str(REPO / "out" / "sim_drone_ds"), data=str(data), live_fn=None, keep_partial=False, calm=False, scene="hall")
    Path(rec_args.out).mkdir(parents=True, exist_ok=True)
    mine = [j for i, j in enumerate(jobs) if i % args.workers == args.worker]
    todo = [j for j in mine if not (data / f"{j[0]}_{j[1]}_{j[2]}.h5").exists()]
    print(f"worker {args.worker}/{args.workers}: {len(mine)} flights, {len(mine) - len(todo)} done, {len(todo)} to go")
    t0 = time.perf_counter()
    for n, (b, head, seed, _) in enumerate(todo, 1):
        record(b, head, seed, rec_args, write_h5, write_bin)
        left = (time.perf_counter() - t0) / n * (len(todo) - n)
        print(f"  {n}/{len(todo)} done, about {left / 60:.0f} min left", flush=True)


if __name__ == "__main__":
    main()
