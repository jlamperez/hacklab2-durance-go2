"""Interactive 3D viewer: watch balls being rolled at the simulated Go2 in real time.

    pixi run view                 # opens a window; close it to stop
    pixi run view --speed 0.25    # slow motion

Mouse: left-drag rotates, right-drag pans, scroll zooms. Space pauses.
This viewer shows the scene only. For the dog-eye and event-camera panels,
record a video with `pixi run sim`.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import mujoco.viewer
import numpy as np

from reflex.sim.scene import BallShot, Go2Sim


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0, help="1.0 = real time, 0.25 = slow motion")
    ap.add_argument("--period", type=float, default=3.0, help="simulated seconds between shots")
    ap.add_argument("--duration", type=float, default=0.0, help="stop after this many wall seconds (0 = until closed)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    sim = Go2Sim(seed=args.seed)
    next_shot = 0.5
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        viewer.cam.lookat[:] = [1.2, 0.0, 0.2]
        viewer.cam.distance = 4.5
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -18
        wall_start = time.perf_counter()
        sim_start = sim.time
        while viewer.is_running():
            if args.duration and time.perf_counter() - wall_start > args.duration:
                break
            # Advance physics until simulated time catches up with (scaled) wall time.
            target = sim_start + (time.perf_counter() - wall_start) * args.speed
            while sim.time < target:
                if sim.time >= next_shot:
                    shot = BallShot.sample(rng)
                    sim.launch(shot)
                    print(f"t = {sim.time:5.1f} s  shot: {shot.label:6s} {shot.speed:.1f} m/s  r = {shot.radius * 100:.0f} cm", flush=True)
                    next_shot = sim.time + args.period
                sim.step()
            viewer.sync()
            time.sleep(0.002)
        # MuJoCo 4.0.1 segfaults while tearing the passive viewer down (its render
        # thread races with glfw.terminate). Nothing needs saving, so leave directly.
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
