"""Interactive 3D viewer: watch balls being rolled at the simulated Go2 in real time.

    pixi run view                 # opens a window; close it to stop
    pixi run view --speed 0.25    # slow motion

Mouse: left-drag rotates, right-drag pans, scroll zooms. Space pauses.
Pose keys: X stand, C crouch, Q lean left, E lean right, Z sit, H hello (wave a paw). Chosen among the letters whose
built-in viewer shortcut can be undone (S, W, R, L, K, G toggle render options that cannot).
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

from reflex.sim.scene import POSES, BallShot, Go2Sim  # noqa: F401 (POSES: see tests)

# Key -> pose. Keys arrive as GLFW key codes, which for letters are the uppercase ASCII code.
KEY_POSES = {"X": "stand", "C": "crouch", "Q": "lean_left", "E": "lean_right"}
KEY_TRICKS = {"Z": "sit", "H": "hello"}     # slow tricks, see MOTIONS in scene.py


def pose_for_key(keycode: int) -> str | None:
    """Pose selected by a key press, or None if the key is not a pose key."""
    if not 0 <= keycode < 256:
        return None
    return KEY_POSES.get(chr(keycode).upper()) or KEY_TRICKS.get(chr(keycode).upper())


# GLFW key codes
KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP, KEY_SPACE = 262, 263, 264, 265, 32
DRIVE_STEP = {KEY_UP: (0.2, 0, 0), KEY_DOWN: (-0.2, 0, 0), KEY_LEFT: (0, 0, 0.3), KEY_RIGHT: (0, 0, -0.3),
              ord("A"): (0, 0.15, 0), ord("D"): (0, -0.15, 0)}
DRIVE_LIMITS = ((-0.5, 1.0), (-0.45, 0.45), (-0.9, 0.9))


def drive_command(keycode: int, command) -> tuple[float, float, float] | None:
    """New (vx, vy, yaw rate) after a key press, or None if the key does not drive.

    Each press adds a step, like a throttle; space stops. Keys are only seen when
    pressed (the viewer does not report releases), so speed stays until changed.
    """
    if keycode == KEY_SPACE:
        return (0.0, 0.0, 0.0)
    step = DRIVE_STEP.get(keycode)
    if step is None:
        return None
    out = []
    for c, d, (lo, hi) in zip(command, step, DRIVE_LIMITS):
        v = round(min(max(float(c) + d, lo), hi), 2)
        out.append(0.0 if abs(v) < 0.01 else v)
    return tuple(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0, help="1.0 = real time, 0.25 = slow motion")
    ap.add_argument("--period", type=float, default=3.0, help="simulated seconds between shots")
    ap.add_argument("--no-balls", action="store_true", help="do not roll balls; pose keys only")
    ap.add_argument("--drive", action="store_true",
                    help="load the walking policy and drive the dog with the arrow keys")
    ap.add_argument("--duration", type=float, default=0.0, help="stop after this many wall seconds (0 = until closed)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    sim = Go2Sim(seed=args.seed)
    next_shot = float("inf") if args.no_balls else 0.5
    pending: list[str] = []   # filled by the viewer thread, drained by the main loop

    if args.drive:
        sim.enable_walking()

    def on_key(keycode: int) -> None:
        pose = pose_for_key(keycode)
        if pose is not None:
            pending.append(pose)
        elif args.drive:
            cmd = drive_command(keycode, sim.command)
            if cmd is not None:
                pending.append(cmd)

    print("pose keys: X stand, C crouch, Q lean left, E lean right, Z sit, H hello (wave a paw)", flush=True)
    if args.drive:
        print("drive keys: UP/DOWN faster/slower, LEFT/RIGHT turn, A/D sidestep, SPACE stop "
              "(each press adds speed)", flush=True)
    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=on_key) as viewer:
        viewer.cam.lookat[:] = [1.2, 0.0, 0.2]
        viewer.cam.distance = 4.5
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -18
        if args.drive:      # follow the dog
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = sim.model.body("base").id
            viewer.cam.distance, viewer.cam.azimuth = 3.0, 150
        # The viewer has its own shortcuts on the same letters (A: auto-connect, D: hide
        # static bodies, C: contact points). They cannot be disabled, so its display
        # options are put back every frame.
        vis_flags = viewer.opt.flags.copy()
        wall_start = time.perf_counter()
        sim_start = sim.time
        while viewer.is_running():
            if args.duration and time.perf_counter() - wall_start > args.duration:
                break
            # Advance physics until simulated time catches up with (scaled) wall time.
            target = sim_start + (time.perf_counter() - wall_start) * args.speed
            while pending:
                pose = pending.pop(0)
                if isinstance(pose, tuple):
                    sim.set_velocity(*pose)
                    print(f"t = {sim.time:5.1f} s  walk: forward {pose[0]:+.2f} m/s  left {pose[1]:+.2f} m/s  "
                          f"turn {pose[2]:+.2f} rad/s", flush=True)
                    continue
                if pose in KEY_TRICKS.values():
                    sim.play(pose)
                else:
                    sim.set_pose(pose)
                print(f"t = {sim.time:5.1f} s  pose: {pose}", flush=True)
            while sim.time < target:
                if sim.time >= next_shot:
                    shot = BallShot.sample(rng)
                    sim.launch(shot)
                    print(f"t = {sim.time:5.1f} s  shot: {shot.label:6s} {shot.speed:.1f} m/s  r = {shot.radius * 100:.0f} cm", flush=True)
                    next_shot = sim.time + args.period
                sim.step()
            with viewer.lock():
                viewer.opt.flags[:] = vis_flags
            viewer.sync()
            time.sleep(0.002)
        # MuJoCo 4.0.1 segfaults while tearing the passive viewer down (its render
        # thread races with glfw.terminate). Nothing needs saving, so leave directly.
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
