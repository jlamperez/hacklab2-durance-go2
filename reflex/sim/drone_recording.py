"""Simulated event-camera recordings: the Go2 stands in pose mode, a tiny-whoop drone flies
one of the predefined behaviors in front of it.

    python -m reflex.sim.drone_recording --behavior orbit                    # one flight, with video
    python -m reflex.sim.drone_recording --behavior orbit --head follow --live
    python -m reflex.sim.drone_recording --behavior all --head still follow --seeds 5 --no-video
    python -m reflex.sim.drone_recording --list

Behaviors: reflex/sim/drone_behaviors.py (hover, cross, approach, orbit, figure8, zigzag,
rise_drop, dart, leave_return, waypoints; none = no drone, for negatives). Drone: sized like an Eachine UZ65 (approximate).

Head modes (the robot never walks; the camera moves like a trunk posture change, see
Go2Sim.set_head):
  still   robot motionless
  follow  the "head" turns towards the drone with a delay and a speed limit, giving the camera
          motion the final follow-the-drone behavior will cause. Scripted from the simulator's
          ground truth, not a behavior.
  sweep   the head pans and tilts on its own, independent of the drone (camera motion that
          must not be mistaken for a target)

Writes data/sim_drone/<behavior>_<head>_<seed>.h5 (events + labels, same format as the real
recordings) and .bin (events only, river-tools format), and out/sim_drone/....mp4 unless
--no-video. Labels per camera image in the .h5: /labels/t (us), x, y (pixel of the drone
center, -1 if not in front), visible (inside the image), dist (m), head_yaw, head_pitch (rad),
radius_px (drone half-size in the image), support (events of the last 10 ms within
radius_px + 4 px of the center). For training, use labels with visible and enough support: a
hovering drone in front of a still camera is in view but makes almost no events.
Simulation only: rendered images turned into events by the emulator, noise calibrated on the
real GenX320.
"""
from __future__ import annotations

import argparse
import math
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from reflex.events import repr as ev_repr
from reflex.events.emu import EmuConfig, EventEmulator, empty_events
from reflex.sim.drone_behaviors import BEHAVIORS, Flight, make
from reflex.sim.follow_demo import GENX320_NOISE, panel
from reflex.sim.scene import Go2Sim, to_gray

REPO = Path(__file__).resolve().parents[2]
FONT = cv2.FONT_HERSHEY_SIMPLEX
HEADS = ("still", "follow", "sweep")
# The real event camera: on a small tripod 10 cm above the Go2's head and 10 cm behind its
# front, level, 76 deg field of view. In the model the head's front is at x = 0.33 m and its top
# at z = 0.09 m (base frame), so the lens sits at (0.23, 0, 0.19), about 0.5 m above the floor.
# Level (pitch 0) and the lens center being "10 cm above" are our reading, not measured.
CAMERA_MOUNT = dict(offset=[0.23, 0.0, 0.19], pitch_deg=0.0, fovy=76.0)
# Trunk posture limits for the virtual head: a guess, the Go2's real Euler range is not known.
YAW_LIMIT, PITCH_LIMIT = math.radians(35), math.radians(20)
DRONE_HALF_SIZE = 0.043      # m: half of the UZ65's ~85 mm across
SUPPORT_MS = 10.0            # window for the label's event support


class Head:
    """Virtual head motion: none, following the drone, or sweeping on its own."""

    def __init__(self, mode: str, rng: np.random.Generator, dt: float):
        self.mode, self.dt = mode, dt
        self.yaw = self.pitch = 0.0
        self.delay = deque(maxlen=max(1, round(rng.uniform(0.1, 0.25) / dt)))   # perception + actuation delay
        self.max_rate = (rng.uniform(1.0, 2.0), rng.uniform(0.6, 1.2))          # rad/s yaw, pitch
        self.gain = rng.uniform(3.0, 6.0)
        self.sweep = (rng.uniform(0.6, 1.0) * YAW_LIMIT, rng.uniform(0.2, 0.5), rng.uniform(0, 6.3),
                      rng.uniform(0.3, 0.8) * PITCH_LIMIT, rng.uniform(0.15, 0.4), rng.uniform(0, 6.3))

    def update(self, t: float, drone_pos, cam_pos) -> tuple[float, float]:
        if self.mode == "still":
            return 0.0, 0.0
        if self.mode == "sweep":
            ay, fy, py, ap, fp, pp = self.sweep
            return ay * math.sin(2 * math.pi * fy * t + py), ap * math.sin(2 * math.pi * fp * t + pp)
        # follow: aim the camera axis at where the drone was a moment ago
        rel = np.asarray(drone_pos) - np.asarray(cam_pos)
        want_yaw = math.atan2(rel[1], rel[0])
        elev = math.atan2(rel[2], math.hypot(rel[0], rel[1]))
        want_pitch = math.radians(CAMERA_MOUNT["pitch_deg"]) - elev   # + = nose down; 0 when the drone is on the neutral axis
        self.delay.append((want_yaw, want_pitch))
        ty, tp = self.delay[0]
        ty = max(-YAW_LIMIT, min(YAW_LIMIT, ty))
        tp = max(-PITCH_LIMIT, min(PITCH_LIMIT, tp))
        for attr, target, rate in (("yaw", ty, self.max_rate[0]), ("pitch", tp, self.max_rate[1])):
            cur = getattr(self, attr)
            step = max(-rate * self.dt, min(rate * self.dt, self.gain * (target - cur) * self.dt))
            setattr(self, attr, cur + step)
        return self.yaw, self.pitch


def record(behavior: str, head_mode: str, seed: int, args, write_h5, write_bin) -> dict:
    rng = np.random.default_rng(seed)
    sim = Go2Sim(seed=seed, eye_hw=(320, 320), third_hw=(320, 480), fast_eye=True, drone=True,
                 scene=args.scene)
    sim.set_camera_mount(**CAMERA_MOUNT)
    sim.reset()                                   # standing pose, held by the PD controller
    path = make(behavior, rng)
    duration = args.duration or path.duration + 0.5
    flight = Flight(path, rng, sim.dt, calm=args.calm) if behavior != "none" else None
    head = Head(head_mode, rng, sim.dt)
    emu = EventEmulator(320, 320, EmuConfig(**GENX320_NOISE), rng)
    render_every = max(1, round(1 / (args.render_hz * sim.dt)))
    frame_every = max(1, round(args.frame_ms * 1e-3 / sim.dt))
    stem = f"{behavior}_{head_mode}_{seed}"

    writer = None
    if not args.no_video:
        import imageio.v2 as imageio
        writer = imageio.get_writer(Path(args.out) / f"{stem}.mp4", fps=round(1000 / args.frame_ms),
                                    codec="libx264", quality=7, macro_block_size=2)
    chunks, recent, labels = [], deque(), []
    cam_id = sim.model.camera("eye").id
    emu.reset(to_gray(sim.render_eye()), 0)
    t_wall = time.perf_counter()
    lab = (-1.0, -1.0, False, 0.0)
    try:
        for i in range(1, int(duration / sim.dt) + 1):
            if flight is None:
                pos, yaw = path.at(sim.time)
                sim.set_drone(pos, yaw)
            else:
                pos, quat = flight.step()
                sim.set_drone(pos, quat=quat)
            sim.set_head(*head.update(sim.time, pos, sim.data.cam_xpos[cam_id]))
            sim.step()
            if i % render_every == 0:
                ev = emu.step(to_gray(sim.render_eye()), int(sim.time * 1e6))
                chunks.append(ev)
                recent.append(ev)
                while len(recent) > max(1, round(0.02 / (render_every * sim.dt))):
                    recent.popleft()
                lab = sim.project_to_eye(pos)
                focal = 160 / math.tan(math.radians(sim.model.cam_fovy[cam_id]) / 2)
                radius = focal * DRONE_HALF_SIZE / max(lab[3], 1e-3)
                support, t_now = 0, int(sim.time * 1e6)
                if lab[2]:
                    for e in reversed(chunks):
                        if not len(e) or e["t"][-1] < t_now - SUPPORT_MS * 1000:
                            break
                        e = e[e["t"] >= t_now - SUPPORT_MS * 1000]
                        support += int(np.count_nonzero(np.hypot(e["x"] - lab[0], e["y"] - lab[1]) <= radius + 4))
                labels.append((t_now, *lab, head.yaw, head.pitch, radius, support))
            if (writer is not None or args.live_fn is not None) and i % frame_every == 0:
                win = np.concatenate(recent) if recent else empty_events()
                ev_img = ev_repr.render(ev_repr.to_frame(win, 320, 320), saturation=1.0)[..., ::-1].copy()
                if lab[2]:                             # ground truth: where the drone is
                    cv2.circle(ev_img, (int(lab[0]), int(lab[1])), 12, (80, 255, 80), 1)
                frame = np.hstack([panel(sim.render_third(), "third person", (320, 480)),
                                   panel(sim.render_eye(), "dog camera (RGB, tripod)"),
                                   panel(ev_img, "events 20 ms, green = drone (truth)")])
                bar = np.zeros((28, frame.shape[1], 3), np.uint8)
                cv2.putText(bar, f"sim  {behavior} / head {head_mode}   t = {sim.time:5.2f} s   "
                                 f"drone {lab[3]:.1f} m {'in view' if lab[2] else 'not in view'}   "
                                 f"head yaw {math.degrees(head.yaw):+4.0f} pitch {math.degrees(head.pitch):+4.0f} deg   "
                                 f"{len(win) / 20:5.1f} k ev/s", (8, 19), FONT, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
                frame = np.vstack([frame, bar])
                if writer is not None:
                    writer.append_data(frame)
                if args.live_fn is not None:
                    args.live_fn(frame)
    except KeyboardInterrupt:
        if not getattr(args, "keep_partial", True):      # batch runs: no half flights
            raise
        print(f"  stopped at {sim.time:.1f} s; saving what was recorded")
    finally:
        if writer is not None:
            writer.close()

    ev = np.concatenate(chunks) if chunks else empty_events()
    lab_arr = np.array(labels, dtype=float).reshape(-1, 9)
    meta = {"label": "none" if behavior == "none" else "drone", "behavior": behavior, "head": head_mode, "seed": seed, "simulated": True,
            "instruction": f"simulated: robot standing (pose mode), drone '{behavior}', head {head_mode}",
            "session": "sim_drone", "camera": f"tripod 10 cm above the head (sim), 76 deg, head {head_mode}", "ball": "none",
            "drone_model": "Eachine UZ65-like tiny whoop, approximate size",
            "t_start_us": int(ev["t"][0]) if len(ev) else 0, "t_end_us": int(ev["t"][-1]) if len(ev) else 0}
    h5 = write_h5(Path(args.data) / f"{stem}.h5", np.rec.fromarrays([ev["x"], ev["y"], ev["p"], ev["t"]],
                                                                   names="x,y,polarity,timestamp"), meta)
    import h5py
    with h5py.File(h5, "a") as f:
        g = f.create_group("labels")
        g.attrs["description"] = ("ground truth per camera image: drone center in pixels (-1 if behind the "
                                  "camera), visible = inside the 320x320 image, dist in m, head yaw/pitch in rad, "
                                  "radius_px = drone half-size in the image, support = events of the last "
                                  f"{SUPPORT_MS:g} ms within radius_px + 4 px")
        for k, (name, typ) in enumerate((("t", np.int64), ("x", np.float32), ("y", np.float32), ("visible", np.uint8),
                                          ("dist", np.float32), ("head_yaw", np.float32), ("head_pitch", np.float32),
                                          ("radius_px", np.float32), ("support", np.int32))):
            g.create_dataset(name, data=lab_arr[:, k].astype(typ), compression="gzip")
    write_bin(Path(args.data) / f"{stem}.bin", ev["t"], ev["x"], ev["y"], ev["p"])
    vis = float(lab_arr[:, 3].mean()) if len(lab_arr) else 0.0
    span = (ev["t"][-1] - ev["t"][0]) / 1e6 if len(ev) else 0.0
    print(f"{stem:28s} {span:5.1f} s  {len(ev):>9,} events ({len(ev) / max(span, 1e-9):>7,.0f}/s)  "
          f"drone visible {vis:4.0%}  ({time.perf_counter() - t_wall:.0f} s wall)", flush=True)
    return {"stem": stem, "events": len(ev), "visible": vis}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--behavior", nargs="+", default=["waypoints"], help="names, or 'all'")
    ap.add_argument("--head", nargs="+", default=["still"], choices=HEADS)
    ap.add_argument("--seeds", type=int, default=1, help="flights per behavior and head mode")
    ap.add_argument("--seed0", type=int, default=0, help="first seed")
    ap.add_argument("--duration", type=float, default=0.0, help="seconds per flight (0 = the behavior's own length)")
    ap.add_argument("--render-hz", type=float, default=500.0, help="camera images per simulated second (event timing)")
    ap.add_argument("--frame-ms", type=float, default=40.0, help="video frame every N ms of sim time (40 = real time at 25 fps)")
    ap.add_argument("--scene", default="hall", choices=("hall", "lab", "park"),
                    help="hall (default): room with a dark concrete floor, a rug in half the rooms, drawn per seed")
    ap.add_argument("--calm", action="store_true", help="drone flies its path much more smoothly (less jitter, drift, jerks)")
    ap.add_argument("--live", action="store_true", help="show it in a window while it runs (slower than real time)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--list", action="store_true", help="list the behaviors and exit")
    ap.add_argument("--out", default=str(REPO / "out" / "sim_drone"))
    ap.add_argument("--data", default=str(REPO / "data" / "sim_drone"))
    args = ap.parse_args()

    if args.list:
        for name, f in BEHAVIORS.items():
            print(f"{name:13s} {f.__doc__.strip()}")
        return
    names = list(BEHAVIORS) if args.behavior == ["all"] else args.behavior
    unknown = [n for n in names if n not in BEHAVIORS]
    if unknown:
        raise SystemExit(f"unknown behavior(s) {unknown}; --list shows them")
    from reflex.events.h5 import write_bin, write_h5

    Path(args.out).mkdir(parents=True, exist_ok=True)
    Path(args.data).mkdir(parents=True, exist_ok=True)
    args.live_fn = None
    if args.live:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(13, 3.6))
        fig.canvas.manager.set_window_title("Simulated drone, robot in pose mode")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        shown = ax.imshow(np.zeros((348, 1120, 3), np.uint8))
        plt.show(block=False)

        def live(frame):
            if not plt.fignum_exists(fig.number):
                raise KeyboardInterrupt
            shown.set_data(frame)
            fig.canvas.draw_idle()
            plt.pause(0.001)
        args.live_fn = live

    for name in names:
        for head in args.head:
            for k in range(args.seeds):
                record(name, head, args.seed0 + k, args, write_h5, write_bin)
    print(f"events in {args.data}" + ("" if args.no_video else f", videos in {args.out}"))


if __name__ == "__main__":
    main()
