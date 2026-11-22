"""Follow-the-ball in simulation, without ROS: events -> detector -> behavior -> simulated Go2.

    python -m reflex.sim.follow_demo                          # all scenarios, videos + table
    python -m reflex.sim.follow_demo --scenarios stop-left drone --seeds 3
    python -m reflex.sim.follow_demo --scenarios stop-right --live    # watch it (slowed down)

Scenarios (the dog starts facing +x, camera on a mast like the Durance robot):
  stop-left / stop-center / stop-right  ball rolls in from the side and comes to rest about
                                        2 m away in that third of the image -> expect: turn
                                        towards it, then greet
  cross        ball rolls across and out of the room          -> expect: no greeting
  drone        a drone flies in, hovers, flies out             -> expect: no greeting
  walker-stop  a person walks in and stands still              -> expect: no greeting

Detector: the no-learning CentroidTracker (reflex/runtime/ball_tracker.py), a baseline; the
SNN replaces it later. Behavior: FollowFSM (reflex/runtime/follow_fsm.py). Turning uses the
walking policy (the real robot: sport-mode Move); the greeting is the simulator's imitation
of Hello.

Writes, per episode, a video (third person, dog's camera, events with the detection) and the
events as HDF5 in the format of the real recordings (capture/replay.py plays them), plus a
results table. Simulation only: the emulator, the camera mount and the turning dynamics are
approximations of the real robot.
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np

from reflex.events import repr as ev_repr
from reflex.events.emu import EmuConfig, EventEmulator, empty_events
from reflex.runtime.ball_tracker import CentroidTracker, TrackerConfig
from reflex.runtime.follow_fsm import FollowConfig, FollowFSM, Greet, StopTurn, Turn
from reflex.sim.scene import Go2Sim, to_gray

REPO = Path(__file__).resolve().parents[2]
SCENARIOS = ("stop-left", "stop-center", "stop-right", "cross", "drone", "walker-stop")
EXPECT_GREET = {"stop-left", "stop-center", "stop-right"}

# Camera like the Durance robot (photo in Durance_Vision_Lab_Track.pdf): on a mast above the
# front of the back, looking forward and slightly down. Lens unknown: 70 deg is a guess.
MAST = dict(offset=[0.22, 0.0, 0.16], pitch_deg=-12.0, fovy=70.0)
# Emulator noise calibrated on the real GenX320 (still scene: about 20-40 events/s in total,
# takes 15/16 of sat_follow2), not the emulator's default of ~0.5 Hz per pixel.
GENX320_NOISE = dict(noise_rate_hz=3e-4, hot_pixel_frac=0.0)

ACTION_T0 = 0.8           # s: the dog settles first, then the scenario starts
FONT = cv2.FONT_HERSHEY_SIMPLEX


# --------------------------------------------------------------------------- scripted objects

class Script:
    """Moves the scenario's objects; knows the ground truth for the evaluation."""
    stop_t: float | None = None             # when the ball comes to rest (absolute sim time)

    def apply(self, sim: Go2Sim, t: float) -> None: ...

    def target_xy(self, t: float) -> np.ndarray | None:
        """Where the thing the dog should look at is, or None."""
        return None


class BallScript(Script):
    """Ball rolls in a straight line from `start` and decelerates to rest at `stop`.
    With `stop=None` it keeps its speed and is removed after `leave_after` s."""

    def __init__(self, start, end, v0: float, radius: float, rgb, stops: bool, leave_after: float = 1e9):
        self.start, self.end = np.asarray(start, float), np.asarray(end, float)
        self.dir = (self.end - self.start) / np.linalg.norm(self.end - self.start)
        self.length, self.v0, self.radius, self.rgb, self.stops = np.linalg.norm(self.end - self.start), v0, radius, rgb, stops
        self.acc = v0 ** 2 / (2 * self.length) if stops else 0.0
        self.stop_t = ACTION_T0 + v0 / self.acc if stops else None
        self.leave_t = ACTION_T0 + leave_after
        self.quat = np.array([1.0, 0, 0, 0])
        self._last = None

    def pos(self, t: float) -> np.ndarray | None:
        if t < ACTION_T0 or t > self.leave_t:
            return None
        u = t - ACTION_T0
        s = self.v0 * u - 0.5 * self.acc * u * u if self.stops else self.v0 * u
        if self.stops and u >= self.v0 / self.acc:
            s = self.length
        return self.start + self.dir * s

    def apply(self, sim, t):
        p = self.pos(t)
        if p is None:
            sim.park_ball()
            self._last = None
            return
        if self._last is not None:                   # roll: rotate about z x direction of travel
            d = p - self._last
            dist = float(np.linalg.norm(d))
            if dist > 1e-9:
                dq = np.zeros(4)
                mujoco.mju_axisAngle2Quat(dq, np.array([-d[1], d[0], 0.0]) / dist, dist / self.radius)
                q = np.zeros(4)
                mujoco.mju_mulQuat(q, dq, self.quat)
                self.quat = q / np.linalg.norm(q)
        self._last = p
        sim.set_ball_kinematic(p, self.quat, self.radius, self.rgb)

    def target_xy(self, t):
        return self.pos(t)


class DroneScript(Script):
    def __init__(self, x: float, z: float, y_in: float, y_hover: float, y_out: float):
        self.x, self.z, self.y_in, self.y_hover, self.y_out = x, z, y_in, y_hover, y_out

    def apply(self, sim, t):
        u = t - ACTION_T0
        if u < 0 or u > 5.0:
            sim.park_drone()
            return
        if u < 1.5:
            y = self.y_in + (self.y_hover - self.y_in) * u / 1.5
        elif u < 3.5:
            y = self.y_hover + 0.05 * math.sin(2 * math.pi * 0.7 * u)
        else:
            y = self.y_hover + (self.y_out - self.y_hover) * (u - 3.5) / 1.5
        sim.set_drone([self.x, y, self.z + 0.03 * math.sin(2 * math.pi * 1.3 * u)])


class WalkerScript(Script):
    def __init__(self, x: float, y_in: float, y_stop: float, rgb):
        self.x, self.y_in, self.y_stop, self.rgb = x, y_in, y_stop, rgb

    def apply(self, sim, t):
        u = t - ACTION_T0
        if u < 0:
            sim.park_walker()
            return
        walk_s = abs(self.y_stop - self.y_in) / 1.2           # 1.2 m/s
        a = min(1.0, u / walk_s)
        bob = 0.03 * abs(math.sin(math.pi * u / 0.5)) if a < 1 else 0.0
        sim.set_walker(self.x, self.y_in + (self.y_stop - self.y_in) * a, bob, self.rgb)


def make_script(name: str, rng: np.random.Generator) -> Script:
    color = tuple(float(c) for c in rng.uniform(0.3, 1.0, 3))
    if name.startswith("stop-"):
        zone = name.split("-")[1]
        ang = {"left": rng.uniform(18, 28), "center": rng.uniform(-6, 6), "right": -rng.uniform(18, 28)}[zone]
        dist = rng.uniform(1.6, 2.3)
        stop = dist * np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))])
        side = {"left": 1, "right": -1, "center": rng.choice([-1, 1])}[zone]
        start = stop + [rng.uniform(0.2, 0.8), side * rng.uniform(1.4, 1.9)]
        return BallScript(start, stop, rng.uniform(1.4, 2.0), rng.uniform(0.07, 0.11), color, stops=True)
    if name == "cross":
        x = rng.uniform(1.8, 2.4)
        return BallScript([x, 2.2], [x, -3.0], rng.uniform(1.2, 1.8), rng.uniform(0.07, 0.11), color,
                          stops=False, leave_after=4.0)
    if name == "drone":
        return DroneScript(rng.uniform(1.8, 2.6), rng.uniform(0.6, 0.9), 2.2, rng.uniform(-0.3, 0.3), -2.6)
    if name == "walker-stop":
        return WalkerScript(rng.uniform(2.4, 3.0), 2.8, rng.uniform(-0.4, 0.4), color)
    raise ValueError(f"unknown scenario {name!r}; choose from {SCENARIOS}")


# --------------------------------------------------------------------------- one episode

@dataclass
class Result:
    scenario: str
    seed: int
    greeted: bool
    greet_t: float | None
    stop_t: float | None
    heading_err_deg: float | None
    turns: int
    success: bool
    events: int
    log: str


def heading_error_deg(sim: Go2Sim, target_xy) -> float:
    x, y, h = sim.base_pose()
    want = math.atan2(target_xy[1] - y, target_xy[0] - x)
    return math.degrees((want - h + math.pi) % (2 * math.pi) - math.pi)


def panel(img, title, hw=(320, 320)):
    out = np.ascontiguousarray(cv2.resize(img, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST))
    cv2.rectangle(out, (0, 0), (hw[1], 22), (0, 0, 0), -1)
    cv2.putText(out, title, (6, 15), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def run_episode(name: str, seed: int, args, writer=None, live=None) -> tuple[Result, np.ndarray]:
    rng = np.random.default_rng(seed)
    sim = Go2Sim(seed=seed, eye_hw=(320, 320), third_hw=(320, 480), fast_eye=True, drone=True)
    sim.set_camera_mount(**MAST)
    sim.enable_walking()
    sim.reset()
    # standing still in the pose controller; a Turn switches to the walking policy
    script = make_script(name, rng)
    emu = EventEmulator(320, 320, EmuConfig(**GENX320_NOISE), rng)
    tracker = CentroidTracker(TrackerConfig())
    fsm = FollowFSM(FollowConfig(hfov_deg=MAST["fovy"]))
    render_every = max(1, round(1 / (args.render_hz * sim.dt)))
    decide_every = max(1, round(args.decide_ms * 1e-3 / sim.dt))
    frame_every = max(1, round(args.frame_ms * 1e-3 / sim.dt))
    recent: deque[np.ndarray] = deque()
    all_events = []
    pending: list[tuple[float, object]] = []   # actions waiting for the actuation delay
    obs = None
    greet_t, turns = None, 0
    trace = []          # per decision: time, state, detection, true bearing of the target
    n_steps = int(args.duration / sim.dt)
    emu.reset(to_gray(sim.render_eye()), 0)
    for i in range(1, n_steps + 1):
        t = sim.time
        script.apply(sim, t)
        sim.step()
        if i % render_every == 0:
            ev = emu.step(to_gray(sim.render_eye()), int(sim.time * 1e6))
            recent.append(ev)
            all_events.append(ev)
            t_cut = int((sim.time - args.window_ms * 1e-3) * 1e6)
            while recent and (len(recent[0]) == 0 or recent[0]["t"][-1] < t_cut):
                recent.popleft()
        if i % decide_every == 0:
            win = np.concatenate(recent) if recent else empty_events()
            win = win[win["t"] >= int((sim.time - args.window_ms * 1e-3) * 1e6)]
            obs = tracker.observe(win["x"], win["y"])
            tgt = script.target_xy(sim.time)
            true_b = heading_error_deg(sim, tgt) if tgt is not None else None
            trace.append((round(sim.time, 3), fsm.state, obs.found, round(fsm.bearing_deg(obs.x_frac), 1) if obs.found else None,
                          None if true_b is None else round(true_b, 1), obs.n_blob, obs.n_total))
            for a in fsm.update(obs, sim.time):
                pending.append((sim.time + args.delay_ms * 1e-3, a))
        while pending and pending[0][0] <= sim.time:
            _, a = pending.pop(0)
            if isinstance(a, Turn):
                sim.set_velocity(0.0, 0.0, a.rate)
                turns += 1
            elif isinstance(a, StopTurn):
                # Stand still (pose controller), like sport mode's balance stand: stepping in
                # place under the walking policy shakes the camera enough to fake a ball.
                sim.set_pose("stand")
            elif isinstance(a, Greet):
                sim.play("hello")
                greet_t = sim.time
        if (writer is not None or live is not None) and i % frame_every == 0:
            win = np.concatenate(recent) if recent else empty_events()
            ev_img = ev_repr.render(ev_repr.to_frame(win, 320, 320), saturation=1.0)[..., ::-1].copy()
            if obs is not None and obs.found:
                cv2.circle(ev_img, (int(obs.x_frac * 320), int(obs.y_frac * 320)), 14, (80, 255, 80), 2)
            for xf in (1 / 3, 2 / 3):
                cv2.line(ev_img, (int(xf * 320), 22), (int(xf * 320), 319), (70, 70, 70), 1)
            frame = np.hstack([panel(sim.render_third(), "third person", (320, 480)),
                               panel(sim.render_eye(), "dog camera (RGB, mast)"),
                               panel(ev_img, "events, last %d ms" % args.window_ms)])
            bar = np.zeros((28, frame.shape[1], 3), np.uint8)
            last = fsm.events[-1][1] if fsm.events else ""
            cv2.putText(bar, f"{name}  t = {sim.time:5.2f} s   state: {fsm.state:9s}  {last}",
                        (8, 19), FONT, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
            frame = np.vstack([frame, bar])
            if writer is not None:
                writer.append_data(frame)
            if live is not None:
                live(frame)

    events = np.concatenate(all_events) if all_events else empty_events()
    greeted = greet_t is not None
    err = None
    if greeted and script.target_xy(greet_t) is not None:
        err = heading_error_deg(sim, script.target_xy(greet_t))   # dog does not move while greeting
    if name in EXPECT_GREET:
        success = greeted and script.stop_t is not None and greet_t > script.stop_t and err is not None and abs(err) < 15
    else:
        success = not greeted
    log = "; ".join(f"{tt:.2f}s {w}" for tt, w in fsm.events)
    with (Path(args.out) / f"{name}_{seed}_trace.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "state", "found", "seen_bearing_deg", "true_bearing_deg", "n_blob", "n_total"])
        w.writerows(trace)
    return Result(name, seed, greeted, greet_t, script.stop_t, err, turns, success, len(events), log), events


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=SCENARIOS)
    ap.add_argument("--seeds", type=int, default=1, help="episodes per scenario")
    ap.add_argument("--duration", type=float, default=9.0, help="simulated seconds per episode")
    ap.add_argument("--render-hz", type=float, default=250.0, help="camera images per simulated second (event emulator rate)")
    ap.add_argument("--window-ms", type=int, default=30, help="events the detector looks at")
    ap.add_argument("--decide-ms", type=float, default=20.0, help="detector + behavior update period")
    ap.add_argument("--delay-ms", type=float, default=0.0, help="actuation delay (real sport mode: unmeasured)")
    ap.add_argument("--frame-ms", type=float, default=40.0, help="video frame every N ms of sim time (40 = real time at 25 fps)")
    ap.add_argument("--out", default=str(REPO / "out" / "follow"), help="videos and results")
    ap.add_argument("--data", default=str(REPO / "data" / "sim_follow"), help="HDF5 events per episode")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--live", action="store_true", help="show the episode in a window while it runs (slower than real time)")
    args = ap.parse_args()

    out, data = Path(args.out), Path(args.data)
    out.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    live = None
    if args.live:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(13, 3.6))
        fig.canvas.manager.set_window_title("Follow the ball (simulation)")
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        shown = ax.imshow(np.zeros((348, 1120, 3), np.uint8))
        plt.show(block=False)

        def live(frame):
            shown.set_data(frame)
            fig.canvas.draw_idle()
            plt.pause(0.001)

    import imageio.v2 as imageio
    from reflex.events.h5 import write_h5

    results = []
    for name in args.scenarios:
        for k in range(args.seeds):
            seed = 1000 * SCENARIOS.index(name) + k
            t0 = time.perf_counter()
            writer = None if args.no_video else imageio.get_writer(out / f"{name}_{seed}.mp4", fps=round(1000 / args.frame_ms),
                                                                    codec="libx264", quality=7, macro_block_size=2)
            try:
                res, events = run_episode(name, seed, args, writer, live)
            finally:
                if writer is not None:
                    writer.close()
            ev = np.zeros(len(events), dtype=[("x", ">u2"), ("y", ">u2"), ("polarity", "u1"), ("timestamp", ">i8")])
            ev["x"], ev["y"], ev["polarity"], ev["timestamp"] = events["x"], events["y"], events["p"], events["t"]
            write_h5(data / f"sim_{name}_{seed}.h5", ev, {
                "label": name, "instruction": f"simulated: {name}; behavior log: {res.log}",
                "session": "sim_follow", "camera": "mast (sim)", "t_start_us": int(ACTION_T0 * 1e6),
                "t_end_us": int((res.stop_t or args.duration) * 1e6),
                "marks_us": [int(res.greet_t * 1e6)] if res.greet_t else [], "simulated": True})
            results.append(res)
            err = f"{res.heading_err_deg:+.0f} deg" if res.heading_err_deg is not None else "-"
            late = f"{res.greet_t - res.stop_t:.2f} s" if res.greeted and res.stop_t else "-"
            print(f"{name:12s} seed {seed:5d}: {'OK  ' if res.success else 'FAIL'} greeted={res.greeted!s:5s} "
                  f"heading error {err:8s} greet after stop {late:7s} turns {res.turns}  "
                  f"({len(events):,} events, {time.perf_counter() - t0:.0f} s)  | {res.log}", flush=True)

    with (out / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "seed", "success", "greeted", "greet_t", "stop_t", "heading_err_deg", "turns", "events", "log"])
        for r in results:
            w.writerow([r.scenario, r.seed, r.success, r.greeted, r.greet_t, r.stop_t, r.heading_err_deg, r.turns, r.events, r.log])
    print("\nscenario       success")
    for name in args.scenarios:
        rs = [r for r in results if r.scenario == name]
        print(f"{name:14s} {sum(r.success for r in rs)}/{len(rs)}")
    print(f"\nvideos and results.csv in {out}; events in {data}")


if __name__ == "__main__":
    main()
