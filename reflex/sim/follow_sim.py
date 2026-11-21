"""Follow a ball in simulation, without ROS: the closed loop of the follow controller.

    python -m reflex.sim.follow_sim                      # 40 s, prints a summary
    python -m reflex.sim.follow_sim --video out/follow.mp4
    python -m reflex.sim.follow_sim --mode gaze          # trunk only, yaw
    python -m reflex.sim.follow_sim --mode look --video out/look.mp4   # nine gestures, feet planted

What runs: walking Go2 (learned policy) -> head camera -> event emulator -> 50 ms window ->
CentroidTracker -> FollowController -> Move / StopMove, applied after an actuation delay.
The ball is moved by script along `TargetPath`: it rolls slowly, stops, changes direction.

The SNN is not in this loop: it is trained on real recordings and would be tested on
them. This checks the controller, its timing and its safety limits against a camera
that moves with the dog. In gaze and look modes the dog stands (no walking policy) and
Euler turns its trunk with the feet planted (`Go2Sim.set_trunk`, our stand-in for sport
mode's posture mode); the ball goes left, right, near and far in front of it instead.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from reflex.data.windows import WindowConfig, make_window
from reflex.events.emu import EmuConfig, EventEmulator
from reflex.runtime.follow import FollowConfig, FollowController
from reflex.runtime.target import CentroidTracker
from reflex.sim.scene import Go2Sim, to_gray

BALL_RADIUS = 0.11
BALL_RGB = (0.95, 0.45, 0.1)


@dataclass
class TargetPath:
    """Ball moved by script: straight lines between waypoints, with a pause at each.

    World frame, the dog starts at the origin looking along +x. The default path stays
    in front of the dog and goes left, right and away, so that following it needs turns.
    """
    waypoints: list = field(default_factory=lambda: [(1.4, 0.0), (1.6, 0.8), (2.4, 0.6), (2.6, -0.6),
                                                     (3.4, -0.4), (3.6, 0.6), (4.4, 0.5)])
    speed: float = 0.35         # m/s while rolling
    pause_s: float = 1.5        # still at each waypoint: an event camera does not see it then

    def at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Position and velocity (world, m and m/s) at time t."""
        p = np.asarray(self.waypoints, dtype=float)
        for a, b in zip(p[:-1], p[1:]):
            if t < self.pause_s:
                return a.copy(), np.zeros(2)
            t -= self.pause_s
            length = float(np.linalg.norm(b - a))
            dur = length / self.speed
            if t < dur:
                d = (b - a) / length
                return a + d * self.speed * t, d * self.speed
            t -= dur
        return p[-1].copy(), np.zeros(2)


# For the modes that stand still: in front of the dog, left / right / near / far.
LOOK_PATH = [(1.2, 0.0), (1.4, 0.9), (2.6, 0.9), (2.6, -0.9), (1.2, -0.7), (1.0, 0.0), (3.0, 0.0)]


def place_ball(sim: Go2Sim, xy, vel) -> None:
    sim.launch_raw(xy, vel, BALL_RADIUS, BALL_RGB)


def run(duration_s: float = 40.0, mode: str = "walk", delay_ms: float = 300.0, seed: int = 0,
        video: str | None = None, cfg: FollowConfig | None = None, path: TargetPath | None = None,
        camera_hz: float = 100.0) -> dict:
    sim = Go2Sim(seed=seed)
    standing = (cfg.mode if cfg else mode) in ("gaze", "look")
    if not standing:
        sim.enable_walking()
    rng = np.random.default_rng(seed)
    h, w = sim.eye_hw
    emu = EventEmulator(h, w, EmuConfig(), rng)
    wcfg = WindowConfig()
    path = path or (TargetPath(waypoints=LOOK_PATH) if standing else TargetPath())
    tracker = CentroidTracker()
    hfov = math.degrees(2 * math.atan(math.tan(math.radians(60) / 2) * w / h))
    ctrl = FollowController(cfg or FollowConfig(mode=mode, hfov_deg=hfov, vfov_deg=60.0))
    stride = max(1, round(0.02 / sim.dt))          # one window every 20 ms, as in the bridge
    # Rendering is the slow part (software rendering: ~0.1 s a frame). The bridge renders
    # every physics step; here every 1/camera_hz s, which the emulator handles.
    render_every = max(1, min(stride, round(1.0 / (camera_hz * sim.dt))))
    recent = deque(maxlen=math.ceil(wcfg.window_ms * 1e-3 / (render_every * sim.dt)) + 1)
    pending: list = []                              # (apply_t, command)
    writer = None
    if video:
        import imageio.v2 as imageio
        writer = imageio.get_writer(video, fps=25, codec="libx264", quality=7, macro_block_size=2)

    place_ball(sim, *path.at(0.0))
    emu.reset(to_gray(sim.render_eye()), 0)
    calib, calibrated = [], False
    log = {"in_view": 0, "frames": 0, "dist": [], "commands": {}, "halted_at": None, "gestures": [],
           "err": [], "err0": []}
    eye_id, base_id = sim.model.camera("eye").id, sim.model.body("base").id
    t_enable = 1.5                                  # first second: calibrate on a still scene
    steps = int(duration_s / sim.dt)
    for i in range(steps):
        t = sim.time
        if t >= t_enable:
            place_ball(sim, *path.at(t - t_enable))
        while pending and pending[0][0] <= t:
            _, cmd = pending.pop(0)
            if cmd.kind == "move":
                sim.set_velocity(cmd.vx, 0.0, cmd.wz)
            elif cmd.kind == "stop":
                sim.set_velocity(0.0, 0.0, 0.0)
            elif cmd.kind == "euler":
                sim.set_trunk(0.0, cmd.pitch, cmd.yaw)
            # balance_stand: the dog already stands
        sim.step()
        if i % render_every == 0:
            recent.append(emu.step(to_gray(sim.render_eye()), int(sim.time * 1e6)))
        if i % stride:
            continue
        t_us = int(sim.time * 1e6)
        window = make_window(np.concatenate(recent), t_us, wcfg)
        if not calibrated:
            if sim.time > 0.4:
                calib.append(window)
            if sim.time >= t_enable:
                tracker.calibrate(np.stack(calib))
                calibrated = True
                for cmd in ctrl.enable(sim.time):
                    pending.append((sim.time + delay_ms / 1e3, cmd))
            continue
        target = tracker.predict(window, t_us)
        for cmd in ctrl.update(target, sim.time):
            log["commands"][cmd.kind] = log["commands"].get(cmd.kind, 0) + 1
            if ctrl.cfg.mode == "look":
                log["gestures"].append((round(sim.time, 1), ctrl.gesture))
            pending.append((sim.time + delay_ms / 1e3, cmd))
        if ctrl.state == "halted" and log["halted_at"] is None:
            log["halted_at"] = round(sim.time, 1)

        # How well it follows: ball within +-30 deg of the heading and 0.3..2.5 m away.
        x, y, hd = sim.base_pose()
        bx, by = sim.ball_xy()
        dist = math.hypot(bx - x, by - y)
        bearing = (math.atan2(by - y, bx - x) - hd + math.pi) % (2 * math.pi) - math.pi
        log["frames"] += 1
        log["in_view"] += abs(bearing) < math.radians(30) and 0.3 < dist < 2.5
        log["dist"].append(dist)
        if ctrl.cfg.mode == "look" and path.at(sim.time - t_enable)[1].any():
            # While the ball rolls (paused, an event camera cannot see it): angle between where
            # the trunk points and the ball, and the same for a dog that keeps looking ahead.
            eye = sim.data.cam_xpos[eye_id]
            to_ball = np.array([bx, by, BALL_RADIUS]) - eye
            to_ball /= np.linalg.norm(to_ball)
            fwd = sim.data.xmat[base_id].reshape(3, 3)[:, 0]
            ahead = np.array([1.0, 0.0, 0.0])
            log["err"].append(math.degrees(math.acos(np.clip(fwd @ to_ball, -1, 1))))
            log["err0"].append(math.degrees(math.acos(np.clip(ahead @ to_ball, -1, 1))))
        if writer is not None and i % (2 * stride) == 0:
            writer.append_data(_frame(sim, recent, ctrl, target, h, w))
    if writer is not None:
        writer.close()
    x, y, _ = sim.base_pose()
    return {
        "mode": ctrl.cfg.mode, "duration_s": duration_s,
        "ball_in_view": round(log["in_view"] / max(log["frames"], 1), 3),
        "median_distance_m": round(float(np.median(log["dist"])), 2),
        "dog_moved_m": round(math.hypot(x, y), 2),
        "commanded_travel_m": round(ctrl.travel_m, 2),
        "upright": round(sim.upright(), 2),
        "commands": log["commands"], "halted_at_s": log["halted_at"],
        "min_events_threshold": round(tracker.cfg.min_events, 1),
        **({"median_deg_trunk_to_ball": round(float(np.median(log["err"])), 1) if log["err"] else None,
            "median_deg_if_looking_ahead": round(float(np.median(log["err0"])), 1) if log["err0"] else None,
            "gestures": log["gestures"]} if ctrl.cfg.mode == "look" else {}),
    }


def _frame(sim, recent, ctrl, target, h, w):
    import cv2

    from reflex.events import repr as ev_repr
    from reflex.sim.demo import FONT, _panel
    ev = ev_repr.render(ev_repr.to_frame(np.concatenate(list(recent)[-10:]), h, w), saturation=1.5)[..., ::-1]
    ev = np.ascontiguousarray(ev)
    if target is not None and target.found:
        cx, cy = int((target.u + 1) / 2 * w), int((target.v + 1) / 2 * h)
        cv2.circle(ev, (cx, cy), 10, (0, 255, 0), 2)
    frame = np.hstack([_panel(sim.render_third(), "third person"), _panel(ev, "events + target")])
    bar = np.zeros((30, frame.shape[1], 3), dtype=np.uint8)
    b = ctrl.burst
    txt = (f"t {sim.time:5.1f} s  state {ctrl.state:9s}  "
           + (f"vx {b.vx:.2f} wz {b.wz:+.2f}" if b is not None and b.kind == "move" else "")
           + (f"  look {ctrl.gesture}" if ctrl.cfg.mode == "look" else f"  travel {ctrl.travel_m:.1f} m"))
    cv2.putText(bar, txt, (8, 21), FONT, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([frame, bar])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--mode", choices=("walk", "gaze", "look"), default="walk")
    ap.add_argument("--delay-ms", type=float, default=300.0, help="actuation delay of the robot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", default=None, help="write an mp4 here")
    ap.add_argument("--camera-hz", type=float, default=100.0, help="frames rendered per second (speed vs fidelity)")
    a = ap.parse_args()
    print(json.dumps(run(a.duration, a.mode, a.delay_ms, a.seed, a.video, camera_hz=a.camera_hz), indent=2))


if __name__ == "__main__":
    main()
