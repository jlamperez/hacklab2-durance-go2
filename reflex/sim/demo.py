"""Record a demo video: balls rolled at a simulated Go2, seen as RGB and as events.

    pixi run sim                      # writes out/demo.mp4, out/demo.png, out/demo_ep*.npz
    pixi run sim --episodes 5 --seed 3

The head camera is rendered at the physics rate (500 Hz of simulated time) and fed
to the event emulator. The video is slowed down so the events are visible.
"""
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from reflex.events import io as ev_io
from reflex.events import repr as ev_repr
from reflex.events.emu import EmuConfig, EventEmulator
from reflex.sim.scene import BallShot, Go2Sim, to_gray

PANEL_HW = (270, 480)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _panel(img_rgb: np.ndarray, title: str) -> np.ndarray:
    out = cv2.resize(img_rgb, (PANEL_HW[1], PANEL_HW[0]), interpolation=cv2.INTER_NEAREST)
    out = np.ascontiguousarray(out)
    cv2.rectangle(out, (0, 0), (PANEL_HW[1], 24), (0, 0, 0), -1)
    cv2.putText(out, title, (8, 17), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def run_episode(sim: Go2Sim, shot: BallShot, emu: EventEmulator, steps_per_frame: int,
                pre_s: float = 0.15, max_s: float = 2.5, display_ms: float = 10.0):
    """Yield (video_frame_rgb, n_events_shown) and return all events at the end.

    The event panel shows a sliding window of the last `display_ms` of events.
    """
    sim.reset()
    h, w = sim.eye_hw
    emu.reset(to_gray(sim.render_eye()), 0)
    all_events, launched, arrived_t = [], False, None
    recent = deque(maxlen=max(1, round(display_ms * 1e-3 / sim.dt)))
    n_steps = int(max_s / sim.dt)
    for i in range(1, n_steps + 1):
        if not launched and sim.time >= pre_s:
            sim.launch(shot)
            launched = True
        sim.step()
        eye = sim.render_eye()
        events = emu.step(to_gray(eye), int(sim.time * 1e6))
        all_events.append(events)
        recent.append(events)
        if i % steps_per_frame == 0:
            window = np.concatenate(recent)
            ev_img = ev_repr.render(ev_repr.to_frame(window, h, w), saturation=1.5)[..., ::-1]  # BGR -> RGB
            rate = len(window) / (len(recent) * sim.dt)
            frame = np.hstack([
                _panel(sim.render_third(), "third person"),
                _panel(eye, "dog eye (RGB)"),
                _panel(ev_img, "event camera (emulated)"),
            ])
            bar = np.zeros((30, frame.shape[1], 3), dtype=np.uint8)
            txt = (f"t = {sim.time * 1e3:6.0f} ms   shot: {shot.label.upper()}  "
                   f"{shot.speed:.1f} m/s  r = {shot.radius * 100:.0f} cm   "
                   f"events: {rate / 1e3:7.1f} k/s")
            cv2.putText(bar, txt, (8, 21), FONT, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
            yield np.vstack([frame, bar]), len(window)
        # Stop shortly after the ball reaches the dog (it may bounce off or roll past).
        if launched and arrived_t is None and sim.ball_xy()[0] < 0.45:
            arrived_t = sim.time
        if arrived_t is not None and sim.time - arrived_t > 0.25:
            break
    return np.concatenate(all_events)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out/demo.mp4")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scene", default="lab", choices=("lab", "park"), help="park: outdoor scenery")
    ap.add_argument("--fps", type=int, default=50, help="video frame rate")
    ap.add_argument("--steps-per-frame", type=int, default=2,
                    help="physics steps (2 ms each) per video frame; 2 at 50 fps = 0.2x speed")
    ap.add_argument("--fixed-sensor", action="store_true", help="do not randomize the emulator")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    sim = Go2Sim(seed=args.seed, scene=args.scene)
    h, w = sim.eye_hw
    sheet = []
    labels = ["left", "center", "right"]
    with imageio.get_writer(out, fps=args.fps, codec="libx264", quality=8, macro_block_size=2) as writer:
        for ep in range(args.episodes):
            shot = BallShot.sample(rng, label=labels[ep % 3])
            cfg = EmuConfig() if args.fixed_sensor else EmuConfig.randomized(rng)
            emu = EventEmulator(h, w, cfg, rng)
            gen = run_episode(sim, shot, emu, args.steps_per_frame)
            frames, activity = [], []
            try:
                while True:
                    frame, n_shown = next(gen)
                    writer.append_data(frame)
                    frames.append(frame)
                    activity.append(n_shown)
            except StopIteration as stop:
                events = stop.value
            ev_io.save_npz(out.with_name(f"{out.stem}_ep{ep}.npz"), events, h, w,
                           label=shot.label, speed=shot.speed, radius=shot.radius)
            sheet.append(frames[int(np.argmax(activity))])   # busiest frame, for the contact sheet
            print(f"episode {ep}: {shot.label:6s} {shot.speed:.1f} m/s  "
                  f"{len(events):7d} events  {len(frames)} video frames  "
                  f"(c_on={cfg.c_on:.2f} c_off={cfg.c_off:.2f} noise={cfg.noise_rate_hz:.2f} Hz)")
    imageio.imwrite(out.with_suffix(".png"), np.vstack(sheet))
    print(f"wrote {out} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
