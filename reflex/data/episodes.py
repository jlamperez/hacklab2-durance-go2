"""Simulated episodes: threats and hard negatives, with a label for every instant.

Kinds:
  approach  ball rolled at the dog, arrives left / center / right      -> that side
  miss      ball rolled past the dog, well to one side                 -> none
  crossing  ball crosses the view sideways without approaching         -> none
  slow      ball creeps towards the dog too slowly to be a threat      -> none
  walker    a person-sized shape walks across                          -> none
  idle      nothing happens; only sensor noise and camera shake        -> none

The hard negatives exist so the network cannot learn "many events = threat".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from reflex.events.emu import EmuConfig, EventEmulator
from reflex.sim.scene import LABELS, BallShot, Go2Sim, to_gray

KINDS = ("approach", "miss", "crossing", "slow", "walker", "idle")
KIND_PROBS = (0.45, 0.10, 0.13, 0.10, 0.12, 0.10)

# A rolled ball counts as a threat while it is this far in front of the dog (m).
THREAT_NEAR, THREAT_FAR = 0.10, 2.20
DOG_FRONT_X = 0.35


@dataclass
class Episode:
    kind: str
    label: str                       # label while the threat is active; "none" for negatives
    events: np.ndarray               # EVENT_DTYPE
    t_us: np.ndarray                 # time of each physics step
    threat: np.ndarray               # bool per step: is the labelled threat active?
    meta: dict = field(default_factory=dict)

    def label_at(self, t_us: int) -> str:
        i = min(int(np.searchsorted(self.t_us, t_us)), len(self.t_us) - 1)
        return self.label if self.threat[i] else "none"


def _rand_rgb(rng):
    return tuple(float(c) for c in rng.uniform(0.15, 1.0, size=3))


def run_episode(sim: Go2Sim, rng: np.random.Generator, kind: str | None = None,
                label: str | None = None, randomize: bool = True, on_step=None) -> Episode:
    """Simulate one episode and return its events and per-step labels.

    `on_step(sim, events, threat_active)` is called after every physics step; the
    closed loop uses it to react while the episode runs.
    """
    kind = kind or str(rng.choice(KINDS, p=KIND_PROBS))
    sim.reset()
    meta: dict = {}
    if randomize:
        meta.update(sim.randomize_camera(rng))
        shake = float(rng.choice([0.0, rng.uniform(0.2, 1.5)], p=[0.25, 0.75]))
        sim.set_shake(rng, shake)
        meta["shake"] = shake
        cfg = EmuConfig.randomized(rng)
    else:
        sim.reset_camera()
        cfg = EmuConfig()
    h, w = sim.eye_hw
    emu = EventEmulator(h, w, cfg, rng)
    emu.reset(to_gray(sim.render_eye()), 0)

    pre_s = float(rng.uniform(0.08, 0.30))        # quiet time before anything moves
    ep_label, max_s, shot, walker = "none", 1.2, None, None
    if kind in ("approach", "miss"):
        shot = BallShot.sample(rng, label=(label or str(rng.choice(LABELS[1:]))) if kind == "approach" else "none")
        ep_label, max_s = shot.label, pre_s + 3.0
        meta.update(speed=shot.speed, radius=shot.radius)
    elif kind == "slow":
        shot = BallShot.sample(rng)
        shot.speed = float(rng.uniform(0.15, 0.50))
        shot.x_start = float(rng.uniform(0.9, 2.5))
        max_s = pre_s + float(rng.uniform(0.6, 1.2))
        meta.update(speed=shot.speed, radius=shot.radius)
    elif kind == "crossing":
        side = rng.choice([-1, 1])
        cross = dict(start=(float(rng.uniform(1.0, 4.0)), side * 2.6),
                     vel=(float(rng.uniform(-0.3, 0.3)), -side * float(rng.uniform(2.0, 6.0))),
                     radius=float(rng.uniform(0.05, 0.12)), rgb=_rand_rgb(rng))
        max_s = pre_s + 5.2 / abs(cross["vel"][1])
    elif kind == "walker":
        side = rng.choice([-1, 1])
        walker = dict(x=float(rng.uniform(1.5, 5.0)), y0=side * 3.2, vy=-side * float(rng.uniform(0.8, 1.8)),
                      vx=float(rng.uniform(-0.25, 0.25)), rgb=_rand_rgb(rng), phase=float(rng.uniform(0, 6.28)))
        max_s = pre_s + float(rng.uniform(1.0, 1.8))
    else:   # idle
        max_s = float(rng.uniform(0.5, 1.0))

    n_max = int(max_s / sim.dt)
    t_us = np.zeros(n_max, dtype=np.int64)
    threat = np.zeros(n_max, dtype=bool)
    chunks, launched, n = [], False, 0
    for i in range(n_max):
        if not launched and sim.time >= pre_s:
            launched = True
            if shot is not None:
                sim.launch(shot)
            elif kind == "crossing":
                sim.launch_raw(cross["start"], cross["vel"], cross["radius"], cross["rgb"])
        if walker is not None and launched:
            tw = sim.time - pre_s
            sim.set_walker(walker["x"] + walker["vx"] * tw, walker["y0"] + walker["vy"] * tw,
                           bob=0.02 * np.sin(2 * np.pi * 1.8 * tw + walker["phase"]), rgb=walker["rgb"])
        sim.step()
        ev = emu.step(to_gray(sim.render_eye()), int(sim.time * 1e6))
        chunks.append(ev)
        t_us[i] = int(sim.time * 1e6)
        ahead = sim.ball_xy()[0] - DOG_FRONT_X
        threat[i] = bool(kind == "approach" and launched and THREAT_NEAR < ahead < THREAT_FAR)
        n = i + 1
        if on_step is not None:
            on_step(sim, ev, bool(threat[i]))
        # Ball episodes end shortly after the ball reaches (or passes) the dog.
        if shot is not None and kind != "slow" and launched and ahead < -0.3:
            break
        if kind == "approach" and launched and ahead < THREAT_NEAR and i > 5 and not threat[i]:
            if threat[:i].any() and (t_us[i] - t_us[np.nonzero(threat[:i])[0][-1]]) > 150_000:
                break
    events = np.concatenate(chunks)
    return Episode(kind, ep_label, events, t_us[:n], threat[:n], meta)
