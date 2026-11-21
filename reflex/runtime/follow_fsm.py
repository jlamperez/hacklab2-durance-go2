"""Follow-the-ball behavior: turn the body towards a moving ball, greet when it comes to rest.

    idle --ball seen twice--> track --off center--> turning --done--> settle --> track
    track --ball gone, last seen inside the image--> greeting --> cooldown --> idle
    track --ball gone at an image edge--> search (turn towards that edge) --> settle --> track
    track / search --nothing seen for lost_s--> idle

Event-camera facts this relies on:
  - a ball that comes to rest stops producing events: "gone, but not at an edge" = stopped
  - while the robot turns, the whole scene produces events: observations are ignored
    while turning and for `settle_s` after

Same logic in simulation and on the robot. Actions are abstract; the caller maps them:
  Turn(rate)   real robot: sport-mode Move(0, 0, rate);  simulation: walking policy
  StopTurn     real robot: StopMove (or Move(0, 0, 0))
  Greet        real robot: sport-mode Hello;  simulation: the imitated hello trick
Every state has a timeout, so the robot always returns to idle (standing still).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from reflex.runtime.ball_tracker import BallObservation


@dataclass
class FollowConfig:
    hfov_deg: float = 70.0      # horizontal field of view of the camera
    deadband_deg: float = 8.0   # closer to the center than this: no turn
    turn_rate: float = 0.8      # rad/s while turning
    max_turn_deg: float = 40.0  # longest single turn
    settle_s: float = 0.35      # ignore the camera this long after a turn
    stop_s: float = 0.5         # ball gone this long (robot still) -> it stopped
    min_track_s: float = 0.25   # ball must have been tracked this long before a greeting counts
    edge_frac: float = 0.1      # last seen this close to the image border -> it left the image
    lost_s: float = 3.0         # nothing seen this long -> give up, back to idle
    greet_s: float = 6.0        # how long the greeting takes (camera ignored meanwhile)
    cooldown_s: float = 1.0
    confirm: int = 2            # consecutive detections needed to start tracking
    same_spot: float = 0.08     # while tracking, a detection counts only if the previous update
                                # saw the ball within this fraction of the image width


@dataclass(frozen=True)
class Turn:
    rate: float                 # rad/s, + = left


@dataclass(frozen=True)
class StopTurn:
    pass


@dataclass(frozen=True)
class Greet:
    pass


class FollowFSM:
    def __init__(self, cfg: FollowConfig | None = None):
        self.cfg = cfg or FollowConfig()
        self.state = "idle"
        self._until = 0.0           # end of the current turn / settle / greeting / cooldown
        self._after_settle = "track"
        self._seen = 0              # consecutive detections in idle
        self._last_seen = 0.0
        self._track_start = 0.0
        self._last_x = 0.5
        self._prev_x: float | None = None  # previous update's detection (None: nothing seen)
        self.events: list[tuple[float, str]] = []   # (time, what happened), for logs and evaluation

    @property
    def moving(self) -> bool:
        return self.state == "turning"

    def bearing_deg(self, x_frac: float) -> float:
        """Angle of an image column from the optical axis, + = left of center."""
        half = math.radians(self.cfg.hfov_deg) / 2
        return -math.degrees(math.atan((x_frac - 0.5) * 2 * math.tan(half)))

    def _log(self, t: float, what: str) -> None:
        self.events.append((t, what))

    def _turn(self, t: float, angle_deg: float, then: str) -> Turn:
        c = self.cfg
        angle = max(-c.max_turn_deg, min(c.max_turn_deg, angle_deg))
        self.state, self._until, self._after_settle = "turning", t + math.radians(abs(angle)) / c.turn_rate, then
        self._log(t, f"turn {angle:+.0f} deg")
        return Turn(math.copysign(c.turn_rate, angle))

    def update(self, obs: BallObservation, t: float) -> list:
        """Feed the latest observation at time t (seconds). Returns the actions to execute."""
        c, out = self.cfg, []
        s = self.state

        if s == "turning":
            if t >= self._until:
                self.state, self._until = "settle", t + c.settle_s
                out.append(StopTurn())
            return out
        if s == "settle":
            if t >= self._until:
                self.state = self._after_settle
                self._last_seen = t         # the ball may have moved out of sight while turning
                self._prev_x = None
            return out
        if s == "greeting":
            if t >= self._until:
                self.state, self._until = "cooldown", t + c.cooldown_s
            return out
        if s == "cooldown":
            if t >= self._until:
                self.state, self._seen = "idle", 0
            return out

        if s == "idle":
            self._seen = self._seen + 1 if obs.found else 0
            if self._seen >= c.confirm:
                self.state, self._track_start, self._last_seen = "track", t, t
                self._last_x = obs.x_frac
                self._log(t, "ball found")
            return out

        # track / search: a detection must repeat at the same spot (a single flash of a
        # background edge, e.g. from the robot's own small movements, is ignored)
        prev, self._prev_x = self._prev_x, (obs.x_frac if obs.found else None)
        confirmed = obs.found and prev is not None and abs(obs.x_frac - prev) <= c.same_spot
        if obs.found and not confirmed:
            return out
        if confirmed:
            if s == "search":
                self.state, self._track_start = "track", t
                self._log(t, "ball found again")
            self._last_seen, self._last_x = t, obs.x_frac
            bearing = self.bearing_deg(obs.x_frac)
            if abs(bearing) > c.deadband_deg:
                # after an aiming turn the ball is expected near the center
                self._last_x = 0.5
                out.append(self._turn(t, bearing, then="track"))
            return out

        gone = t - self._last_seen
        if s == "track" and gone >= c.stop_s:
            at_edge = self._last_x < c.edge_frac or self._last_x > 1 - c.edge_frac
            if at_edge:
                self._log(t, "ball left the image")
                self.state = "search"
                self._last_seen = t
                out.append(self._turn(t, c.max_turn_deg * (1 if self._last_x < 0.5 else -1), then="search"))
            elif t - self._track_start >= c.min_track_s:
                self.state, self._until = "greeting", t + c.greet_s
                self._log(t, "ball stopped: greet")
                out.append(Greet())
            else:
                self.state = "idle"
            return out
        if gone >= c.lost_s:
            self._log(t, "lost")
            self.state, self._seen = "idle", 0
        return out
