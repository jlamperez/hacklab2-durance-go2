"""Follow a target (a ball, a drone) with move, stop, look.

    disabled --enable--> looking --(N confident sightings)--> moving --(move_s)--> settling
                            ^                                                         |
                            +--------------------------(settle_s)---------------------+
    any state --(travel budget used up)--> halted (stays there until enabled again)

Why stop-and-look: an event camera on a moving dog sees the whole room move, and the
target is lost in it. Standing still, only the target fires. So the dog only acts on
what it saw while standing, moves for a short burst, stops, and waits for the camera
to calm down before looking again. Slower than continuous tracking, but it works with
a detector (or an SNN) trained on a still camera.

Two modes:
  walk   turn towards the target and, when it is roughly ahead and not close yet,
         walk towards it (sport-mode Move, then StopMove)
  gaze   stand on the spot and turn the trunk towards it (sport-mode Euler, yaw only)

Safety is built in, because the robot has none of its own in sport mode:
  - starts disabled; `enable()` is an explicit step
  - every motion is a short burst (move_s) ended by StopMove, never an open-ended Move
  - speeds are capped (max_vx, max_wz), well below what the robot can do
  - the dog only walks forward, and only towards a target roughly ahead of it
  - it stops walking when the target is close (low in the image)
  - a travel budget (max_travel_m of commanded forward motion) halts it for good
  - nothing confident seen: it does nothing (it is already standing)
None of this replaces a person holding the Unitree remote.

Pure logic, no ROS: the follow node and the headless simulation both drive this class.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from reflex.runtime.target import Target

MODES = ("walk", "gaze")


@dataclass
class FollowConfig:
    mode: str = "walk"
    min_confidence: float = 0.5
    consecutive: int = 2          # confident sightings in a row before acting
    hfov_deg: float = 90.0        # camera's horizontal field of view: turns u into an angle
    move_s: float = 0.8           # length of one burst of motion
    settle_s: float = 0.8         # after a burst: sightings ignored (actuation delay + own motion)
    resend_s: float = 0.1         # Move is repeated this often during a burst
    turn_gain: float = 0.6        # fraction of the angle to the target turned per burst
    max_wz: float = 0.5           # rad/s
    max_vx: float = 0.3           # m/s, forward only
    walk_cone_deg: float = 20.0   # walk forward only if the target is within this angle
    v_near: float = 0.6           # target at or below this image height (v) is close: do not walk
    v_slow: float = 0.3           # forward speed ramps from max_vx at v_slow down to 0 at v_near
    max_travel_m: float = 3.0     # commanded forward travel before halting for good
    max_yaw: float = 0.35         # gaze: trunk yaw limit (rad). Real Go2 limit NOT verified
    lost_s: float = 2.0           # gaze: nothing seen for this long -> trunk back to straight


@dataclass(frozen=True)
class Command:
    kind: str                 # move / stop / euler / balance_stand
    vx: float = 0.0           # move: m/s, + forward
    wz: float = 0.0           # move: rad/s, + turns left
    yaw: float = 0.0          # euler: rad, + turns left
    reason: str = ""


class FollowController:
    def __init__(self, cfg: FollowConfig | None = None):
        self.cfg = cfg or FollowConfig()
        if self.cfg.mode not in MODES:
            raise ValueError(f"unknown mode {self.cfg.mode!r}, expected one of {MODES}")
        self.state = "disabled"
        self.travel_m = 0.0
        self.yaw = 0.0                     # gaze: current trunk yaw set point
        self.burst: Command | None = None  # the motion being executed
        self._streak = 0
        self._t_state = 0.0
        self._t_sent = -1e9
        self._t_seen = 0.0
        self.last_target: Target | None = None

    # --- switches --------------------------------------------------------------------------
    def enable(self, t_s: float) -> list[Command]:
        """Start following. Also clears a halt: the travel budget starts again."""
        self.travel_m, self.yaw, self._streak = 0.0, 0.0, 0
        self._set("looking", t_s)
        self._t_seen = t_s
        return [Command("balance_stand", reason="enabled")]

    def disable(self, t_s: float, reason: str = "disabled") -> list[Command]:
        """Stop now. Safe to call in any state, any number of times."""
        self._set("disabled", t_s)
        return self._stop(reason)

    @property
    def enabled(self) -> bool:
        return self.state not in ("disabled", "halted")

    # --- main entry ------------------------------------------------------------------------
    def update(self, target: Target | None, t_s: float) -> list[Command]:
        """Feed a new estimate, or None just to advance time. Returns the commands to send."""
        c = self.cfg
        if target is not None:
            self.last_target = target
        if self.state in ("disabled", "halted"):
            return []
        if self.state == "moving":
            if t_s - self._t_state >= c.move_s:
                self._set("settling", t_s)
                return self._stop("burst done") if c.mode == "walk" else []
            if c.mode == "walk" and t_s - self._t_sent >= c.resend_s:
                self._t_sent = t_s
                return [self.burst]
            return []
        if self.state == "settling":
            if t_s - self._t_state >= c.settle_s:
                self._set("looking", t_s)
                self._t_seen = t_s
            return []

        # looking: the dog stands still, sightings count
        if target is None:
            return self._maybe_recentre(t_s)
        if not (target.found and target.confidence >= c.min_confidence):
            self._streak = 0
            return self._maybe_recentre(t_s)
        self._streak += 1
        self._t_seen = t_s
        if self._streak < c.consecutive:
            return []
        self._streak = 0
        return self._act(target, t_s)

    # --- internals -------------------------------------------------------------------------
    def angle_to(self, u: float) -> float:
        """Horizontal angle to the target, rad, + is to the dog's left (u < 0)."""
        return -math.atan(u * math.tan(math.radians(self.cfg.hfov_deg) / 2))

    def _act(self, target: Target, t_s: float) -> list[Command]:
        c = self.cfg
        angle = self.angle_to(target.u)
        if c.mode == "gaze":
            self.yaw = _clip(self.yaw + c.turn_gain * angle, c.max_yaw)
            self.burst = Command("euler", yaw=self.yaw, reason=f"target at {math.degrees(angle):+.0f} deg")
            self._set("moving", t_s)
            return [self.burst]
        wz = _clip(c.turn_gain * angle / c.move_s, c.max_wz)
        vx = 0.0
        if abs(angle) <= math.radians(c.walk_cone_deg) and target.v < c.v_near:
            ramp = (c.v_near - target.v) / max(c.v_near - c.v_slow, 1e-6)
            vx = c.max_vx * min(1.0, ramp)
            if self.travel_m + vx * c.move_s > c.max_travel_m:
                self._set("halted", t_s)
                return self._stop(f"travel budget of {c.max_travel_m:.1f} m used up: enable again to continue")
            self.travel_m += vx * c.move_s
        if vx == 0.0 and abs(wz) < 0.05:
            return []                     # close and straight ahead: nothing to do
        why = "close, turn only" if target.v >= c.v_near else f"v {target.v:+.2f}"
        self.burst = Command("move", vx=vx, wz=wz, reason=f"target at {math.degrees(angle):+.0f} deg, {why}")
        self._set("moving", t_s)
        self._t_sent = t_s
        return [self.burst]

    def _maybe_recentre(self, t_s: float) -> list[Command]:
        if self.cfg.mode == "gaze" and self.yaw != 0.0 and t_s - self._t_seen >= self.cfg.lost_s:
            self.yaw = 0.0
            self._set("settling", t_s)
            return [Command("euler", yaw=0.0, reason="target lost: looking straight again")]
        return []

    def _stop(self, reason: str) -> list[Command]:
        self.burst = None
        if self.cfg.mode == "gaze":
            self.yaw = 0.0
            return [Command("euler", yaw=0.0, reason=reason)]
        return [Command("stop", reason=reason)]

    def _set(self, state: str, t_s: float) -> None:
        self.state, self._t_state = state, t_s


def _clip(x: float, m: float) -> float:
    return max(-m, min(m, x))
