"""Six ball zones in, sport-mode requests out: the behavior node's `mode:=look`.

The FPGA zone network (models/ball_v3) names one of six zones, `upper_left ... lower_right`,
or `none` (/durance/ball_zone, from fpga_perception). The reflex modes (keeper, dodge) only
get left / center / right and lose the row. Look mode keeps both: the column turns the trunk
(Euler yaw), the row tilts it (Euler pitch).

The gestures and their safety come from the follow controller's look mode
(reflex/runtime/follow.py, the logic behind `ros-look`): a zone becomes its centre
(zone_target.py), the controller needs `consecutive` sightings, holds a gesture, and looks
straight again after `lost_s` without the ball. This module adds what the follow node adds
on top: the latest zone fed at a fixed rate (the zone is published only when it changes),
and the controller's commands as Unitree requests, with posture mode (Euler only acts after
Pose(true), sport_api.POSE) and hard limits on the angles.
"""
from __future__ import annotations

from dataclasses import dataclass

from reflex.runtime import sport_api as api
from reflex.runtime.follow import FollowConfig, FollowController
from reflex.runtime.zone_target import target_from_zone

HARD_MAX_YAW = 0.5     # rad, same limits as the follow node
HARD_MAX_PITCH = 0.35  # rad


@dataclass(frozen=True)
class Request:
    cmd: api.SportCommand
    param: str = ""
    reason: str = ""


class ZoneLook:
    """Feed it zone names and time; it returns the requests to send.

    `update(zone, t)` with the newest zone name (or None when nothing new arrived): call it at
    a steady rate, the controller counts sightings per call. `pitch_sign=-1` flips Euler pitch,
    whose sign on the real Go2 is not verified.
    """

    def __init__(self, cfg: FollowConfig, pitch_sign: float = 1.0, confidence: float = 1.0):
        cfg.mode = "look"
        cfg.look_yaw = min(abs(cfg.look_yaw), HARD_MAX_YAW)
        cfg.look_pitch = min(abs(cfg.look_pitch), HARD_MAX_PITCH)
        self.ctrl = FollowController(cfg)
        self.pitch_sign = -1.0 if pitch_sign < 0 else 1.0
        self.confidence = confidence          # the zone network sends no probability
        self.zone: str | None = None          # latest zone heard, None before the first
        self.posture = False                  # posture mode as last commanded

    def enable(self, t_s: float) -> list[Request]:
        return self._requests(self.ctrl.enable(t_s))

    def disable(self, t_s: float, reason: str = "disabled") -> list[Request]:
        return self._requests(self.ctrl.disable(t_s, reason))

    def set_zone(self, name: str) -> None:
        """Keep the latest zone. Unknown names raise ValueError and are not kept."""
        target_from_zone(name)
        self.zone = name.strip().lower()

    def tick(self, t_s: float, t_us: int = 0, source_alive: bool = True) -> list[Request]:
        """Feed the latest zone (`none` if its publisher is gone) and advance time."""
        if self.zone is None:
            return self._requests(self.ctrl.update(None, t_s))
        zone = self.zone if source_alive else "none"
        return self._requests(self.ctrl.update(target_from_zone(zone, t_us, self.confidence), t_s))

    @property
    def gesture(self) -> str:
        return self.ctrl.gesture

    def _requests(self, commands) -> list[Request]:
        out: list[Request] = []
        for c in commands:
            if c.kind == "euler":
                yaw = max(-HARD_MAX_YAW, min(HARD_MAX_YAW, c.yaw))
                pitch = self.pitch_sign * max(-HARD_MAX_PITCH, min(HARD_MAX_PITCH, c.pitch)) + 0.0  # no -0.0
                straight = yaw == 0.0 and pitch == 0.0
                if not straight:
                    out += self._posture(True, "Euler only acts in posture mode")
                out.append(Request(api.EULER, api.euler_params(0.0, pitch, yaw), c.reason))
                if straight:
                    out += self._posture(False, "trunk straight again")
            elif c.kind == "balance_stand":
                out.append(Request(api.BALANCE_STAND, "", c.reason))
            elif c.kind == "stop":
                out.append(Request(api.STOP_MOVE, "", c.reason))
            else:                                 # look mode never walks
                raise ValueError(f"look mode produced a {c.kind!r} command")
        return out

    def _posture(self, on: bool, reason: str) -> list[Request]:
        if self.posture == on:
            return []
        self.posture = on
        return [Request(api.POSE, api.pose_params(on), reason)]
