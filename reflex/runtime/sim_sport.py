"""What the simulated dog does with a sport-mode request.

The real Go2 runs Unitree's controller. The simulator only has four poses, so each
request is translated to the closest one. This is the single place where that
approximation lives; it is pure Python so it can be tested without ROS.
"""
from __future__ import annotations

import json

from reflex.runtime import sport_api

DEADBAND = 0.05


def pose_for_request(api_id: int, parameter: str = "") -> str | None:
    """Pose or motion for a request: stand / crouch / lean_left / lean_right / sit / hello /
    damp, or None if ignored.

    Sign convention, dog frame (+y is left): Move with vy > 0 and Euler with roll > 0
    both lean the simulated dog to its left. Whether a real Go2 rolls to the same side
    for a positive Euler roll is NOT verified.
    """
    cmd = sport_api.BY_ID.get(int(api_id))
    if cmd is None:
        return None
    if cmd in (sport_api.MOVE, sport_api.EULER):
        try:
            p = json.loads(parameter) if parameter else {}
        except json.JSONDecodeError:
            return None
        value = float(p.get("y" if cmd is sport_api.MOVE else "x", 0.0))
        if value > DEADBAND:
            return "lean_left"
        if value < -DEADBAND:
            return "lean_right"
        return "stand"
    if cmd is sport_api.STAND_DOWN:
        return "crouch"
    if cmd is sport_api.SIT:
        return "sit"
    if cmd is sport_api.HELLO:
        return "hello"       # our own imitation: sit, raise a front paw, wave, stand up
    if cmd is sport_api.RISE_SIT:
        return "stand"
    if cmd is sport_api.DAMP:
        return "damp"
    if cmd in (sport_api.STAND_UP, sport_api.BALANCE_STAND, sport_api.RECOVERY_STAND, sport_api.STOP_MOVE):
        return "stand"
    return None   # stretch, dances, jumps, flips: no equivalent in the simulator
