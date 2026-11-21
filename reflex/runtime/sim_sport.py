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

    Sign convention, dog frame (+y is left): Move with vy > 0 leans the simulated dog to
    its left. Euler roll follows the right-hand rule about +x, so roll > 0 lowers the
    right side and leans the dog to its right, as the real Go2 did on 21 Nov 2026
    (Euler x=+0.4). Move vy > 0 stepped the real Go2 to its left the same day.
    """
    cmd = sport_api.BY_ID.get(int(api_id))
    if cmd is None:
        return None
    if cmd in (sport_api.MOVE, sport_api.EULER):
        try:
            p = json.loads(parameter) if parameter else {}
        except json.JSONDecodeError:
            return None
        # Positive value = lean left: Move's vy as is, Euler's roll negated (see docstring).
        value = float(p.get("y", 0.0)) if cmd is sport_api.MOVE else -float(p.get("x", 0.0))
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


class PostureMode:
    """Sport mode's posture mode, as seen on the real Go2 on 21 Nov 2026.

    Pose(true) enters it, Pose(false) leaves it. Inside it Euler leans the trunk and Move
    does not walk (the real robot turned its trunk instead). Outside it Move walks and Euler
    is accepted (code 0) but has no effect. The robot starts outside it. One unexplained
    observation: at the start of the session the real robot did lean on Euler without a
    Pose(true) first; the code always sends Pose(true), so that case does not matter to it.
    """

    def __init__(self) -> None:
        self.on = False

    def handle(self, api_id: int, parameter: str = "") -> bool:
        """Update the mode for a request. True if the request should take effect."""
        if int(api_id) == sport_api.POSE.api_id:
            try:
                self.on = bool(json.loads(parameter).get("data", False)) if parameter else False
            except (json.JSONDecodeError, AttributeError):
                pass
            return False          # changes the mode only; nothing to show
        if int(api_id) == sport_api.EULER.api_id:
            return self.on
        if int(api_id) == sport_api.MOVE.api_id:
            return not self.on
        return True
