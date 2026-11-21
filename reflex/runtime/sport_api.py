"""Unitree Go2 sport-mode commands used by this project.

Single source of truth for `api_id` values. Every entry was read from Unitree's own
example client; none is written from memory. To add a command, look it up in the
source below and add it here with the same care.

Source: https://github.com/unitreerobotics/unitree_ros2 at commit 668d1ec
  ids:        example/src/include/common/ros2_sport_client.h
  parameters: example/src/src/common/ros2_sport_client.cpp

Transport: publish `unitree_api/msg/Request` on `/api/sport/request` with
`header.identity.api_id` set and `parameter` holding a JSON string (empty if none).

Not verified on a real robot yet.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

SPORT_REQUEST_TOPIC = "/api/sport/request"
SOURCE_COMMIT = "668d1ec"


@dataclass(frozen=True)
class SportCommand:
    name: str
    api_id: int
    source: str   # constant name in ros2_sport_client.h


DAMP = SportCommand("Damp", 1001, "ROBOT_SPORT_API_ID_DAMP")                      # motors limp: emergency stop
BALANCE_STAND = SportCommand("BalanceStand", 1002, "ROBOT_SPORT_API_ID_BALANCESTAND")
STOP_MOVE = SportCommand("StopMove", 1003, "ROBOT_SPORT_API_ID_STOPMOVE")
STAND_UP = SportCommand("StandUp", 1004, "ROBOT_SPORT_API_ID_STANDUP")
STAND_DOWN = SportCommand("StandDown", 1005, "ROBOT_SPORT_API_ID_STANDDOWN")
RECOVERY_STAND = SportCommand("RecoveryStand", 1006, "ROBOT_SPORT_API_ID_RECOVERYSTAND")
EULER = SportCommand("Euler", 1007, "ROBOT_SPORT_API_ID_EULER")
MOVE = SportCommand("Move", 1008, "ROBOT_SPORT_API_ID_MOVE")
# Posture mode on/off. On the real Go2 (21 Nov 2026) Euler was accepted (code 0) but did
# nothing until Pose(true) was sent; in posture mode Move turns the trunk instead of walking.
POSE = SportCommand("Pose", 1028, "ROBOT_SPORT_API_ID_POSE")
SIT = SportCommand("Sit", 1009, "ROBOT_SPORT_API_ID_SIT")
RISE_SIT = SportCommand("RiseSit", 1010, "ROBOT_SPORT_API_ID_RISESIT")
HELLO = SportCommand("Hello", 1016, "ROBOT_SPORT_API_ID_HELLO")
STRETCH = SportCommand("Stretch", 1017, "ROBOT_SPORT_API_ID_STRETCH")
CONTENT = SportCommand("Content", 1020, "ROBOT_SPORT_API_ID_CONTENT")
DANCE1 = SportCommand("Dance1", 1022, "ROBOT_SPORT_API_ID_DANCE1")
DANCE2 = SportCommand("Dance2", 1023, "ROBOT_SPORT_API_ID_DANCE2")
SCRAPE = SportCommand("Scrape", 1029, "ROBOT_SPORT_API_ID_SCRAPE")
HEART = SportCommand("Heart", 1036, "ROBOT_SPORT_API_ID_HEART")
# Acrobatics: the robot leaves the ground. Clear space, and only if the organizers allow it.
FRONT_FLIP = SportCommand("FrontFlip", 1030, "ROBOT_SPORT_API_ID_FRONTFLIP")
FRONT_JUMP = SportCommand("FrontJump", 1031, "ROBOT_SPORT_API_ID_FRONTJUMP")
FRONT_POUNCE = SportCommand("FrontPounce", 1032, "ROBOT_SPORT_API_ID_FRONTPOUNCE")
LEFT_FLIP = SportCommand("LeftFlip", 2041, "ROBOT_SPORT_API_ID_LEFTFLIP")
BACK_FLIP = SportCommand("BackFlip", 2043, "ROBOT_SPORT_API_ID_BACKFLIP")

# Tricks: none takes a parameter in Unitree's client. HandStand (2044) does (a flag) and
# is left out. Which of these a given Go2 model and firmware accepts is not known.
GESTURES = (SIT, RISE_SIT, HELLO, STRETCH, CONTENT, DANCE1, DANCE2, SCRAPE, HEART)
ACROBATICS = (FRONT_FLIP, FRONT_JUMP, FRONT_POUNCE, LEFT_FLIP, BACK_FLIP)

ALL = (DAMP, BALANCE_STAND, STOP_MOVE, STAND_UP, STAND_DOWN, RECOVERY_STAND, EULER, MOVE, POSE) + GESTURES + ACROBATICS
BY_ID = {c.api_id: c for c in ALL}

# Note: the client at this commit has no BodyHeight command. To lower the trunk on the
# real robot use StandDown, or Euler for a lean.


def move_params(vx: float, vy: float, vyaw: float) -> str:
    """SportClient::Move: js["x"]=vx, js["y"]=vy, js["z"]=vyaw (m/s, m/s, rad/s)."""
    return json.dumps({"x": float(vx), "y": float(vy), "z": float(vyaw)})


def euler_params(roll: float, pitch: float, yaw: float) -> str:
    """SportClient::Euler: js["x"]=roll, js["y"]=pitch, js["z"]=yaw (rad).

    Roll follows the right-hand rule about +x (forward): roll > 0 lowers the right side.
    Checked on the real Go2 on 21 Nov 2026. Only acts in posture mode (see POSE).
    """
    return json.dumps({"x": float(roll), "y": float(pitch), "z": float(yaw)})


def pose_params(flag: bool) -> str:
    """SportClient::Pose: js["data"]=flag (true enters posture mode, false leaves it)."""
    return json.dumps({"data": bool(flag)})


def requests_for_pose(pose: str, last_pose: str, reaction: str, roll: float, vy: float) -> list:
    """Sport-mode requests that take the robot from `last_pose` to `pose`.

    pose: lean_left / lean_right / crouch / stand. reaction: "euler" leans the trunk,
    "move" steps sideways. Signs follow the dog frame (+y is left); leaning left needs a
    negative roll. Euler only acts in posture mode, and Move only walks outside it, so a
    lean enters posture mode and the way back leaves it before BalanceStand.
    """
    side = {"lean_left": 1.0, "lean_right": -1.0}.get(pose)
    if side is not None:
        if reaction == "move":
            return [(MOVE, move_params(0.0, side * vy, 0.0))]
        return [(POSE, pose_params(True)), (EULER, euler_params(-side * roll, 0.0, 0.0))]
    if pose == "crouch":
        return [(STAND_DOWN, "")]
    # Back to standing: undo whatever the last reaction was.
    if last_pose == "crouch":
        return [(STAND_UP, ""), (BALANCE_STAND, "")]
    if reaction == "move":
        return [(STOP_MOVE, "")]
    return [(EULER, euler_params(0.0, 0.0, 0.0)), (POSE, pose_params(False)), (BALANCE_STAND, "")]
