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
HELLO = SportCommand("Hello", 1016, "ROBOT_SPORT_API_ID_HELLO")

ALL = (DAMP, BALANCE_STAND, STOP_MOVE, STAND_UP, STAND_DOWN, RECOVERY_STAND, EULER, MOVE, HELLO)
BY_ID = {c.api_id: c for c in ALL}

# Note: the client at this commit has no BodyHeight command. To lower the trunk on the
# real robot use StandDown, or Euler for a lean.


def move_params(vx: float, vy: float, vyaw: float) -> str:
    """SportClient::Move: js["x"]=vx, js["y"]=vy, js["z"]=vyaw (m/s, m/s, rad/s)."""
    return json.dumps({"x": float(vx), "y": float(vy), "z": float(vyaw)})


def euler_params(roll: float, pitch: float, yaw: float) -> str:
    """SportClient::Euler: js["x"]=roll, js["y"]=pitch, js["z"]=yaw (rad)."""
    return json.dumps({"x": float(roll), "y": float(pitch), "z": float(yaw)})
