"""Checks the api_id table against Unitree's header when the checkout is present."""
import json
import re

import pytest

from reflex.runtime import sport_api
from reflex.sim.scene import REPO_ROOT

HEADER = REPO_ROOT / "third_party/unitree_ros2/example/src/include/common/ros2_sport_client.h"


def test_ids_are_unique():
    assert len(sport_api.BY_ID) == len(sport_api.ALL)


def test_param_encoding_matches_the_client():
    assert json.loads(sport_api.move_params(0.1, -0.3, 0.5)) == {"x": 0.1, "y": -0.3, "z": 0.5}
    assert json.loads(sport_api.euler_params(0.2, 0.0, 0.0)) == {"x": 0.2, "y": 0.0, "z": 0.0}
    assert json.loads(sport_api.pose_params(True)) == {"data": True}


def _names(requests):
    return [cmd.name for cmd, _ in requests]


def test_euler_lean_enters_posture_mode_with_the_checked_sign():
    left = sport_api.requests_for_pose("lean_left", "stand", "euler", roll=0.3, vy=0.3)
    assert _names(left) == ["Pose", "Euler"]
    assert json.loads(left[0][1]) == {"data": True}
    assert json.loads(left[1][1])["x"] == -0.3     # negative roll lowers the left side
    right = sport_api.requests_for_pose("lean_right", "stand", "euler", roll=0.3, vy=0.3)
    assert json.loads(right[1][1])["x"] == 0.3


def test_back_from_euler_leaves_posture_mode_before_walking():
    back = sport_api.requests_for_pose("stand", "lean_left", "euler", roll=0.3, vy=0.3)
    assert _names(back) == ["Euler", "Pose", "BalanceStand"]
    assert json.loads(back[0][1]) == {"x": 0.0, "y": 0.0, "z": 0.0}
    assert json.loads(back[1][1]) == {"data": False}


def test_move_reaction_steps_and_stops_without_posture_mode():
    left = sport_api.requests_for_pose("lean_left", "stand", "move", roll=0.3, vy=0.3)
    assert _names(left) == ["Move"] and json.loads(left[0][1])["y"] == 0.3
    assert _names(sport_api.requests_for_pose("stand", "lean_left", "move", 0.3, 0.3)) == ["StopMove"]


def test_crouch_and_back():
    assert _names(sport_api.requests_for_pose("crouch", "stand", "euler", 0.3, 0.3)) == ["StandDown"]
    assert _names(sport_api.requests_for_pose("stand", "crouch", "euler", 0.3, 0.3)) == ["StandUp", "BalanceStand"]


@pytest.mark.skipif(not HEADER.exists(), reason="unitree_ros2 checkout missing")
def test_every_id_matches_unitree_header():
    found = dict(re.findall(r"const int32_t (ROBOT_SPORT_API_ID_\w+) = (\d+);", HEADER.read_text()))
    for cmd in sport_api.ALL:
        assert cmd.source in found, f"{cmd.source} not in Unitree header"
        assert int(found[cmd.source]) == cmd.api_id, cmd.name
