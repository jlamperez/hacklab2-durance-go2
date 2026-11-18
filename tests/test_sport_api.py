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


@pytest.mark.skipif(not HEADER.exists(), reason="unitree_ros2 checkout missing")
def test_every_id_matches_unitree_header():
    found = dict(re.findall(r"const int32_t (ROBOT_SPORT_API_ID_\w+) = (\d+);", HEADER.read_text()))
    for cmd in sport_api.ALL:
        assert cmd.source in found, f"{cmd.source} not in Unitree header"
        assert int(found[cmd.source]) == cmd.api_id, cmd.name
