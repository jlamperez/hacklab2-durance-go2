"""Six zones -> sport-mode requests (reflex/runtime/zone_look.py), the behavior node's mode:=look."""
import json

import pytest

from reflex.runtime import sport_api as api
from reflex.runtime.follow import FollowConfig
from reflex.runtime.zone_look import HARD_MAX_PITCH, HARD_MAX_YAW, ZoneLook

RATE = 0.05   # 20 Hz, as the behavior node ticks


def run(look: ZoneLook, zone: str, t0: float, seconds: float, alive: bool = True) -> list:
    if zone is not None:
        look.set_zone(zone)
    out, n = [], int(round(seconds / RATE))
    for i in range(n):
        out += look.tick(t0 + i * RATE, source_alive=alive)
    return out


def eulers(requests):
    return [json.loads(r.param) for r in requests if r.cmd == api.EULER]


def names(requests):
    return [r.cmd.name for r in requests]


def make(**kw):
    look = ZoneLook(FollowConfig(camera_on_trunk=False, look_yaw=0.15, look_pitch=0.1, **kw))
    return look, look.enable(0.0)


def test_enable_stands_balanced():
    _, out = make()
    assert names(out) == ["BalanceStand"]


@pytest.mark.parametrize("zone, gesture, yaw, pitch", [
    ("upper_left", "left_up", 0.15, -0.1), ("upper_middle", "up", 0.0, -0.1),
    ("upper_right", "right_up", -0.15, -0.1), ("lower_left", "left_down", 0.15, 0.1),
    ("lower_middle", "down", 0.0, 0.1), ("lower_right", "right_down", -0.15, 0.1)])
def test_each_zone_enters_posture_mode_then_euler(zone, gesture, yaw, pitch):
    look, _ = make()
    out = run(look, zone, 0.0, 1.0)
    assert names(out)[:2] == ["Pose", "Euler"]
    assert json.loads(out[0].param) == {"data": True}
    e = eulers(out)[0]
    assert (e["x"], e["y"], e["z"]) == pytest.approx((0.0, pitch, yaw))   # left = +yaw, up = nose up = -y
    assert look.gesture == gesture


def test_lost_ball_looks_straight_and_leaves_posture_mode():
    look, _ = make(lost_s=1.0)
    run(look, "upper_left", 0.0, 1.0)
    out = run(look, "none", 1.0, 2.0)
    assert names(out) == ["Euler", "Pose"]
    assert eulers(out)[0] == {"x": 0.0, "y": 0.0, "z": 0.0}
    assert json.loads(out[1].param) == {"data": False}
    assert not look.posture


def test_silent_publisher_counts_as_none():
    look, _ = make(lost_s=1.0)
    run(look, "lower_right", 0.0, 1.0)
    out = run(look, None, 1.0, 2.0, alive=False)
    assert eulers(out) == [{"x": 0.0, "y": 0.0, "z": 0.0}]


def test_nothing_before_the_first_zone():
    look, _ = make()
    assert run(look, None, 0.0, 3.0) == []


def test_disable_straightens_once_in_posture_mode():
    look, _ = make()
    run(look, "upper_right", 0.0, 1.0)
    out = look.disable(1.0)
    assert names(out) == ["Euler", "Pose"]
    assert names(look.disable(1.1)) == ["Euler"]          # already out of posture mode


def test_inverted_pitch_and_hard_limits():
    look = ZoneLook(FollowConfig(camera_on_trunk=False, look_yaw=2.0, look_pitch=2.0), pitch_sign=-1)
    look.enable(0.0)
    e = eulers(run(look, "upper_left", 0.0, 1.0))[0]
    assert e["z"] == pytest.approx(HARD_MAX_YAW)
    assert e["y"] == pytest.approx(HARD_MAX_PITCH)          # up, inverted: +y


def test_unknown_zone_is_rejected_and_not_kept():
    look, _ = make()
    look.set_zone("upper_left")
    with pytest.raises(ValueError):
        look.set_zone("left")
    assert look.zone == "upper_left"
