"""Zone name -> Target (reflex/runtime/zone_target.py), for the FPGA ball node's /durance/ball_zone."""
import pytest

from reflex.runtime.follow import FollowConfig, FollowController
from reflex.runtime.zone_target import ZONES, target_from_decision, target_from_zone, zone_center


def test_none_is_nothing_seen():
    for name in ("none", "", " NONE "):
        t = target_from_zone(name, 7)
        assert not t.found and t.t_us == 7


def test_zone_is_its_centre():
    assert ZONES[0] == "upper_left" and ZONES[5] == "lower_right"
    t = target_from_zone("lower_right", 3, 0.9)
    assert (t.u, t.v, t.confidence, t.t_us) == pytest.approx((2 / 3, 0.5, 0.9, 3))
    assert target_from_zone("upper_middle").u == pytest.approx(0.0)
    assert zone_center(3) == pytest.approx((-2 / 3, 0.5))           # lower_left


def test_unknown_zone_raises():
    with pytest.raises(ValueError):
        target_from_zone("left")


@pytest.mark.parametrize("zone, gesture", [
    ("lower_left", "left_down"), ("lower_middle", "down"), ("lower_right", "right_down"),
    ("upper_left", "left_up"), ("upper_middle", "up"), ("upper_right", "right_up")])
def test_look_mode_points_at_each_zone(zone, gesture):
    ctrl = FollowController(FollowConfig(mode="look", camera_on_trunk=False))
    ctrl.enable(0.0)
    for i in range(20):                       # 20 Hz, as zone_target republishes
        ctrl.update(target_from_zone(zone, i * 50_000), i * 0.05)
    assert ctrl.gesture == gesture


def test_decision_is_a_column_at_the_middle_row():
    t = target_from_decision("left", 0.55, 9)
    assert (t.u, t.v, t.confidence, t.t_us) == pytest.approx((-2 / 3, 0.0, 0.55, 9))
    assert target_from_decision("right", 0.5).u == pytest.approx(2 / 3)
    assert not target_from_decision("none", 0.9).found
    with pytest.raises(ValueError):
        target_from_decision("upper_left", 1.0)


@pytest.mark.parametrize("label, gesture", [("left", "left"), ("center", "center"), ("right", "right")])
def test_look_mode_follows_fpga_decisions(label, gesture):
    # fpga_perception: one decision per 10 ms frame, confidence around 0.5-0.55 on 22 Nov
    ctrl = FollowController(FollowConfig(mode="look", camera_on_trunk=False, min_confidence=0.45))
    ctrl.enable(0.0)
    for i in range(50):
        ctrl.update(target_from_decision(label, 0.5, i * 10_000), i * 0.01)
    assert ctrl.gesture == gesture
