"""Majority vote over zone decisions (reflex/runtime/zone_vote.py), the stand-in for ball_node."""
import pytest

from reflex.runtime.zone_vote import ZoneVoteConfig, ZoneVoter


def test_publishes_only_on_change():
    v = ZoneVoter(ZoneVoteConfig(window=5, min_votes=3))
    out = [v.update(z) for z in ["lower_left"] * 5]
    assert out == [None, None, "lower_left", None, None]
    assert v.current == "lower_left"


def test_single_outliers_do_not_switch():
    v = ZoneVoter(ZoneVoteConfig(window=5, min_votes=3))
    for z in ["lower_left"] * 3:
        v.update(z)
    assert [v.update(z) for z in ("lower_right", "none", "lower_left")] == [None, None, None]
    assert v.current == "lower_left"


def test_falls_back_to_none():
    v = ZoneVoter(ZoneVoteConfig(window=3, min_votes=2))
    for z in ("upper_right", "upper_right"):
        v.update(z)
    assert v.update("none") is None
    assert v.update("none") == "none"


def test_window_one_passes_everything_through():
    v = ZoneVoter(ZoneVoteConfig(window=1, min_votes=1))
    assert [v.update(z) for z in ("lower_middle", "lower_middle", "upper_left")] == ["lower_middle", None, "upper_left"]


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        ZoneVoter(ZoneVoteConfig(window=3, min_votes=4))
    with pytest.raises(ValueError):
        ZoneVoter().update("left")
