import math

import numpy as np
import pytest

from reflex.runtime.follow import FollowConfig, FollowController
from reflex.runtime.target import CentroidConfig, CentroidTracker, Target, no_target

SHAPE = (10, 2, 45, 80)


def blob(row, col, size=4, count=3, shape=SHAPE):
    w = np.zeros(shape, dtype=np.uint8)
    w[-4:, :, row:row + size, col:col + size] = count
    return w


# --- centroid tracker -----------------------------------------------------------------------

def test_tracker_sees_nothing_in_an_empty_window():
    assert not CentroidTracker().predict(np.zeros(SHAPE, dtype=np.uint8)).found


@pytest.mark.parametrize("col,sign", [(4, -1), (72, 1)])
def test_tracker_reports_the_side_of_the_blob(col, sign):
    t = CentroidTracker().predict(blob(30, col))
    assert t.found and np.sign(t.u) == sign and t.v > 0


def test_tracker_v_grows_towards_the_bottom_of_the_image():
    tr = CentroidTracker()
    assert tr.predict(blob(40, 38)).v > tr.predict(blob(20, 38)).v


def test_tracker_rejects_motion_everywhere():
    w = np.ones(SHAPE, dtype=np.uint8)            # the whole image fires: the dog moved
    assert not CentroidTracker().predict(w).found


def test_tracker_ignores_the_zone_above_zone_top():
    assert not CentroidTracker().predict(blob(2, 38)).found
    assert CentroidTracker(CentroidConfig(zone_top=0.0)).predict(blob(2, 38)).found


def test_tracker_calibration_raises_the_threshold_above_the_background():
    rng = np.random.default_rng(0)
    idle = (rng.random((20,) + SHAPE) < 0.01).astype(np.uint8)
    tr = CentroidTracker()
    thr = tr.calibrate(idle)
    assert thr >= 5.0
    assert not any(tr.predict(w).found for w in idle)


# --- follow controller ----------------------------------------------------------------------

def seen(u, v=0.0, conf=0.9):
    return Target(u, v, conf)


def run(ctrl, targets, t0=0.0, dt=0.02):
    """Feed targets every dt; return (t, command) for everything sent."""
    out, t = [], t0
    for tg in targets:
        out += [(t, c) for c in ctrl.update(tg, t)]
        t += dt
    return out, t


def enabled(**kw):
    ctrl = FollowController(FollowConfig(**kw))
    assert [c.kind for c in ctrl.enable(0.0)] == ["balance_stand"]
    return ctrl


def test_disabled_controller_sends_nothing():
    ctrl = FollowController()
    sent, _ = run(ctrl, [seen(0.5)] * 50)
    assert sent == []


def test_target_on_the_right_turns_right_and_target_on_the_left_turns_left():
    for u, sign in ((0.6, -1), (-0.6, 1)):
        ctrl = enabled()
        sent, _ = run(ctrl, [seen(u)] * 3)
        moves = [c for _, c in sent if c.kind == "move"]
        assert moves and np.sign(moves[0].wz) == sign


def test_one_sighting_is_not_enough():
    ctrl = enabled(consecutive=2)
    sent, _ = run(ctrl, [seen(0.6), no_target(), seen(0.6), no_target()])
    assert sent == []


def test_low_confidence_is_ignored():
    ctrl = enabled()
    sent, _ = run(ctrl, [seen(0.6, conf=0.2)] * 20)
    assert sent == []


def test_every_burst_ends_with_stopmove_and_is_short():
    cfg = dict(move_s=0.5, settle_s=0.6)
    ctrl = enabled(**cfg)
    sent, _ = run(ctrl, [seen(0.0, v=-0.5)] * 200)          # straight ahead, far: walk
    kinds = [c.kind for _, c in sent]
    assert "move" in kinds and kinds[-1] in ("stop", "move")
    t_first_move = None
    for t, c in sent:
        if c.kind == "move" and t_first_move is None:
            t_first_move = t
        if c.kind == "stop" and t_first_move is not None:
            assert t - t_first_move <= cfg["move_s"] + 0.05
            t_first_move = None


def test_nothing_seen_during_settling_counts():
    ctrl = enabled(move_s=0.2, settle_s=1.0)
    sent, t = run(ctrl, [seen(0.6)] * 2)
    assert ctrl.state == "moving"
    sent, t = run(ctrl, [None] * 12, t0=t)                    # burst ends
    assert ctrl.state == "settling"
    sent, t = run(ctrl, [seen(-0.8)] * 20, t0=t)              # still settling: ignored
    assert not any(c.kind == "move" for _, c in sent)


def test_walks_forward_only_when_target_is_ahead_and_not_close():
    ctrl = enabled()
    _, _ = run(ctrl, [seen(0.0, v=-0.3)] * 2)
    assert ctrl.burst.vx > 0
    ctrl = enabled()
    _, _ = run(ctrl, [seen(0.9, v=-0.3)] * 2)                 # far to the side: turn only
    assert ctrl.burst.vx == 0 and ctrl.burst.wz < 0
    ctrl = enabled()
    sent, _ = run(ctrl, [seen(0.0, v=0.8)] * 2)               # close and ahead: stay
    assert sent == []


def test_speeds_are_capped():
    ctrl = enabled(max_vx=0.3, max_wz=0.5)
    sent, _ = run(ctrl, [seen(u, v=-1.0) for u in np.linspace(-1, 1, 400)])
    for _, c in sent:
        assert 0.0 <= c.vx <= 0.3 + 1e-9 and abs(c.wz) <= 0.5 + 1e-9


def test_travel_budget_halts_until_enabled_again():
    ctrl = enabled(max_travel_m=0.5, move_s=0.5, settle_s=0.1, max_vx=0.3)
    sent, t = run(ctrl, [seen(0.0, v=-0.8)] * 500)
    assert ctrl.state == "halted"
    assert sent[-1][1].kind == "stop"
    assert ctrl.travel_m <= 0.5
    more, t = run(ctrl, [seen(0.0, v=-0.8)] * 100, t0=t)
    assert more == []
    ctrl.enable(t)
    assert ctrl.state == "looking" and ctrl.travel_m == 0.0


def test_disable_stops_at_any_time():
    ctrl = enabled()
    run(ctrl, [seen(0.0, v=-0.5)] * 2)
    assert ctrl.state == "moving"
    assert [c.kind for c in ctrl.disable(1.0)] == ["stop"]
    sent, _ = run(ctrl, [seen(0.0, v=-0.5)] * 50, t0=1.0)
    assert sent == []


def test_gaze_turns_the_trunk_within_limits_and_recentres_when_lost():
    ctrl = enabled(mode="gaze", max_yaw=0.35, lost_s=1.0, move_s=0.2, settle_s=0.2)
    sent, t = run(ctrl, [seen(-0.9)] * 200)                   # target far left, for 4 s
    eulers = [c for _, c in sent if c.kind == "euler"]
    assert eulers and all(0 < c.yaw <= 0.35 for c in eulers)
    assert not any(c.kind == "move" for _, c in sent)
    sent, t = run(ctrl, [no_target()] * 100, t0=t)            # gone for 2 s
    assert [c.yaw for _, c in sent if c.kind == "euler"] == [0.0]


# --- look mode: nine gestures ---------------------------------------------------------------

def looker(**kw):
    kw = {"mode": "look", "move_s": 0.2, "settle_s": 0.2, "lost_s": 1.0, **kw}
    return enabled(**kw)


def gestures_for(ctrl, targets, t0=0.0):
    sent, t = run(ctrl, targets, t0=t0)
    assert all(c.kind == "euler" for _, c in sent)       # never walks
    return [c.reason.split(":")[0].removeprefix("look ") for _, c in sent], t


# u, v for a target on a fixed camera (camera_on_trunk=False): image position -> gesture.
@pytest.mark.parametrize("u,v,name", [
    (0.0, 0.0, None), (-0.8, 0.0, "left"), (0.8, 0.0, "right"), (0.0, -0.8, "up"), (0.0, 0.8, "down"),
    (-0.8, -0.8, "left_up"), (-0.8, 0.8, "left_down"), (0.8, -0.8, "right_up"), (0.8, 0.8, "right_down"),
])
def test_look_maps_each_ninth_of_the_image_to_its_gesture(u, v, name):
    ctrl = looker(camera_on_trunk=False)
    names, _ = gestures_for(ctrl, [seen(u, v)] * 20)
    assert names == ([] if name is None else [name])      # centre: already there, nothing sent
    assert ctrl.gesture == (name or "center")


def test_look_signs_left_is_positive_yaw_and_up_is_nose_up():
    ctrl = looker(camera_on_trunk=False, look_yaw=0.3, look_pitch=0.2)
    sent, _ = run(ctrl, [seen(-0.8, -0.8)] * 20)
    (_, c), = sent
    assert c.yaw == pytest.approx(0.3) and c.pitch == pytest.approx(-0.2)
    ctrl = looker(camera_on_trunk=False, look_yaw=0.3, look_pitch=0.2)
    (_, c), = run(ctrl, [seen(0.8, 0.8)] * 20)[0]
    assert c.yaw == pytest.approx(-0.3) and c.pitch == pytest.approx(0.2)


def test_look_with_camera_on_trunk_does_not_look_back_once_facing_the_target():
    # Target 20 deg left. After the left gesture (17 deg) the camera sees it 3 deg left of
    # centre; the controller adds the gesture back and stays on "left".
    ctrl = looker(look_yaw=0.3)
    u_before = math.tan(math.radians(-20)) / math.tan(math.radians(45))
    u_after = math.tan(math.radians(-20 + math.degrees(0.3))) / math.tan(math.radians(45))
    names, t = gestures_for(ctrl, [seen(u_before)] * 10)
    assert names == ["left"]
    names, _ = gestures_for(ctrl, [seen(u_after)] * 100, t0=t)
    assert names == [] and ctrl.gesture == "left"


def test_look_hysteresis_stops_flicker_on_a_boundary():
    ctrl = looker(camera_on_trunk=False, look_h_deg=12.0, look_hyst_deg=4.0)
    u = lambda deg: -math.tan(math.radians(deg)) / math.tan(math.radians(45))
    names, t = gestures_for(ctrl, [seen(u(14))] * 20)        # past 12 but not past 12 + 4
    assert names == []
    names, t = gestures_for(ctrl, [seen(u(17))] * 20, t0=t)  # past 16: left
    assert names == ["left"]
    names, t = gestures_for(ctrl, [seen(u(10))] * 20, t0=t)  # back under 12 but not under 8
    assert names == []
    names, _ = gestures_for(ctrl, [seen(u(6))] * 20, t0=t)   # under 8: centre
    assert names == ["center"]


def test_look_waits_for_settling_and_recentres_when_lost():
    ctrl = looker(camera_on_trunk=False)
    sent, t = run(ctrl, [seen(-0.8)] * 5)                    # 2 sightings -> left, at t = 0.02
    assert [c.reason.startswith("look left") for _, c in sent] == [True]
    sent, t = run(ctrl, [seen(0.8)] * 15, t0=t)              # moving + settling: 0.4 s
    assert sent == []
    sent, t = run(ctrl, [seen(0.8)] * 5, t0=t)
    assert [c.reason.split(":")[0] for _, c in sent] == ["look right"]
    sent, _ = run(ctrl, [no_target()] * 100, t0=t)
    assert [(c.yaw, c.pitch) for _, c in sent] == [(0.0, 0.0)] and ctrl.gesture == "center"


def test_look_disable_and_manual_gestures():
    ctrl = looker()
    (c,) = ctrl.look("right_down", 0.0)
    assert c.yaw < 0 and c.pitch > 0 and ctrl.gesture == "right_down"
    (c,) = ctrl.disable(0.1)
    assert (c.kind, c.yaw, c.pitch) == ("euler", 0.0, 0.0) and ctrl.gesture == "center"
    (c,) = ctrl.look("up", 0.2)                               # works while disabled too
    assert c.pitch < 0 and ctrl.state == "disabled"
    with pytest.raises(ValueError):
        ctrl.look("backflip", 0.3)


def test_angle_matches_the_field_of_view():
    ctrl = FollowController(FollowConfig(hfov_deg=90.0))
    assert math.isclose(ctrl.angle_to(1.0), -math.radians(45), rel_tol=1e-6)
    assert ctrl.angle_to(0.0) == 0.0


# --- scripted target path (simulation) ------------------------------------------------------

def test_target_path_pauses_then_rolls_at_its_speed():
    from reflex.sim.follow_sim import TargetPath
    path = TargetPath(waypoints=[(1.0, 0.0), (2.0, 0.0)], speed=0.5, pause_s=1.0)
    p0, v0 = path.at(0.5)
    assert np.allclose(p0, (1.0, 0.0)) and not v0.any()            # pausing: still
    p1, v1 = path.at(1.0 + 1.0)
    assert np.allclose(p1, (1.5, 0.0)) and np.allclose(v1, (0.5, 0.0))
    p2, v2 = path.at(100.0)
    assert np.allclose(p2, (2.0, 0.0)) and not v2.any()            # end of the path: stays
