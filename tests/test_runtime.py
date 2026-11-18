import numpy as np
import pytest

from reflex.runtime.detectors import Decision, SimpleDetector
from reflex.runtime.reflex_fsm import BEHAVIORS, FSMConfig, ReflexFSM
from reflex.sim.scene import POSES

SHAPE = (10, 2, 45, 80)


def window_with_burst(col_lo, col_hi, row_lo=25, row_hi=40, count=3):
    w = np.zeros(SHAPE, dtype=np.uint8)
    w[-4:, :, row_lo:row_hi, col_lo:col_hi] = count
    return w


def test_simple_detector_is_quiet_on_an_empty_window():
    assert SimpleDetector().predict(np.zeros(SHAPE, dtype=np.uint8)).label == "none"


@pytest.mark.parametrize("cols,label", [((2, 20), "left"), ((32, 48), "center"), ((60, 78), "right")])
def test_simple_detector_reports_the_side_of_the_burst(cols, label):
    d = SimpleDetector().predict(window_with_burst(*cols))
    assert d.label == label and d.confidence >= 0.5


def test_simple_detector_ignores_the_upper_part_of_the_image():
    assert SimpleDetector().predict(window_with_burst(30, 50, row_lo=0, row_hi=12)).label == "none"


def test_simple_detector_ignores_old_events():
    w = np.zeros(SHAPE, dtype=np.uint8)
    w[:4, :, 25:40, 30:50] = 3            # only in the oldest time steps
    assert SimpleDetector().predict(w).label == "none"


def test_calibration_puts_the_threshold_above_the_background():
    rng = np.random.default_rng(0)
    idle = (rng.random((20, *SHAPE)) < 0.02).astype(np.uint8)
    det = SimpleDetector()
    thr = det.calibrate(idle)
    assert all(det.predict(w).label == "none" for w in idle)
    assert det.predict(window_with_burst(30, 50, count=5)).label == "center" and thr > 5


def threat(label="left", conf=0.9):
    return Decision(label, conf)


def test_fsm_needs_consecutive_agreeing_threats():
    fsm = ReflexFSM(FSMConfig(consecutive=3))
    assert fsm.update(threat("left"), 0.00) is None
    assert fsm.update(threat("left"), 0.02) is None
    act = fsm.update(threat("left"), 0.04)
    assert act is not None and act.name == "lean_left" and act.label == "left"


def test_fsm_streak_is_broken_by_none_low_confidence_or_a_different_side():
    for breaker in (Decision("none", 0.9), threat("left", conf=0.3), threat("right")):
        fsm = ReflexFSM(FSMConfig(consecutive=2))
        fsm.update(threat("left"), 0.00)
        assert fsm.update(breaker, 0.02) is None
        assert fsm.state == "idle"


def test_fsm_holds_then_returns_to_stand_then_rests():
    cfg = FSMConfig(consecutive=1, hold_s=0.5, refractory_s=1.0)
    fsm = ReflexFSM(cfg)
    assert fsm.update(threat("center"), 0.0).name == "crouch"
    assert fsm.update(threat("left"), 0.3) is None            # ignored while reacting
    back = fsm.update(None, 0.5)
    assert back.name == "stand" and fsm.state == "refractory"
    assert fsm.update(threat("left"), 1.0) is None            # ignored while resting
    assert fsm.update(None, 1.5) is None and fsm.state == "idle"
    assert fsm.update(threat("right"), 1.6).name == "lean_right"


def test_dodge_mode_moves_away_from_the_ball():
    fsm = ReflexFSM(FSMConfig(mode="dodge", consecutive=1))
    assert fsm.update(threat("left"), 0.0).name == "lean_right"


def test_every_behavior_is_a_pose_the_simulator_knows():
    for mode in BEHAVIORS.values():
        assert set(mode.values()) <= set(POSES)
