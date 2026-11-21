import numpy as np

from reflex.runtime.ball_tracker import BallObservation, CentroidTracker
from reflex.runtime.follow_fsm import FollowConfig, FollowFSM, Greet, StopTurn, Turn

RNG = np.random.default_rng(0)


def blob(cx, cy, n=80, spread=6):
    return RNG.normal(cx, spread, n).clip(0, 319), RNG.normal(cy, spread, n).clip(0, 319)


def test_tracker_finds_a_compact_blob():
    x, y = blob(250, 160)
    obs = CentroidTracker().observe(x, y)
    assert obs.found and abs(obs.x_frac - 250 / 320) < 0.02 and abs(obs.y_frac - 0.5) < 0.02


def test_tracker_ignores_sparse_noise():
    assert not CentroidTracker().observe(RNG.uniform(0, 320, 5), RNG.uniform(0, 320, 5)).found


def test_tracker_rejects_events_spread_over_the_image():
    # the camera turning: thousands of events everywhere, no dominant blob
    x, y = RNG.uniform(0, 320, 4000), RNG.uniform(0, 320, 4000)
    assert not CentroidTracker().observe(x, y).found


def test_tracker_ignores_a_little_noise_next_to_the_ball():
    bx, by = blob(60, 200)
    x = np.concatenate([bx, RNG.uniform(0, 320, 10)])
    y = np.concatenate([by, RNG.uniform(0, 320, 10)])
    obs = CentroidTracker().observe(x, y)
    assert obs.found and abs(obs.x_frac - 60 / 320) < 0.03


def seen(x):
    return BallObservation(True, x, 0.6, 80, 90)


NONE = BallObservation(False)


def run(fsm, obs_at, t0, t1, dt=0.02):
    """Feed obs_at(t) from t0 to t1; return all actions with their times."""
    actions, t = [], t0
    while t < t1:
        actions += [(round(t, 2), a) for a in fsm.update(obs_at(t), t)]
        t += dt
    return actions


def test_centered_ball_that_stops_is_greeted_without_turning():
    fsm = FollowFSM()
    acts = run(fsm, lambda t: seen(0.5) if t < 1.0 else NONE, 0, 2.0)
    kinds = [type(a) for _, a in acts]
    assert kinds == [Greet] and fsm.state == "greeting"
    assert 1.45 <= acts[0][0] <= 1.6          # stop_s after the ball disappeared


def test_ball_on_the_left_makes_the_robot_turn_left_then_stop_turning():
    fsm = FollowFSM()
    acts = run(fsm, lambda t: seen(0.2), 0, 1.5)
    turns = [a for _, a in acts if isinstance(a, Turn)]
    assert turns and turns[0].rate > 0                           # + = left
    assert any(isinstance(a, StopTurn) for _, a in acts)


def test_observations_are_ignored_while_turning():
    fsm = FollowFSM()
    run(fsm, lambda t: seen(0.2), 0, 0.09)       # found, then confirmed at the same spot
    assert fsm.state == "turning"
    # during the turn the camera sees the ball "disappear": must not count as stopped
    acts = run(fsm, lambda t: NONE, 0.09, 0.35)
    assert not any(isinstance(a, Greet) for _, a in acts)


def test_ball_leaving_at_the_edge_triggers_a_search_turn_not_a_greeting():
    fsm = FollowFSM(FollowConfig(deadband_deg=60))      # wide deadband: no aiming turns here
    acts = run(fsm, lambda t: seen(0.96) if t < 1.0 else NONE, 0, 2.0)
    assert not any(isinstance(a, Greet) for _, a in acts)
    turns = [a for _, a in acts if isinstance(a, Turn)]
    assert turns and turns[0].rate < 0                           # towards the right edge


def test_every_state_times_out_to_idle():
    fsm = FollowFSM()
    run(fsm, lambda t: seen(0.5) if t < 1.0 else NONE, 0, 1.6)  # greeting started
    run(fsm, lambda t: NONE, 1.6, 1.6 + fsm.cfg.greet_s + fsm.cfg.cooldown_s + 0.2)
    assert fsm.state == "idle"


def test_a_single_flash_elsewhere_while_tracking_causes_no_turn():
    fsm = FollowFSM()
    run(fsm, lambda t: seen(0.5), 0, 0.2)             # tracking a centered ball
    acts = run(fsm, lambda t: seen(0.15) if abs(t - 0.3) < 0.01 else seen(0.5), 0.2, 0.5)
    assert not any(isinstance(a, Turn) for _, a in acts)


def test_a_flicker_of_detections_does_not_start_tracking():
    fsm = FollowFSM()
    run(fsm, lambda t: seen(0.5) if int(t / 0.02) % 2 == 0 else NONE, 0, 1.0)
    assert fsm.state == "idle"
