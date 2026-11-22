"""SNN ball detector adapter (reflex/runtime/ball_snn.py).

The tests that load the network need PyTorch, riverops (installed or the unpacked
wheel in riverops-0/), the scripts in Training/ (feat/training) and
out/models/ball_zones_best.pt; they are skipped when any of these is missing.
"""
import importlib.util

import numpy as np
import pytest

from reflex.events.io import EVENT_DTYPE
from reflex.data.windows import make_window
from reflex.runtime.ball_snn import REPO, BallSNNConfig, frame_from_window, fullres_config, zone_center


def _events(xs, ys, ts, ps=None):
    ev = np.zeros(len(xs), dtype=EVENT_DTYPE)
    ev["x"], ev["y"], ev["t"] = xs, ys, ts
    ev["p"] = 0 if ps is None else ps
    return ev


def test_fullres_window_is_one_bin_at_full_resolution():
    cfg = fullres_config(10.0)
    assert cfg.shape == (1, 2, 320, 320)
    ev = _events([5, 5, 300], [7, 7, 310], [1_000, 2_000, 9_000], [0, 1, 1])
    w = make_window(ev, 10_000, cfg)
    assert w.shape == (1, 2, 320, 320)
    assert w[0, 0, 7, 5] == 1 and w[0, 1, 7, 5] == 1 and w[0, 1, 310, 300] == 1


def test_frame_is_binary_and_ignores_polarity_and_bins():
    w = np.zeros((3, 2, 320, 320), dtype=np.uint8)
    w[0, 0, 10, 20] = 4
    w[2, 1, 10, 20] = 1
    w[1, 1, 100, 200] = 7
    f = frame_from_window(w)
    assert f.shape == (320, 320) and f.dtype == np.uint8
    assert set(np.unique(f)) == {0, 1}
    assert f.sum() == 2 and f[10, 20] == 1 and f[100, 200] == 1


def test_frame_with_polarity_keeps_two_channels():
    w = np.zeros((3, 2, 320, 320), dtype=np.uint8)
    w[0, 0, 10, 20] = 4
    w[2, 1, 10, 20] = 1
    w[1, 1, 100, 200] = 7
    f = frame_from_window(w, polarity=True)
    assert f.shape == (2, 320, 320) and f.dtype == np.uint8
    assert f[0, 10, 20] == 1 and f[1, 10, 20] == 1 and f[1, 100, 200] == 1 and f.sum() == 3


def test_zone_centres_follow_the_target_contract():
    # zones row by row from the top left, as train_ball_zones.CLASSES
    assert zone_center(0) == pytest.approx((-2 / 3, -0.5))     # upper_left
    assert zone_center(1) == pytest.approx((0.0, -0.5))        # upper_middle
    assert zone_center(5) == pytest.approx((2 / 3, 0.5))       # lower_right


def _model_available() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    if importlib.util.find_spec("riverops") is None and not (REPO / "riverops-0" / "riverops").is_dir():
        return False
    return (REPO / "Training" / "train_ball_zones.py").is_file() and \
        (REPO / BallSNNConfig.checkpoint).is_file()


needs_model = pytest.mark.skipif(not _model_available(),
                                 reason="needs torch, riverops, Training/ and out/models/ball_zones_best.pt")


@pytest.fixture(scope="module")
def tracker():
    from reflex.runtime.ball_snn import BallSNNTracker
    return BallSNNTracker()


@needs_model
def test_checkpoint_loads(tracker):
    assert tracker.architecture == "zones-small-polarity-v2" and tracker.zones
    assert tracker.sensor == 320 and tracker.time_steps >= 1 and tracker.window_ms > 0
    assert tracker.channels == 2 and len(tracker.classes) == 7


@needs_model
def test_sparse_frame_is_nothing_seen_without_running_the_network(tracker):
    f = np.zeros((2, 320, 320), dtype=np.uint8)
    f[0, 50, 60] = 1
    t = tracker.predict(f, 123)
    assert not t.found and t.t_us == 123 and tracker.last["p_ball"] is None


@needs_model
def test_output_follows_the_target_contract(tracker):
    rng = np.random.default_rng(0)
    f = np.zeros((2, 320, 320), dtype=np.uint8)
    yy, xx = np.mgrid[:320, :320]
    ring = np.abs(np.hypot(xx - 240, yy - 200) - 18) < 2          # a ball's edge, as events show it
    f[0][ring & (rng.random((320, 320)) < 0.35)] = 1
    f[1][ring & (rng.random((320, 320)) < 0.35)] = 1
    t = tracker.predict(f, 5)
    assert -1.0 <= t.u <= 1.0 and -1.0 <= t.v <= 1.0
    assert t.confidence == 0.0 or t.confidence >= tracker.cfg.min_p
    assert 0.0 <= tracker.last["p_ball"] <= 1.0 and tracker.last["zone"] in tracker.classes
    # a found target sits at its zone's centre
    if t.found:
        assert (t.u, t.v) == pytest.approx(zone_center(tracker.classes.index(tracker.last["zone"])))


@needs_model
def test_wrong_frame_size_is_refused(tracker):
    with pytest.raises(ValueError):
        tracker.predict(np.ones((80, 80), dtype=np.uint8))
    with pytest.raises(ValueError):
        tracker.predict(np.ones((320, 320), dtype=np.uint8))      # one channel: polarity missing


@needs_model
def test_window_from_a_recording_runs(tracker):
    ev = _events(np.full(400, 250), np.repeat(np.arange(180, 220), 10),
                 np.linspace(0, 29_000, 400).astype(np.int64), np.tile([0, 1], 200))
    t = tracker.predict(tracker.frame(make_window(ev, 30_000, fullres_config(30.0))), 30_000)
    assert t.t_us == 30_000 and tracker.last["pixels"] >= tracker.cfg.min_pixels
