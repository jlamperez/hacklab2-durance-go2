"""SNN ball detector adapter (reflex/runtime/ball_snn.py).

The tests that load the network need PyTorch, riverops (installed or the unpacked
wheel in riverops-0/), the scripts in Training/ and out/models/ball_small_best.pt;
they are skipped when any of these is missing.
"""
import importlib.util

import numpy as np
import pytest

from reflex.events.io import EVENT_DTYPE
from reflex.data.windows import make_window
from reflex.runtime.ball_snn import REPO, BallSNNConfig, frame_from_window, fullres_config


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


def _model_available() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    if importlib.util.find_spec("riverops") is None and not (REPO / "riverops-0" / "riverops").is_dir():
        return False
    return (REPO / "Training" / "train_ball_small.py").is_file() and \
        (REPO / BallSNNConfig.checkpoint).is_file()


needs_model = pytest.mark.skipif(not _model_available(),
                                 reason="needs torch, riverops, Training/ and out/models/ball_small_best.pt")


@pytest.fixture(scope="module")
def tracker():
    from reflex.runtime.ball_snn import BallSNNTracker
    return BallSNNTracker()


@needs_model
def test_checkpoint_loads(tracker):
    assert tracker.architecture == "unet-position-small-v1"
    assert tracker.sensor == 320 and tracker.time_steps >= 1 and tracker.window_ms > 0


@needs_model
def test_sparse_frame_is_nothing_seen_without_running_the_network(tracker):
    f = np.zeros((320, 320), dtype=np.uint8)
    f[50, 60] = 1
    t = tracker.predict(f, 123)
    assert not t.found and t.t_us == 123 and tracker.last["p_ball"] is None


@needs_model
def test_output_follows_the_target_contract(tracker):
    rng = np.random.default_rng(0)
    f = np.zeros((320, 320), dtype=np.uint8)
    yy, xx = np.mgrid[:320, :320]
    ring = np.abs(np.hypot(xx - 240, yy - 200) - 18) < 2          # a ball's edge, as events show it
    f[ring & (rng.random((320, 320)) < 0.7)] = 1
    t = tracker.predict(f, 5)
    assert -1.0 <= t.u <= 1.0 and -1.0 <= t.v <= 1.0
    assert t.confidence == 0.0 or t.confidence >= tracker.cfg.min_p
    p = tracker.last["p_ball"]
    assert 0.0 <= p <= 1.0
    # u, v are the pixel position normalised to [-1, 1]
    assert tracker.last["x_px"] / 160 - 1 == pytest.approx(t.u, abs=0.02) or not t.found


@needs_model
def test_wrong_frame_size_is_refused(tracker):
    with pytest.raises(ValueError):
        tracker.predict(np.ones((80, 80), dtype=np.uint8))
