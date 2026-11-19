import numpy as np
import pytest

from reflex.data.episodes import KINDS, run_episode
from reflex.data.windows import WindowConfig, make_window
from reflex.sim.scene import GO2_XML, Go2Sim

pytestmark = pytest.mark.skipif(not GO2_XML.exists(), reason="Go2 model missing: run `pixi run setup`")


@pytest.fixture(scope="module")
def sim():
    return Go2Sim(seed=0)


@pytest.mark.parametrize("label", ["left", "center", "right"])
def test_approach_is_labelled_only_while_the_ball_is_close(sim, label):
    ep = run_episode(sim, np.random.default_rng(3), kind="approach", label=label)
    assert ep.label == label and ep.threat.any()
    first = int(np.argmax(ep.threat))
    assert ep.label_at(int(ep.t_us[0])) == "none"           # before the ball is close
    assert ep.label_at(int(ep.t_us[first + 5])) == label


@pytest.mark.parametrize("kind", [k for k in KINDS if k != "approach"])
def test_negatives_are_never_labelled_as_threats(sim, kind):
    ep = run_episode(sim, np.random.default_rng(4), kind=kind)
    assert ep.label == "none" and not ep.threat.any()
    assert len(ep.events) > 0
    assert (np.diff(ep.events["t"]) >= 0).all()


def test_moving_negatives_make_more_events_than_an_idle_dog(sim):
    def rate(kind):
        ep = run_episode(sim, np.random.default_rng(5), kind=kind, randomize=False)
        return len(ep.events) / (ep.t_us[-1] * 1e-6)
    # Idle is pure sensor noise here. A walker adds real motion on top: that is what
    # makes it a hard negative for anything that just counts events.
    assert rate("walker") > rate("idle") + 10_000


def test_window_shape_and_content(sim):
    cfg = WindowConfig()
    ep = run_episode(sim, np.random.default_rng(6), kind="approach", label="center")
    t_end = int(ep.t_us[np.nonzero(ep.threat)[0][-1]])
    win = make_window(ep.events, t_end, cfg)
    assert win.shape == cfg.shape == (10, 2, 45, 80) and win.dtype == np.uint8
    lo, hi = np.searchsorted(ep.events["t"], [t_end - 50_000, t_end])
    assert win.sum() == hi - lo                 # every event in the window is counted once


def test_make_window_equals_bins_then_downsample():
    import numpy as np

    from reflex.data.windows import WindowConfig, make_window
    from reflex.events import repr as ev_repr
    from reflex.events.emu import EVENT_DTYPE
    rng = np.random.default_rng(3)
    cfg = WindowConfig()
    ev = np.zeros(20_000, dtype=EVENT_DTYPE)
    ev["t"] = np.sort(rng.integers(0, 120_000, len(ev)))
    ev["x"], ev["y"] = rng.integers(0, cfg.width, len(ev)), rng.integers(0, cfg.height, len(ev))
    ev["p"] = rng.integers(0, 2, len(ev))
    t_end = 100_000
    lo, hi = np.searchsorted(ev["t"], [t_end - 50_000, t_end])
    ref = ev_repr.downsample(ev_repr.to_bins(ev[lo:hi], cfg.height, cfg.width, t_end - 50_000, t_end, cfg.n_bins),
                             cfg.downsample)
    assert np.array_equal(make_window(ev, t_end, cfg), np.clip(ref, 0, 255).astype(np.uint8))
