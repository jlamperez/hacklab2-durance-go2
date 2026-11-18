import numpy as np

from reflex.data.windows import WindowConfig
from reflex.events.emu import EVENT_DTYPE
from reflex.runtime.windower import StreamWindower


def chunk(t_from, t_to, n=50, seed=0):
    rng = np.random.default_rng(seed)
    ev = np.zeros(n, dtype=EVENT_DTYPE)
    ev["t"] = np.sort(rng.integers(t_from, t_to, n))
    ev["x"], ev["y"], ev["p"] = rng.integers(0, 320, n), rng.integers(0, 180, n), rng.integers(0, 2, n)
    return ev


def test_window_contains_only_the_last_window_ms():
    w = StreamWindower(WindowConfig())
    w.push(chunk(0, 40_000, seed=1))          # too old
    w.push(chunk(60_000, 100_000, seed=2))    # inside the last 50 ms
    win = w.window(100_000)
    assert win.shape == (10, 2, 45, 80) and win.sum() == 50


def test_old_chunks_are_dropped():
    w = StreamWindower()
    for i in range(100):
        w.push(chunk(i * 10_000, (i + 1) * 10_000, seed=i))
        w.window((i + 1) * 10_000)
    assert len(w._chunks) <= 7


def test_empty_stream_gives_an_empty_window():
    assert StreamWindower().window(1_000_000).sum() == 0
