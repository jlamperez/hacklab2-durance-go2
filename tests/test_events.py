import numpy as np

from reflex.events import io as ev_io
from reflex.events import repr as ev_repr
from reflex.events.emu import EmuConfig, EventEmulator

H, W = 32, 48
QUIET = EmuConfig(sigma_c=0.0, noise_rate_hz=0.0, hot_pixel_frac=0.0)


def test_static_scene_gives_no_events():
    emu = EventEmulator(H, W, QUIET)
    frame = np.full((H, W), 0.5, dtype=np.float32)
    emu.reset(frame, 0)
    assert len(emu.step(frame, 2000)) == 0


def test_brightening_gives_on_and_darkening_gives_off():
    emu = EventEmulator(H, W, QUIET)
    dark = np.full((H, W), 0.2, dtype=np.float32)
    bright = dark.copy()
    bright[10:20, 10:20] = 0.8
    emu.reset(dark, 0)
    on = emu.step(bright, 2000)
    assert len(on) > 0 and (on["p"] == 1).all()
    assert on["x"].min() >= 10 and on["x"].max() < 20
    off = emu.step(dark, 4000)
    assert len(off) > 0 and (off["p"] == 0).all()


def test_timestamps_are_sorted_and_inside_the_interval():
    emu = EventEmulator(H, W, EmuConfig(noise_rate_hz=200.0))
    frame = np.full((H, W), 0.5, dtype=np.float32)
    emu.reset(frame, 1000)
    ev = emu.step(frame * 1.8, 3000)
    assert len(ev) > 0
    assert (np.diff(ev["t"]) >= 0).all()
    assert ev["t"].min() >= 1000 and ev["t"].max() <= 3000


def test_moving_edge_has_opposite_polarity_on_each_side():
    emu = EventEmulator(H, W, QUIET)

    def bar(x0):
        f = np.full((H, W), 0.2, dtype=np.float32)
        f[:, x0:x0 + 6] = 0.8
        return f

    emu.reset(bar(10), 0)
    ev = emu.step(bar(12), 2000)       # bright bar moves right
    on, off = ev[ev["p"] == 1], ev[ev["p"] == 0]
    assert on["x"].min() >= 16          # leading edge brightens
    assert off["x"].max() < 12          # trailing edge darkens


def test_frame_and_bins_count_every_event():
    emu = EventEmulator(H, W, EmuConfig(noise_rate_hz=500.0))
    frame = np.full((H, W), 0.5, dtype=np.float32)
    emu.reset(frame, 0)
    ev = emu.step(frame, 10_000)
    assert ev_repr.to_frame(ev, H, W).sum() == len(ev)
    bins = ev_repr.to_bins(ev, H, W, 0, 10_001, n_bins=5)
    assert bins.shape == (5, 2, H, W) and bins.sum() == len(ev)
    assert ev_repr.downsample(bins, 4).shape == (5, 2, H // 4, W // 4)
    assert ev_repr.downsample(bins, 4).sum() == len(ev)


def test_npz_roundtrip(tmp_path):
    emu = EventEmulator(H, W, EmuConfig(noise_rate_hz=500.0))
    frame = np.full((H, W), 0.5, dtype=np.float32)
    emu.reset(frame, 0)
    ev = emu.step(frame, 10_000)
    ev_io.save_npz(tmp_path / "e.npz", ev, H, W, label="left")
    back, meta = ev_io.load_npz(tmp_path / "e.npz")
    assert (back == ev).all()
    assert meta["height"] == H and meta["label"] == "left"
