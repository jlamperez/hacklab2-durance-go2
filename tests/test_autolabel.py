import json

import numpy as np
import pytest

from reflex.river import autolabel as al
from reflex.river import data as rd

CFG = rd.FrameConfig()          # 320 px sensor, 10 ms frames, 80x80 grid
LAB = rd.LabelConfig()
F = 10_000                      # µs per frame


def _ev(t, x, y):
    ev = np.zeros(len(t), dtype=rd.EVENT_DTYPE)
    ev["t"], ev["x"], ev["y"], ev["p"] = t, x, y, 1
    return ev


def _blob(rng, frames, x_of_f, y=160, n=40, r=8):
    """A compact cluster of n events per frame around (x_of_f(f), y)."""
    t, x, yy = [], [], []
    for f in frames:
        t += list(f * F + rng.integers(0, F, n))
        x += list(np.clip(x_of_f(f) + rng.integers(-r, r + 1, n), 0, 319))
        yy += list(np.clip(y + rng.integers(-r, r + 1, n), 0, 319))
    return _ev(t, x, yy)


def _edge(rng, frames, x, n=120, y0=40, y1=300):
    """A long vertical edge (a person, a door frame), many events per frame."""
    t = np.concatenate([f * F + rng.integers(0, F, n) for f in frames])
    return _ev(t, np.full(len(t), x), rng.integers(y0, y1, len(t)))


def _noise(rng, n_frames, rate_per_frame=20):
    n = rate_per_frame * n_frames
    return _ev(rng.integers(0, n_frames * F, n), rng.integers(0, 320, n), rng.integers(0, 320, n))


def _cat(*parts):
    ev = np.concatenate(parts)
    return ev[np.argsort(ev["t"], kind="stable")]


def _write(path, ev):
    rd.write_river_bin(path, ev)
    return path


@pytest.fixture(autouse=True)
def _fresh_cache():
    al.clear_cache()
    yield
    al.clear_cache()


def test_box_sum_matches_brute_force():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 5, (2, 12, 15))
    for kh, kw in ((3, 3), (4, 4), (9, 3), (3, 9)):
        got = al.box_sum(a, kh, kw)
        for y in range(12):
            for x in range(15):
                ref = a[:, max(0, y - kh // 2):y + kh - kh // 2, max(0, x - kw // 2):x + kw - kw // 2].sum((1, 2))
                assert (got[:, y, x] == ref).all()


def test_calibration_masks_hot_pixels_and_sets_threshold_above_noise(tmp_path):
    rng = np.random.default_rng(1)
    n = 300
    hot = _ev(np.arange(0, n * F, F // 2), np.full(2 * n, 17), np.full(2 * n, 33))   # fires every frame
    _write(tmp_path / "s" / "none" / "empty_000.bin", _cat(_noise(rng, n, 200), hot))
    cal = al.calibrate(tmp_path / "s", CFG, LAB)
    assert cal.hot[33, 17] and cal.hot.sum() < 5
    assert cal.min_blob >= LAB.min_blob
    assert "1 empty take" in cal.source


def test_without_empty_take_the_threshold_is_the_floor(tmp_path):
    rng = np.random.default_rng(2)
    _write(tmp_path / "s" / "none" / "walk.bin", _noise(rng, 50))
    cal = al.calibrate(tmp_path / "s", CFG, LAB)
    assert cal.min_blob == LAB.min_blob and "no empty take" in cal.source


def test_tracker_follows_the_ball_not_a_bigger_edge():
    """Ball on the dog's left, a person's edge with three times the events on the right:
    the median says right half, the tracker says left."""
    rng = np.random.default_rng(3)
    n = 60
    ev = _cat(_blob(rng, range(n), lambda f: 40 + f), _edge(rng, range(n), 290), _noise(rng, n))
    tr = al.track_target(ev, CFG, LAB, 0, n)
    assert (rd.bin_labels(tr.x, CFG, LAB) == 1).mean() > 0.9
    med = rd.bin_labels(rd.centroid_track(ev, CFG, LAB, 0, n), CFG, LAB)
    assert (med == 1).mean() < 0.5


def test_gate_keeps_the_track_when_a_second_blob_appears():
    rng = np.random.default_rng(4)
    n = 60
    ball = _blob(rng, range(n), lambda f: 60 + 2 * f, n=30)
    other = _blob(rng, range(30, n), lambda f: 280, y=60, n=60)          # stronger, far away
    tr = al.track_target(_cat(ball, other), CFG, LAB, 0, n)
    assert np.nanmax(np.abs(tr.x[30:] * 320 - (60 + 2 * np.arange(30, n)))) < 12


def test_busy_cells_are_masked_and_short_bursts_dropped():
    rng = np.random.default_rng(5)
    n = 400
    flicker = _blob(rng, range(n), lambda f: 250, y=40, n=30, r=6)       # never moves
    burst = _blob(rng, range(100, 102), lambda f: 40, n=60)             # 2 frames: noise
    ball = _blob(rng, range(200, 260), lambda f: 100 + f // 4)
    tr = al.track_target(_cat(flicker, burst, ball), CFG, LAB, 0, n)
    det = ~np.isnan(tr.x)
    assert not det[:190].any() and not det[270:].any()
    assert det[205:255].all()


def test_overrides_drop_trim_zone_and_ignore(tmp_path):
    rng = np.random.default_rng(6)
    n = 100
    s = tmp_path / "s"
    ball = lambda: _cat(_blob(rng, range(n), lambda f: 40), _noise(rng, n))   # dog's left
    for name in ("a", "b", "c", "d"):
        _write(s / "ball" / f"{name}.bin", ball())
    (s / "overrides.json").write_text(json.dumps({
        "*": {"ignore": [[300, 300, 320, 320]]},
        "ball/a.bin": {"drop": True},
        "ball/b.bin": {"trim": [0.2, 0.5]},
        "ball/c.bin": {"zone": "right"},
        "ball/d.bin": {"ignore": [[0, 0, 100, 320]]},
    }))
    load = lambda name: rd.load_recording(s / "ball" / name, "s", "ball", CFG, LAB, ("ball",))
    assert load("a.bin") is None
    b = load("b.bin")
    assert len(b.counts) == len(b.labels) == 30 and b.trim == (20, 50)
    assert (load("c.bin").labels[10:90] == 3).all()
    d = load("d.bin")
    assert (d.labels == 0).all() and len(d.overrides["ignore"]) == 2


def test_unknown_override_keys_are_rejected(tmp_path):
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "overrides.json").write_text(json.dumps({"ball/x.bin": {"skip": True}}))
    with pytest.raises(ValueError, match="unknown keys"):
        al.read_overrides(tmp_path / "s", LAB)


def test_auto_labels_agree_with_synthetic_ground_truth(tmp_path):
    from reflex.river import synth
    synth.write_dataset(tmp_path, sessions=1, per_kind=3, cfg=synth.SynthConfig(duration_s=4.0), seed=7)
    agree, total = 0, 0
    for path, session, kind in rd.find_recordings(tmp_path):
        if kind != "ball":
            continue
        auto = rd.load_recording(path, session, kind, CFG, LAB, ("ball",), use_gt=False)
        truth = rd.load_recording(path, session, kind, CFG, LAB, ("ball",))
        assert auto.source == "blob" and truth.source == "gt"
        agree += int((auto.labels == truth.labels).sum())
        total += len(truth.labels)
    assert agree / total > 0.8


# --------------------------------------------------------------------------- moving camera (drone)

def _tilt(n, amp=60.0, period=100):
    """Image offset in px of everything static in the world while the camera tilts."""
    return amp * np.sin(2 * np.pi * np.arange(n) / period)


def _tilting_scene(rng, n, corners=True, edges=(60, 110, 170, 230, 280)):
    """Background seen by a camera tilting up and down: long horizontal edges and compact
    corners, all moving with the tilt, firing in proportion to the camera's speed."""
    off = _tilt(n)
    speed = np.abs(np.gradient(off))                     # px per frame, up to ~3.8
    t, x, y = [], [], []
    for f in range(n):
        k = int(40 * speed[f])
        for y0 in edges:
            t += list(f * F + rng.integers(0, F, k))
            x += list(rng.integers(0, 320, k))
            y += list(np.full(k, y0 + off[f]) + rng.integers(-1, 2, k))
        if corners:
            for cx, cy in ((260, 90), (60, 200), (200, 250)):
                k2 = int(25 * speed[f])
                t += list(f * F + rng.integers(0, F, k2))
                x += list(cx + rng.integers(-5, 6, k2))
                y += list(cy + off[f] + rng.integers(-5, 6, k2))
    ok = (np.array(y) >= 0) & (np.array(y) < 320)
    return _ev(np.array(t)[ok], np.array(x)[ok], np.array(y)[ok].astype(int))


def _drone(rng, n, x0=40.0, vx=1.2, y0=140.0, per_frame=25):
    """A small target flying sideways; in the image it also moves with the tilt."""
    xs, ys = x0 + vx * np.arange(n), y0 + _tilt(n)
    ev = _blob(rng, range(n), lambda f: int(xs[f]), n=per_frame, r=5)
    ev["y"] = np.clip(np.repeat(ys.astype(int), per_frame) + rng.integers(-5, 6, len(ev)), 0, 319)
    return _cat(ev), xs, ys


def test_scene_motion_finds_the_tilt():
    rng = np.random.default_rng(10)
    n = 200
    ev = _tilting_scene(rng, n)
    A = al.active_maps(ev, CFG, 0, n)
    m = al.scene_motion(A, LAB)
    off = _tilt(n) / 4                                   # cells
    moving = m.state == al.MOVING
    assert moving.sum() > 0.5 * n
    f = np.flatnonzero(moving)
    truth = off[f] - off[f - LAB.ego_lag]
    assert np.median(np.abs(m.shift[f, 0] - truth)) <= 1.0     # dy
    assert np.median(np.abs(m.shift[f, 1])) == 0                # no dx in a tilt


def test_ego_tracker_follows_the_drone_not_the_background():
    rng = np.random.default_rng(11)
    n = 200
    drone, xs, ys = _drone(rng, n)
    ev = _cat(_tilting_scene(rng, n), drone)
    tr = al.track_target_ego(ev, CFG, LAB, 0, n)
    det = ~np.isnan(tr.y)
    # Precision over recall: a slow drone (3 cells per 100 ms relative to the scene) is only
    # told apart from the background part of the time, but what is labelled is the drone.
    assert det.mean() > 0.25
    on_drone = np.hypot(tr.x[det] * 320 - xs[det], tr.y[det] * 320 - ys[det]) < 16
    assert on_drone.mean() > 0.9
    old = al.track_target(ev, CFG, LAB, 0, n)          # the ball tracker takes the corners too
    od = ~np.isnan(old.y)
    assert (np.hypot(old.x[od] * 320 - xs[od], old.y[od] * 320 - ys[od]) < 16).mean() < 0.8


def test_camera_motion_alone_gives_no_target():
    rng = np.random.default_rng(12)
    n = 200
    ev = _tilting_scene(rng, n)
    assert (~np.isnan(al.track_target_ego(ev, CFG, LAB, 0, n).y)).mean() < 0.05
    assert (~np.isnan(al.track_target(ev, CFG, LAB, 0, n).y)).mean() > 0.3   # what step 6 fixes


def test_ego_tracker_with_a_still_camera_finds_the_target():
    rng = np.random.default_rng(13)
    n = 100
    ev = _cat(_blob(rng, range(n), lambda f: 60 + 2 * f, n=25, r=5), _noise(rng, n))
    tr = al.track_target_ego(ev, CFG, LAB, 0, n)
    assert (tr.camera == al.STILL).all()
    det = ~np.isnan(tr.x)
    assert det[5:-5].all()
    assert np.abs(tr.x[det] * 320 - (60 + 2 * np.flatnonzero(det))).max() < 10


def test_drone_takes_use_step_6_and_overrides_switch_it(tmp_path):
    rng = np.random.default_rng(14)
    n = 50
    s = tmp_path / "s"
    for kind in ("ball", "drone"):
        for name in ("a.bin", "b.bin"):
            _write(s / kind / name, _cat(_blob(rng, range(n), lambda f: 100 + f), _noise(rng, n)))
    (s / "overrides.json").write_text(json.dumps({"drone/b.bin": {"ego_motion": False},
                                                  "ball/b.bin": {"ego_motion": True}}))
    def camera(kind, name):
        rec = rd.load_recording(s / kind / name, "s", kind, CFG, LAB, (kind,))
        return rec.track.camera
    assert camera("drone", "a.bin") is not None and camera("ball", "a.bin") is None
    assert camera("drone", "b.bin") is None and camera("ball", "b.bin") is not None


def test_ego_motion_override_must_be_a_bool(tmp_path):
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "overrides.json").write_text(json.dumps({"*": {"ego_motion": "yes"}}))
    with pytest.raises(ValueError, match="ego_motion"):
        al.read_overrides(tmp_path / "s", LAB)
