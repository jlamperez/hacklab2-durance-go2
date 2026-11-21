"""Event sources for the event-camera node: replay of recordings, live camera over TCP."""
import json
import socket
import struct
import time
from pathlib import Path

import numpy as np
import pytest

from reflex.data.windows import WindowConfig
from reflex.events.emu import EVENT_DTYPE
from reflex.events.sources import (RIVER_DTYPE, LiveSource, ReplaySource, RiverStreamDecoder, Take,
                                   find_takes, still_take)
from reflex.runtime.windower import StreamWindower

REPO = Path(__file__).resolve().parents[1]


def events(ts, x=10, y=20, p=1):
    ev = np.zeros(len(ts), dtype=EVENT_DTYPE)
    ev["t"], ev["x"], ev["y"], ev["p"] = ts, x, y, p
    return ev


def river_batch(ev):
    rec = np.empty(len(ev), dtype=RIVER_DTYPE)
    for k in ("t", "x", "y", "p"):
        rec[k] = ev[k]
    payload = rec.tobytes()
    return struct.pack("!I", len(payload)) + payload


# --------------------------------------------------------------------------- live protocol

def test_decoder_reads_batches_split_anywhere():
    a, b = events([1, 2, 3], x=5), events([4, 5], x=7)
    stream = river_batch(a) + river_batch(events([])) + river_batch(b)
    dec = RiverStreamDecoder()
    got = np.concatenate([dec.feed(stream[i:i + 1]) for i in range(len(stream))])
    assert got["t"].tolist() == [1, 2, 3, 4, 5]
    assert got["x"].tolist() == [5, 5, 5, 7, 7]


def test_decoder_rejects_a_foreign_stream():
    with pytest.raises(ValueError):
        RiverStreamDecoder().feed(struct.pack("!I", 14) + b"\0" * 14)


def test_live_windows_keep_moving_when_nothing_arrives():
    src = LiveSource(clock=lambda: 0, log=lambda s: None)
    assert src.t_end(0) is None
    src.ingest(events([1_000, 2_000]), arrival_us=50_000)     # 48 ms delay
    src.ingest(events([3_000]), arrival_us=13_000)            # 10 ms: the best seen
    assert src.t_end(13_000) == 3_000
    assert src.t_end(113_000) == 103_000                      # still scene: time goes on
    assert src.poll(0)["t"].tolist() == [1_000, 2_000, 3_000]
    src.ingest(events([2_500, 4_000]), arrival_us=20_000)     # 2.5 ms comes too late
    assert src.poll(0)["t"].tolist() == [4_000] and src.dropped_late == 1


def test_live_camera_restart_starts_over():
    src = LiveSource(clock=lambda: 0, log=lambda s: None)
    src.ingest(events([50_000_000]), arrival_us=0)
    src.poll(0)
    src.ingest(events([1_000]), arrival_us=10)
    assert src.epoch == 1 and src.poll(0)["t"].tolist() == [1_000]


def test_live_flip_mirrors_the_image():
    src = LiveSource(width=320, height=320, flip_x=True, log=lambda s: None)
    src.ingest(events([1], x=0, y=5), arrival_us=0)
    ev = src.poll(0)
    assert ev["x"][0] == 319 and ev["y"][0] == 5


def test_live_source_over_tcp():
    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    src = LiveSource(port=port, log=lambda s: None)
    src.start()
    try:
        for _ in range(50):
            try:
                conn = socket.create_connection(("127.0.0.1", port), timeout=1)
                break
            except OSError:
                time.sleep(0.05)
        with conn:
            data = river_batch(events([10, 20, 30])) + river_batch(events([40]))
            conn.sendall(data[:7])
            conn.sendall(data[7:])
            deadline = time.monotonic() + 3
            while src.received < 4 and time.monotonic() < deadline:
                time.sleep(0.02)
        assert src.poll(0)["t"].tolist() == [10, 20, 30, 40]
    finally:
        src.close()


# --------------------------------------------------------------------------- replay

def fake_reader(data):
    return lambda path: data[Path(path).name].copy()


def test_replay_plays_takes_back_to_back_in_real_time():
    data = {"a": events([0, 200_000, 400_000]), "b": events([5_000_000, 5_100_000])}
    src = ReplaySource([Take(Path("a"), "a"), Take(Path("b"), "b")], loop=False, gap_s=0.5,
                       lead_s=1.0, reader=fake_reader(data))
    assert len(src.poll(0)) == 0                     # lead: silence
    assert src.poll(1_000_000)["t"].tolist() == [1_000_000]
    assert src.current(1_100_000)[0].name == "a"
    assert src.poll(1_400_000)["t"].tolist() == [1_200_000, 1_400_000]
    assert src.current(1_500_000) is None            # the gap between takes
    # b starts 0.5 s after a ended, whatever its own clock said
    assert src.poll(2_000_000)["t"].tolist() == [1_900_000, 2_000_000]
    assert len(src.poll(9_000_000)) == 0 and src.finished


def test_replay_loops_and_time_only_moves_forward():
    data = {"a": events([0, 100_000])}
    src = ReplaySource([Take(Path("a"), "a")], gap_s=0.2, lead_s=0.0, reader=fake_reader(data))
    t = np.concatenate([src.poll(us) for us in range(0, 2_000_000, 20_000)])["t"]
    assert (np.diff(t) > 0).all() and len(t) == 2 * 7   # a pass takes 0.3 s: 7 complete passes
    assert src.loops == 6


def test_replay_speed_scales_time():
    data = {"a": events([0, 1_000_000])}
    src = ReplaySource([Take(Path("a"), "a")], speed=2.0, lead_s=0.0, loop=False, reader=fake_reader(data))
    assert len(src.poll(0)) == 1 and len(src.poll(500_000)) == 1


def test_replay_skips_empty_takes_and_stops_when_all_are_empty():
    data = {"a": events([])}
    src = ReplaySource([Take(Path("a"), "a")], lead_s=0.0, reader=fake_reader(data))
    assert len(src.poll(10_000_000)) == 0 and src.finished


def test_find_takes_follows_takes_csv_and_drops_and_excludes(tmp_path):
    s = tmp_path / "sess"
    for rel in ("none/take_000_none.h5", "ball/take_003_left.h5", "ball/take_004_left.h5",
                "none/take_002_none.h5", "ball/take_009_x.bin", "ball/take_009_x.h5", "ball/take_010.bin"):
        (s / rel).parent.mkdir(parents=True, exist_ok=True)
        (s / rel).write_bytes(b"")
    (s / "takes.csv").write_text("take,file\n0,take_000_none.h5\n2,take_002_none.h5\n"
                                 "3,take_003_left.h5\n4,take_004_left.h5\n")
    (s / "overrides.json").write_text(json.dumps({"ball/take_003_left.h5": {"drop": True}}))
    names = [t.name for t in find_takes([tmp_path], exclude=["*take_002*"], with_info=False)]
    assert names == ["sess/none/take_000_none.h5", "sess/ball/take_004_left.h5",
                     "sess/ball/take_009_x.h5", "sess/ball/take_010.bin"]


def test_still_take_is_the_sessions_empty_recording(tmp_path):
    s = tmp_path / "sess"
    for rel in ("none/empty_000.h5", "ball/take_007_center.h5"):
        (s / rel).parent.mkdir(parents=True, exist_ok=True)
        (s / rel).write_bytes(b"")
    ball = Take(s / "ball/take_007_center.h5", "take_007_center.h5")
    assert still_take(ball, with_info=False).path == s / "none/empty_000.h5"
    assert still_take(Take(s / "none/empty_000.h5", "x"), with_info=False) is None


# --------------------------------------------------------------------------- real recordings

SESSION = REPO / "data" / "river" / "sat_follow2"


@pytest.mark.skipif(not SESSION.exists(), reason="recordings not in this checkout")
def test_replay_of_a_real_session_gives_genx320_windows():
    pytest.importorskip("h5py")
    takes = find_takes([SESSION], exclude=["*take_002*"])
    assert takes[0].info["take"] == 0 and takes[0].info["label"] == "none"     # the still scene first
    src = ReplaySource(takes[:5], loop=False)
    cfg = WindowConfig(height=320, width=320)
    win = StreamWindower(cfg)
    n = 0
    for us in range(0, 40_000_000, 20_000):
        ev = src.poll(us)
        if len(ev):
            assert ev["x"].max() < 320 and ev["y"].max() < 320
        win.push(ev)
        w = win.window(src.t_end(us))
        assert w.shape == (10, 2, 80, 80)
        n += int(w.sum() > 0)
        if src.finished:
            break
    assert src.finished and n > 100
