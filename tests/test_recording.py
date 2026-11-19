import numpy as np

from reflex.data.recording import DISCARD, ClipRecorder, build_dataset, label_clip


def test_threat_clip_labels_follow_the_hit_mark():
    t = np.arange(0, 2_000_000, 20_000)
    y = label_clip(t, "left", t_hit_us=1_500_000, threat_ms=400, guard_ms=300)
    assert (y[t < 800_000] == 0).all()                                   # ball still far: none
    assert (y[(t >= 800_000) & (t < 1_100_000)] == DISCARD).all()        # ambiguous: left out
    assert (y[(t >= 1_100_000) & (t <= 1_500_000)] == 1).all()           # threat
    assert (y[t > 1_500_000] == DISCARD).all()                           # after arrival


def test_negative_clip_is_all_none():
    assert (label_clip(np.arange(0, 1_000_000, 20_000), "none", -1) == 0).all()


def _record(rec, label, n=60, hit_at=None):
    rec.start(label)
    for i in range(n):
        rec.add(np.full((10, 2, 45, 80), i % 3, dtype=np.uint8), i * 20_000)
    return rec.stop(hit_at)


def test_record_undo_and_build(tmp_path):
    rec = ClipRecorder(tmp_path / "real" / "s1", {"window_ms": 50.0, "n_bins": 10})
    _record(rec, "left", hit_at=1_000_000)
    bad = _record(rec, "right", hit_at=1_000_000)
    assert rec.undo() == bad and not bad.exists()
    _record(rec, "none")
    assert rec.counts() == {"none": 1, "left": 1, "center": 0, "right": 0}
    rec2 = ClipRecorder(tmp_path / "real" / "s2", {"window_ms": 50.0, "n_bins": 10})
    _record(rec2, "center", hit_at=900_000)

    out = tmp_path / "ds"
    info = build_dataset([tmp_path / "real" / "s1", tmp_path / "real" / "s2"], out)
    assert info["clips"] == 3 and info["sessions"] == ["s1", "s2"]
    from reflex.train import load_shards
    data, meta = load_shards(out)
    assert data["x"].shape[1:] == (10, 2, 45, 80) and set(np.unique(data["y"])) == {0, 1, 2}
    assert len(np.unique(data["episode"])) == 3 and meta["sessions"] == ["s1", "s2"]


def test_numbering_continues_in_an_existing_session(tmp_path):
    _record(ClipRecorder(tmp_path / "s"), "none")
    again = ClipRecorder(tmp_path / "s")
    assert _record(again, "none").name == "clip_001.npz"
