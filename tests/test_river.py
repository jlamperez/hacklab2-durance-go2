import numpy as np
import pytest

from reflex.river import data as rd
from reflex.river import synth

CFG = rd.FrameConfig()          # 320 sensor, 10 ms frames, downsample 4 -> 80x80
LAB = rd.LabelConfig(n_bins=3, min_events=5)


def _events(t, x, y, p=None):
    ev = np.zeros(len(t), dtype=rd.EVENT_DTYPE)
    ev["t"], ev["x"], ev["y"] = t, x, y
    ev["p"] = 1 if p is None else p
    return ev


def test_bin_roundtrip(tmp_path):
    ev = _events([0, 5, 10_000], [1, 300, 319], [2, 100, 0], [0, 1, 1])
    rd.write_river_bin(tmp_path / "a.bin", ev)
    back = rd.read_river_bin(tmp_path / "a.bin")
    assert back.dtype == rd.EVENT_DTYPE
    for k in ("t", "x", "y", "p"):
        assert (back[k] == ev[k]).all()


def test_bad_header_is_rejected(tmp_path):
    (tmp_path / "b.bin").write_bytes(b"\x00\x00\x00\x05abc")
    with pytest.raises(ValueError):
        rd.read_river_bin(tmp_path / "b.bin")


def test_frame_counts_bins_time_polarity_and_space():
    ev = _events([0, 1, 9_999, 10_000, 25_000], [0, 3, 4, 319, 160], [0, 0, 0, 319, 160], [0, 0, 1, 1, 0])
    c = rd.frame_counts(ev, CFG)
    assert c.shape == (3, 2, 80, 80)
    assert c[0, 0, 0, 0] == 2           # x 0 and 3 fall in the same 4x4 block
    assert c[0, 1, 0, 1] == 1
    assert c[1, 1, 79, 79] == 1
    assert c[2, 0, 40, 40] == 1
    assert c.sum() == len(ev)
    assert rd.normalize(c, CFG)[0, 0, 0, 0] == pytest.approx(2 / 16)


def test_flip_x_mirrors_frames_and_labels():
    ev = _events(np.zeros(10, int), np.full(10, 10), np.full(10, 10))
    flipped = rd.FrameConfig(flip_x=True)
    assert rd.frame_counts(ev, flipped)[0, 1, 2, 77] == 10
    xn = rd.centroid_track(ev, flipped, LAB, 0, 1)
    assert rd.bin_labels(xn, CFG, LAB)[0] == 1          # image left = dog's left
    assert rd.bin_labels(xn, flipped, LAB)[0] == 3      # mirrored mount: dog's right


def test_centroid_labels_zones_and_none():
    t, x = [], []
    for f, cx in enumerate([20, 160, 300, None]):       # left, center, right, nothing
        if cx is not None:
            t += [f * 10_000 + i for i in range(20)]
            x += [cx] * 20
    t += [35_000]; x += [5]                             # one stray event in frame 3
    ev = _events(t, x, np.full(len(t), 50))
    labels = rd.bin_labels(rd.centroid_track(ev, CFG, LAB, 0, 4), CFG, LAB)
    assert labels.tolist() == [1, 2, 3, 0]


def test_fill_gaps_bridges_short_gaps_only():
    nan = np.nan
    x = np.array([nan, 0.2, nan, nan, 0.5, nan, nan, nan, nan, 0.9, nan])
    out = rd.fill_gaps(x, max_gap=3)
    assert np.allclose(out[1:5], [0.2, 0.3, 0.4, 0.5])
    assert np.isnan(out[[0, 5, 6, 7, 8, 10]]).all()      # long gap and the open ends stay none


def test_h5_recordings_load(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "s1" / "ball" / "take.h5"
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as f:
        f.attrs["format"] = "reflex-events-v1"
        g = f.create_group("events")
        g["t"] = np.arange(50, dtype=np.int64) * 200
        g["x"] = np.full(50, 300, np.uint16)
        g["y"] = np.full(50, 10, np.uint16)
        g["p"] = np.ones(50, np.uint8)
    (found,) = rd.find_recordings(tmp_path)
    rec = rd.load_recording(*found, CFG, rd.LabelConfig(n_bins=3, min_events=5, method="median"), ("ball",))
    assert rec.counts.sum() == 50 and (rec.labels == 3).all()


def test_mirror_label():
    assert [rd.mirror_label(k, 3) for k in range(4)] == [0, 3, 2, 1]


def test_synthetic_recordings_load_with_ground_truth(tmp_path):
    cfg = synth.SynthConfig(duration_s=2.0)
    synth.write_dataset(tmp_path, sessions=1, per_kind=1, cfg=cfg, seed=1)
    found = rd.find_recordings(tmp_path)
    assert sorted(f"{k}/{p.name}" for p, _, k in found) == ["ball/ball_000.bin", "none/empty_000.bin",
                                                         "none/none_000.bin"]
    for path, session, kind in found:
        rec = rd.load_recording(path, session, kind, CFG, LAB, ("ball",))
        assert rec.counts.shape[1:] == (2, 80, 80)
        assert len(rec.labels) == len(rec.counts)
        if kind == "none":
            assert (rec.labels == 0).all()
        else:
            assert (rec.labels > 0).any()


def test_window_dataset_shapes(tmp_path):
    torch = pytest.importorskip("torch")
    synth.write_dataset(tmp_path, sessions=1, per_kind=1, cfg=synth.SynthConfig(duration_s=1.5), seed=2)
    recs = [rd.load_recording(p, s, k, CFG, LAB, ("ball",)) for p, s, k in rd.find_recordings(tmp_path)]
    win = rd.WindowConfig(frames=20, readout=10, stride=10)
    ds = rd.WindowDataset(recs, CFG, LAB, win, augment=True)
    x, y = ds[0]
    assert x.shape == (20, 2, 80, 80) and x.dtype == torch.float32
    assert 0 <= y < LAB.n_classes


def test_recalibrate_bn_matches_data_statistics():
    torch = pytest.importorskip("torch")
    from torch import nn
    from reflex.river.train import recalibrate_bn

    model = nn.Sequential(nn.Conv2d(1, 2, 1, bias=False), nn.BatchNorm2d(2))
    nn.init.ones_(model[0].weight)
    model.eval()                                           # restored afterwards
    batches = [(torch.full((4, 1, 3, 3), 5.0) + torch.randn(4, 1, 3, 3), 0) for _ in range(6)]
    recalibrate_bn(model, batches, torch.device("cpu"), batches=6)
    assert model[1].running_mean == pytest.approx(torch.full((2,), 5.0), abs=0.2)
    assert model[1].momentum == 0.1 and not model.training


def test_fold_for_board_rescale_keeps_spikes_and_fits_q1_7():
    torch = pytest.importorskip("torch")
    pytest.importorskip("riverops")
    from riverops.nn import FixedPointFormat, QuantConfig, fuse_model
    from reflex.river.model import TargetSNN
    from reflex.river.train import Q1_7_MAX, fold_for_board

    torch.manual_seed(0)
    model = TargetSNN(4, 40)
    with torch.no_grad():                                  # BN that would fold to |w| ~ 20
        model.net[0].bn.running_var.fill_(0.0025)
    model.eval()
    q = QuantConfig(weight_format=FixedPointFormat(total_bits=8, frac_bits=7),
                    potential_format=FixedPointFormat(total_bits=16, frac_bits=12),
                    enable_weight_quant=True, enable_potential_quant=True)
    hw, scales = fold_for_board(model, q)
    assert scales[0] > 1.0
    for layer in hw.net:
        assert float(layer.conv.weight.abs().max()) <= Q1_7_MAX + 1e-6
    x = torch.rand(2, 12, 2, 40, 40)
    float_rescaled = fuse_model(model)                     # same rescale, without fixed point
    for layer, s in zip(float_rescaled.net, scales):
        layer.conv.weight.data /= s
        layer.conv.bias.data /= s
        layer.lif.v_threshold /= s
    with torch.no_grad():
        assert torch.equal(fuse_model(model)(x), float_rescaled(x))


def test_board_weights_are_twos_complement():
    """weights.txt: weights and biases as unsigned two's complement, neurons as they are."""
    from reflex.river.train import board_weights_text

    params = {"weights_0": [-128, -16, -1, 0, 1, 127], "bias_0": [-2, 3],
              "threshold_0": 4096, "leak_0": 287}
    text = board_weights_text(params)
    assert text == "\n".join(["128", "240", "255", "0", "1", "127", "254", "3", "4096", "287"])
    with pytest.raises(ValueError):
        board_weights_text({"weights_0": [200]})
