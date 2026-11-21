import json
from pathlib import Path

import numpy as np
import pytest

from reflex.runtime.river_net import RiverNet

BALL_DIR = Path("models/ball")


def one_layer(tmp_path, weight, bias, threshold=4096, leak=0):
    """1x4x4 input -> one 1x1 ConvLIF with a single weight: easy to compute by hand."""
    model = {"input": {"channels": 1, "height": 4, "width": 4},
             "nodes": [{"node_index": "a", "node_type": "ConvLIF2d", "upscaling": 1, "channels": 1,
                        "kernel_size": 1, "stride": 1, "padding": [0, 0, 0, 0]}],
             "edges": [{"source": "input", "target": "a"}], "outputs": ["a"]}
    params = {"weights_0": [weight], "bias_0": [bias], "threshold_0": threshold, "leak_0": leak}
    (tmp_path / "m.json").write_text(json.dumps(model))
    (tmp_path / "p.json").write_text(json.dumps(params))
    return RiverNet(tmp_path / "m.json", tmp_path / "p.json")


def test_integer_lif_matches_hand_computation(tmp_path):
    # weight 64 = 0.5: current 32 * 64 = 2048 per step; threshold 4096 is reached every 2nd step.
    net = one_layer(tmp_path, weight=64, bias=0)
    frame = np.zeros((1, 4, 4), np.float32)
    frame[0, 1, 2] = 1
    counts = net.run(frame, time_steps=4)
    assert counts[0, 1, 2] == 2 and counts.sum() == 2


def test_leak_and_bias(tmp_path):
    # weight 127: 4064 per step, just under 4096: fires on the 2nd of 3 steps only.
    # Bias 1 adds 32 -> 4096 per step: fires on every step.
    assert one_layer(tmp_path, 127, 0).run(np.ones((1, 4, 4), np.float32), 3).sum() == 16 * 1
    assert one_layer(tmp_path, 127, 1).run(np.ones((1, 4, 4), np.float32), 3).sum() == 16 * 3
    # Leak of half the potential per step: 2048, 3072, 3584 ... stays below 4096, no spike.
    assert one_layer(tmp_path, 64, 0, leak=2048).run(np.ones((1, 4, 4), np.float32), 3).sum() == 0


@pytest.mark.skipif(not (BALL_DIR / "ball_readout.json").exists(), reason="ball model not present")
def test_ball_detector_finds_a_ball_like_ring_and_ignores_an_empty_frame():
    from reflex.runtime.detectors import BallDetector

    det = BallDetector()
    empty = np.zeros((1, 2, 180, 320), np.uint8)
    assert det.predict(empty).label == "none"
    win = empty.copy()
    yy, xx = np.mgrid[:180, :320]
    win[0, 1] = np.abs(np.hypot(xx - 270, yy - 120) - 10) < 1.5      # ring in the right third
    d = det.predict(win)
    assert d.label == "right", det.last
    assert abs(det.last["x"] - 270) < 12 and abs(det.last["y"] - 120) < 12
