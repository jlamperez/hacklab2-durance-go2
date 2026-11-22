import json

import numpy as np
import pytest

from reflex.runtime.ball_readout import BallReadout, decode, to_camera


def readout(no_ball_logit=float(np.log(80 * 80))):
    return BallReadout(
        time_steps=4, grid=80, cell_px=4.0, heatmap_channel=0,
        x_offset=(1, 5), y_offset=(5, 9),
        heatmap_logit_scale=16.0, offset_logit_scale=16.0, no_ball_logit=no_ball_logit,
    )


def counts_with_ball(row, col, x_bin, y_bin, t=4):
    counts = np.zeros((9, 80, 80))
    counts[0, row, col] = t
    counts[1 + x_bin, row, col] = t
    counts[5 + y_bin, row, col] = t
    return counts


def test_peak_cell_and_offset_bins_give_the_position():
    ball = decode(counts_with_ball(row=10, col=30, x_bin=3, y_bin=0), readout())
    assert ball.x == pytest.approx(30 * 4 + 3.5, abs=0.05)
    assert ball.y == pytest.approx(10 * 4 + 0.5, abs=0.05)
    assert ball.found and ball.p_ball > 0.99


def test_spikes_per_step_equal_their_counts():
    spikes = np.zeros((4, 9, 80, 80))
    spikes[:, 0, 40, 20] = 1
    spikes[1:, 2, 40, 20] = 1
    spikes[:2, 7, 40, 20] = 1
    assert decode(spikes, readout()) == decode(spikes.sum(axis=0), readout())


def test_silent_heatmap_is_no_ball():
    silent = np.zeros((9, 80, 80))
    assert decode(silent, readout()).p_ball == pytest.approx(0.5)
    assert not decode(silent, readout(no_ball_logit=float(np.log(80 * 80)) + 2)).found


def test_load_reads_the_exported_readout_file(tmp_path):
    path = tmp_path / "ball_small_readout.json"
    path.write_text(json.dumps({
        "time_steps": 4, "grid": 80, "cell_px": 4,
        "channels": {"heatmap": 0, "x_offset": [1, 5], "y_offset": [5, 9]},
        "offset_bins": 4, "heatmap_logit_scale": 16.0, "offset_logit_scale": 16.0,
        "no_ball_logit": float(np.log(80 * 80)), "ball_probability_threshold": 0.5,
    }))
    assert BallReadout.load(path) == readout()
    assert BallReadout.load(path).sensor == 320


def test_to_camera_undoes_the_fit_into_the_square_input():
    assert to_camera(160, 160, 320, 320) == (160, 160)
    assert to_camera(0, 160, 1280, 720) == (0, 360)
    assert to_camera(320, 320, 1280, 720) == (1280, 1000)
