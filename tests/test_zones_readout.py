import json
import struct

import numpy as np
import pytest

from reflex.runtime.spleat_client import cbor_encode, main
from reflex.runtime.zones_readout import ZonesReadout

NAMES = ["none", "left", "center", "right"]

# The board's answers in test B on 22 Nov (docs/fpga-zones3-2026-11-22.md, section 5.4).
BOARD_B = [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0],
           [1, 1, 1, 0], [1, 0, 0, 1], [0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0]]


def test_sums_over_the_last_readout_frames_and_takes_the_most():
    r = ZonesReadout(NAMES, readout=3)
    assert r.push([0, 2, 0, 0]).label == "left"
    r.push([0, 0, 1, 0])
    d = r.push([0, 0, 2, 0])
    assert d.label == "center" and list(d.counts) == [0, 2, 3, 0] and d.frames == 3
    d = r.push([0, 0, 0, 0])                 # the first frame leaves the window
    assert list(d.counts) == [0, 0, 3, 0]


def test_ties_go_to_none_and_margin_holds_zones_back():
    r = ZonesReadout(NAMES, readout=20)
    assert r.push([1, 1, 0, 0]).label == "none"          # tie, as torch's argmax in training
    r = ZonesReadout(NAMES, readout=20, margin=2)
    assert r.push([1, 2, 0, 0]).label == "none"
    assert r.push([0, 1, 0, 0]).label == "left"          # 3 >= 1 + 2


def test_wrong_number_of_values_is_rejected():
    with pytest.raises(ValueError):
        ZonesReadout(NAMES).push([1, 2, 3])


def test_from_model_dir_reads_names_and_readout(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"names": NAMES, "window": {"readout": 7}}))
    r = ZonesReadout.from_model_dir(tmp_path)
    assert r.names == NAMES and r.readout == 7


def test_spleat_read_decodes_saved_board_messages(tmp_path, capsys):
    (tmp_path / "meta.json").write_text(json.dumps({"names": NAMES, "window": {"readout": 20}}))
    (tmp_path / "config.yaml").write_text(
        "inputs:\n  channels: 2\n  height: 80\n  width: 80\n\noutputs:\n  channels: 4\n  height: 1\n  width: 1\n")
    with (tmp_path / "msgs.bin").open("wb") as f:
        for i, v in enumerate(BOARD_B):
            body = cbor_encode({"id": i, "payload": np.array(v, "<f4").tobytes(), "latency": 4000})
            f.write(struct.pack(">I", len(body)) + body)
    assert main(["--file", str(tmp_path / "msgs.bin"), "--zones-dir", str(tmp_path)]) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("#")]
    assert len(lines) == 10
    assert "-> NONE   [none 0 left 0 center 0 right 0] (window 1/20)" in lines[0]
    # after 10 frames: none 5, left 3, center 3, right 1 -> none
    assert "[none 5 left 3 center 3 right 1] (window 10/20)" in lines[-1]
    assert "-> NONE" in lines[-1]


BALL_NAMES = ["upper_left", "upper_middle", "upper_right", "lower_left", "lower_middle",
              "lower_right", "none"]


def test_none_last_ties_and_empty_frames_go_to_none():
    r = ZonesReadout(BALL_NAMES, readout=4)
    assert r.none_index == 6
    assert r.push([0] * 7).label == "none"                            # no spikes at all
    assert r.push([0, 0, 0, 1, 0, 0, 1]).label == "none"              # tie with none
    assert r.push([0, 0, 0, 1, 0, 0, 0]).label == "lower_left"        # 2 > 1


def test_from_model_dir_reads_a_riverops_readout_json(tmp_path):
    (tmp_path / "ball_zones_readout.json").write_text(json.dumps(
        {"time_steps": 4, "classes": BALL_NAMES, "zones": {"rows": 2, "columns": 3}}))
    r = ZonesReadout.from_model_dir(tmp_path)
    assert r.names == BALL_NAMES and r.readout == 4 and r.none_index == 6
    assert ZonesReadout.from_model_dir(tmp_path, readout=12).readout == 12


def test_reflex_label_maps_columns():
    from reflex.runtime.zones_readout import reflex_label
    assert [reflex_label(n) for n in BALL_NAMES] == ["left", "center", "right", "left", "center",
                                                     "right", "none"]
    assert [reflex_label(n) for n in NAMES] == NAMES
