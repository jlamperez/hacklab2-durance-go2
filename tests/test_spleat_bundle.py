import json

import pytest

from reflex.deploy.spleat_bundle import BundleError, build


def tiny_model():
    """input 1x20x20 -> a (10x10) -> b (5x5) -> c = up(b) + ... : a small U with one skip."""
    return {
        "input": {"channels": 1, "height": 20, "width": 20},
        "nodes": [
            {"node_index": "a", "node_type": "ConvLIF2d", "upscaling": 1, "channels": 4,
             "kernel_size": 4, "stride": 2, "padding": [1, 1, 1, 1]},
            {"node_index": "b", "node_type": "ConvLIF2d", "upscaling": 1, "channels": 4,
             "kernel_size": 4, "stride": 2, "padding": [1, 1, 1, 1]},
            {"node_index": "c", "node_type": "ConvLIF2d", "upscaling": 2, "channels": 4,
             "kernel_size": 3, "stride": 1, "padding": [1, 1, 1, 1]},
            {"node_index": "d", "node_type": "ConvLIF2d", "upscaling": 1, "channels": 3,
             "kernel_size": 1, "stride": 1, "padding": [0, 0, 0, 0]},
        ],
        "edges": [{"source": "input", "target": "a"}, {"source": "a", "target": "b"},
                  {"source": "b", "target": "c"}, {"source": "c", "target": "d"},
                  {"source": "a", "target": "d"}],
        "outputs": ["d"],
    }


def tiny_params(model):
    shapes_in = {"a": 1, "b": 4, "c": 4, "d": 4}
    p = {}
    for i, n in enumerate(model["nodes"]):
        count = n["channels"] * shapes_in[n["node_index"]] * n["kernel_size"] ** 2
        p[f"weights_{i}"] = [(j % 255) - 127 for j in range(count)]
    for i, n in enumerate(model["nodes"]):
        p[f"bias_{i}"] = [i] * n["channels"]
    for i, _ in enumerate(model["nodes"]):
        p[f"threshold_{i}"], p[f"leak_{i}"] = 4096, 287
    return p


def run(tmp_path, model, params):
    mp, pp = tmp_path / "m.json", tmp_path / "p.json"
    mp.write_text(json.dumps(model))
    pp.write_text(json.dumps(params))
    return build(mp, pp, tmp_path / "out", 4096.0), tmp_path / "out"


def test_bundle_matches_notebook_flattening_and_graph_shapes(tmp_path):
    model = tiny_model()
    params = tiny_params(model)
    m, out = run(tmp_path, model, params)
    expected = []
    for v in params.values():
        expected.extend(v if isinstance(v, list) else [v])
    text = (out / "weights.txt").read_text()
    assert not text.endswith("\n")
    assert list(map(int, text.split("\n"))) == expected
    assert m["network"]["output"] == [3, 10, 10]
    cfg = (out / "config.yaml").read_text()
    assert "    channels: 3\n    height: 10\n    width: 10\n" in cfg
    cam = json.loads((out / "metavision_camera.json").read_text())
    assert cam["downsampling"] == 16  # 320 / 20
    assert (out / "README.md").exists()


def test_wrong_weight_count_is_rejected(tmp_path):
    model = tiny_model()
    params = tiny_params(model)
    params["weights_3"] = params["weights_3"][:-1]
    with pytest.raises(BundleError, match="weights_3"):
        run(tmp_path, model, params)


def test_keys_out_of_layout_are_rejected(tmp_path):
    model = tiny_model()
    params = tiny_params(model)
    reordered = {k: params[k] for k in sorted(params)}
    with pytest.raises(BundleError, match="layout"):
        run(tmp_path, model, reordered)


def test_added_inputs_must_agree_in_shape(tmp_path):
    model = tiny_model()
    model["nodes"][2]["upscaling"] = 1  # c stays 5x5, a is 10x10: cannot be added into d
    with pytest.raises(BundleError, match="disagree"):
        run(tmp_path, model, tiny_params(model))
