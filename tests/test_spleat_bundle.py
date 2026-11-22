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
    for k, v in params.items():
        v = v if isinstance(v, list) else [v]
        expected.extend([x & 0xFF for x in v] if k.startswith(("weights_", "bias_")) else v)
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


def fake_firmware(tmp_path, model_sha):
    import io
    import tarfile
    fw = tmp_path / "fw"
    fw.mkdir()
    with tarfile.open(fw / "firmware-kv260.tar.gz", "w:gz") as tar:
        for name, data in (("spleat.bit.bin", b"bitstream"), ("spleat.dtbo", b"overlay")):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    (fw / "PROVENANCE.md").write_text(f"| Model file | `m.json`, sha256 `{model_sha}` |\n")
    return fw


def test_firmware_is_added_to_the_deploy_profile_when_built_from_this_model(tmp_path):
    import hashlib
    model = tiny_model()
    mp, pp = tmp_path / "m.json", tmp_path / "p.json"
    mp.write_text(json.dumps(model))
    pp.write_text(json.dumps(tiny_params(model)))
    fw = fake_firmware(tmp_path, hashlib.sha256(mp.read_bytes()).hexdigest())
    m = build(mp, pp, tmp_path / "out", 4096.0, firmware=fw)
    out = tmp_path / "out"
    assert (out / "spleat.bit.bin").read_bytes() == b"bitstream"
    profile = (out / "deploy.yaml").read_text()
    for name in ("weights.txt", "config.yaml", "metavision_camera.json", "spleat.bit.bin", "spleat.dtbo"):
        sha = hashlib.sha256((out / name).read_bytes()).hexdigest()
        assert f"  - source: {name}\n" in profile and f'sha256: "{sha}"' in profile
    assert "/usr/lib/firmware/xilinx/spleat/spleat.dtbo" in profile
    assert m["firmware"]["files"]["spleat.dtbo"] == hashlib.sha256(b"overlay").hexdigest()


def test_firmware_built_from_another_model_is_refused(tmp_path):
    model = tiny_model()
    mp, pp = tmp_path / "m.json", tmp_path / "p.json"
    mp.write_text(json.dumps(model))
    pp.write_text(json.dumps(tiny_params(model)))
    fw = fake_firmware(tmp_path, "0" * 64)
    with pytest.raises(BundleError, match="different architecture"):
        build(mp, pp, tmp_path / "out", 4096.0, firmware=fw)


def test_camera_frequency_follows_the_checkpoint_window(tmp_path):
    import torch
    model = tiny_model()
    mp, pp = tmp_path / "m.json", tmp_path / "p.json"
    mp.write_text(json.dumps(model))
    pp.write_text(json.dumps(tiny_params(model)))
    torch.save({"window_ms": 5.0}, tmp_path / "c.pt")
    build(mp, pp, tmp_path / "out", 4096.0, checkpoint=tmp_path / "c.pt")
    assert json.loads((tmp_path / "out" / "metavision_camera.json").read_text())["frequency"] == 200.0
