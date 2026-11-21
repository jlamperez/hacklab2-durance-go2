"""Turn a riverops export (model JSON + parameters JSON) into the files SPLEAT needs on the KV260.

    pixi run spleat-bundle MODEL.json PARAMETERS.json [--out out/spleat/NAME]

Inputs are the two files a riverops training script writes:
  - MODEL.json: `riverops.export.export_for_river(...)`, the River Model Graph.
  - PARAMETERS.json: `layer.export_params()` per spiking layer, keys `weights_i`,
    `bias_i`, `threshold_i`, `leak_i`.

Outputs, in one folder:
  - weights.txt              -> /root/weights.txt on the board
  - config.yaml              -> /root/.config/spleat-server/config.yaml
  - metavision_camera.json   -> /root/metavision_camera.json
  - README.md, manifest.json: where every value comes from, the checks that passed,
    input and output checksums, and what is still unknown.

Nothing here invents a number. Every value is either copied from the inputs, computed
from them (the computation is in this file), or a default copied from Durance's
documentation, and the manifest says which. The bitstream (spleat.bit.bin, spleat.dtbo)
is NOT produced here: it is generated per model on riversdk.dev from MODEL.json.

Sources referred to below:
  DOC_DEPLOY = Durance's "Deploying a new model on the Kria" page (deploy_a_new_model.html)
  DOC_NOTEBOOK = Durance's motion-buddy notebook, section "Export the model parameters"
               (train_motion_buddy.html)
  RIVEROPS = riverops 0.2.0 wheel, riverops/export/backends/river_model_graph.py and
             riverops/nn/spiking_conv.py (export_params)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# Fixed-point formats riverops exports with (DOC_NOTEBOOK table; RIVEROPS export_params).
WEIGHT_BITS, WEIGHT_FRAC = 8, 7          # weights and biases: Q1.7
POTENTIAL_BITS, POTENTIAL_FRAC = 16, 12  # threshold and leak: Q4.12

# DOC_DEPLOY, "Server configuration file".
SERVER_DEFAULTS = {"host": "0.0.0.0", "port": 3333, "weights_file": "/root/weights.txt",
                   "max_concurrency": 40}
# DOC_DEPLOY, "Despot configuration file" (default content).
CAMERA_DEFAULTS = {
    "frequency": 100.0,
    "noise_filtering_factor": 0.5,
    "biases": {"bias_diff": 51, "bias_diff_off": 40, "bias_diff_on": 40, "bias_fo": 29,
               "bias_hpf": 0, "bias_refr": 82},
}
SENSOR_SIDE = 320                         # DOC_DEPLOY: "The native camera definition is 320x320"
DOWNSAMPLING_ALLOWED = (1, 2, 4, 8, 16, 32, 64)  # DOC_DEPLOY


class BundleError(ValueError):
    """The two files do not describe one consistent, deployable network."""


def _int_range(bits: int) -> tuple[int, int]:
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


# ---------------------------------------------------------------- graph


def graph_shapes(model: dict) -> dict:
    """Per node: input channels, input and output (C, H, W). Checks the graph as it goes.

    Shape rule (PyTorch conv2d after nearest upsampling by `upscaling`):
      out = (in * upscaling + pad_a + pad_b - kernel) // stride + 1
    `padding` is [w, w, h, h]: RIVEROPS `_adapt_padding` turns PyTorch's (pv, ph) into
    (ph, ph, pv, pv).
    Several incoming edges are added, so they must agree in shape; the weight count
    check in `check_parameters` confirms the addition (a concatenation would need more
    weights).
    """
    inp = model["input"]
    shapes = {"input": {"out": (inp["channels"], inp["height"], inp["width"])}}
    sources: dict[str, list[str]] = {}
    for e in model["edges"]:
        sources.setdefault(e["target"], []).append(e["source"])

    for node in model["nodes"]:
        name = node["node_index"]
        if node["node_type"] != "ConvLIF2d":
            raise BundleError(f"{name}: node type {node['node_type']} is not handled here")
        srcs = sources.get(name, [])
        if not srcs:
            raise BundleError(f"{name}: no incoming edge")
        missing = [s for s in srcs if s not in shapes]
        if missing:
            raise BundleError(f"{name}: input {missing} is defined later; nodes must be in order")
        in_shapes = {shapes[s]["out"] for s in srcs}
        if len(in_shapes) != 1:
            raise BundleError(f"{name}: added inputs disagree in shape: "
                              + ", ".join(f"{s}={shapes[s]['out']}" for s in srcs))
        c, h, w = in_shapes.pop()
        k, s, up = node["kernel_size"], node["stride"], node["upscaling"]
        pw0, pw1, ph0, ph1 = node["padding"]
        oh = (h * up + ph0 + ph1 - k) // s + 1
        ow = (w * up + pw0 + pw1 - k) // s + 1
        if oh <= 0 or ow <= 0:
            raise BundleError(f"{name}: output size {oh}x{ow}")
        shapes[name] = {"sources": srcs, "in": (c, h, w), "out": (node["channels"], oh, ow)}

    if len(model["outputs"]) != 1:
        raise BundleError(f"config.yaml holds one output, the model has {model['outputs']}")
    if model["outputs"][0] not in shapes:
        raise BundleError(f"output {model['outputs'][0]} is not a node")
    return shapes


# ---------------------------------------------------------------- parameters


def check_parameters(model: dict, params: dict, shapes: dict) -> list[dict]:
    """Layer table. Parameter index i belongs to the i-th node of MODEL.json.

    That mapping is what riverops training scripts do (they number layers in module
    order, which is also the export order). Counts below catch most mismatches, but two
    layers of identical shape could be swapped without any count changing.
    """
    nodes = model["nodes"]
    n = len(nodes)
    expected = ([f"weights_{i}" for i in range(n)] + [f"bias_{i}" for i in range(n)]
                + [k for i in range(n) for k in (f"threshold_{i}", f"leak_{i}")])
    if list(params) != expected:
        extra = sorted(set(params) - set(expected))
        absent = sorted(set(expected) - set(params))
        raise BundleError(
            "parameter keys are not the documented layout (all weights, all biases, then "
            f"threshold/leak pairs) for {n} layers. missing={absent} unexpected={extra}"
            + ("" if extra or absent else " (same keys, different order)"))

    wlo, whi = _int_range(WEIGHT_BITS)
    plo, phi = _int_range(POTENTIAL_BITS)
    layers = []
    for i, node in enumerate(nodes):
        name = node["node_index"]
        c_in = shapes[name]["in"][0]
        c_out, k = node["channels"], node["kernel_size"]
        w, b = params[f"weights_{i}"], params[f"bias_{i}"]
        thr, leak = params[f"threshold_{i}"], params[f"leak_{i}"]
        if len(w) != c_out * c_in * k * k:
            raise BundleError(f"weights_{i} ({name}): {len(w)} values, expected "
                              f"{c_out}*{c_in}*{k}*{k} = {c_out * c_in * k * k}")
        if len(b) != c_out:
            raise BundleError(f"bias_{i} ({name}): {len(b)} values, expected {c_out}")
        for key, vals, lo, hi in ((f"weights_{i}", w, wlo, whi), (f"bias_{i}", b, wlo, whi)):
            bad = [v for v in vals if not isinstance(v, int) or not lo <= v <= hi]
            if bad:
                raise BundleError(f"{key}: {len(bad)} values outside int{WEIGHT_BITS} "
                                  f"[{lo}, {hi}], e.g. {bad[:3]}")
        for key, v in ((f"threshold_{i}", thr), (f"leak_{i}", leak)):
            if not isinstance(v, int) or not plo <= v <= phi:
                raise BundleError(f"{key} = {v!r} is not an int{POTENTIAL_BITS}")
        if thr <= 0 or leak < 0:
            raise BundleError(f"layer {i}: threshold {thr} must be > 0 and leak {leak} >= 0")
        scale = 1 << POTENTIAL_FRAC
        layers.append({
            "index": i, "node": name, "sources": shapes[name]["sources"],
            "in": list(shapes[name]["in"]), "out": list(shapes[name]["out"]),
            "kernel": k, "stride": node["stride"], "upscaling": node["upscaling"],
            "weights": len(w), "weight_min": min(w), "weight_max": max(w),
            "biases": len(b), "threshold": thr, "threshold_real": thr / scale,
            "leak": leak, "leak_real": leak / scale,
        })
    return layers


def flatten(params: dict) -> list[int]:
    """weights.txt values: exactly the loop of DOC_NOTEBOOK (dict order, lists expanded)."""
    values: list[int] = []
    for value in params.values():
        values.extend(value if isinstance(value, list) else [value])
    return values


# ---------------------------------------------------------------- outputs


def server_config(model: dict, shapes: dict, scaling: float) -> dict:
    inp = model["input"]
    c, h, w = shapes[model["outputs"][0]]["out"]
    return {
        "application": dict(SERVER_DEFAULTS),
        "inputs": {"channels": inp["channels"], "height": inp["height"],
                   "width": inp["width"], "scaling": float(scaling)},
        "outputs": {"channels": c, "height": h, "width": w},
    }


def config_yaml(cfg: dict) -> str:
    """Same keys and nesting as the example in DOC_DEPLOY."""
    def val(v):
        return f'"{v}"' if isinstance(v, str) and not v.startswith("/") else str(v)
    lines = []
    for section, body in cfg.items():
        lines.append(f"{section}:")
        lines += [f"    {k}: {val(v)}" for k, v in body.items()]
    return "\n".join(lines) + "\n"


def camera_config(model: dict) -> dict:
    h, w = model["input"]["height"], model["input"]["width"]
    if h != w or SENSOR_SIDE % h or SENSOR_SIDE // h not in DOWNSAMPLING_ALLOWED:
        raise BundleError(f"input {h}x{w} is not the {SENSOR_SIDE}x{SENSOR_SIDE} sensor "
                          f"downsampled by one of {DOWNSAMPLING_ALLOWED}")
    cfg = dict(CAMERA_DEFAULTS)
    cfg["downsampling"] = SENSOR_SIDE // h
    return {k: cfg[k] for k in ("frequency", "downsampling", "noise_filtering_factor", "biases")}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_state() -> str:
    try:
        root = Path(__file__).resolve().parents[2]
        head = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", root, "status", "--porcelain", "--", "reflex/deploy"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return head + (" (reflex/deploy has uncommitted changes)" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown (not a git checkout)"


def build(model_path: Path, params_path: Path, out_dir: Path, scaling: float) -> dict:
    model = json.loads(model_path.read_text())
    params = json.loads(params_path.read_text())
    if scaling != float(1 << POTENTIAL_FRAC):
        raise BundleError(f"scaling must be 2**fraction_bits = {1 << POTENTIAL_FRAC} for "
                          f"Q{POTENTIAL_BITS - POTENTIAL_FRAC}.{POTENTIAL_FRAC} potentials (DOC_DEPLOY)")

    shapes = graph_shapes(model)
    layers = check_parameters(model, params, shapes)
    values = flatten(params)
    n_expected = sum(l["weights"] + l["biases"] + 2 for l in layers)
    assert len(values) == n_expected, (len(values), n_expected)

    server = server_config(model, shapes, scaling)
    camera = camera_config(model)

    out_dir.mkdir(parents=True, exist_ok=True)
    # DOC_DEPLOY: "the file must not have an empty lines (including at the end)!"
    (out_dir / "weights.txt").write_text("\n".join(map(str, values)))
    (out_dir / "config.yaml").write_text(config_yaml(server))
    (out_dir / "metavision_camera.json").write_text(json.dumps(camera, indent=2) + "\n")

    outputs = ("weights.txt", "config.yaml", "metavision_camera.json")
    manifest = {
        "generated_by": "reflex/deploy/spleat_bundle.py",
        "repo_commit": _git_state(),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": f"pixi run spleat-bundle {model_path} {params_path} --out {out_dir}",
        "inputs": {str(model_path): _sha256(model_path), str(params_path): _sha256(params_path)},
        "outputs": {name: _sha256(out_dir / name) for name in outputs},
        "network": {
            "input": list(shapes["input"]["out"]),
            "output_node": model["outputs"][0],
            "output": list(shapes[model["outputs"][0]]["out"]),
            "layers": len(layers),
            "parameter_values": len(values),
        },
        "layers": layers,
        "server_config": server,
        "camera_config": camera,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out_dir / "README.md").write_text(_readme(manifest))
    return manifest


def _readme(m: dict) -> str:
    net, srv, cam = m["network"], m["server_config"], m["camera_config"]
    (model_in, model_sha), (params_in, params_sha) = m["inputs"].items()
    rows = "\n".join(
        f"| {l['index']} | {l['node']} | {'+'.join(l['sources'])} | {'x'.join(map(str, l['in']))} "
        f"| {l['upscaling']} | {l['kernel']}/{l['stride']} | {'x'.join(map(str, l['out']))} "
        f"| {l['weights']} | [{l['weight_min']}, {l['weight_max']}] | {l['threshold']} | {l['leak']} |"
        for l in m["layers"])
    thr = sorted({l["threshold"] for l in m["layers"]})
    leak = sorted({l["leak"] for l in m["layers"]})
    sums = "\n".join(f"| `{k}` | `{v}` |" for k, v in {**m["inputs"], **m["outputs"]}.items())
    return f"""# SPLEAT bundle

Generated by `{m['generated_by']}` at repo commit {m['repo_commit']}, {m['generated_at_utc']}.
Regenerate with:

```bash
{m['command']}
```

Inputs: `{model_in}` (River Model Graph) and `{params_in}` (fixed-point parameters).
Not in this folder: the bitstream (`spleat.bit.bin`, `spleat.dtbo`). It is generated per
model on riversdk.dev from `{model_in}` (board KV260, Prophesee IP GenX320).

## Files and where they go on the board

| File | Destination | Content |
|---|---|---|
| `weights.txt` | `/root/weights.txt` | {net['parameter_values']} integers, one per line, no trailing newline |
| `config.yaml` | `/root/.config/spleat-server/config.yaml` | server configuration |
| `metavision_camera.json` | `/root/metavision_camera.json` | camera and Despot configuration |
| `manifest.json` | stays here | everything in this README, machine-readable |

## Where every value comes from

**weights.txt**: the values of `{params_in}` in file order, lists expanded, joined by
newlines. That is the loop of Durance's motion-buddy notebook ("Export the model
parameters"), which produces the layout the deploy page asks for: all weights, all
biases, then threshold/leak pairs per layer. The key order was checked against that
layout before writing. No value is changed.

**config.yaml**

| Key | Value | Source |
|---|---|---|
| `application.*` | host {srv['application']['host']}, port {srv['application']['port']}, weights_file {srv['application']['weights_file']}, max_concurrency {srv['application']['max_concurrency']} | defaults on Durance's deploy page |
| `inputs.channels/height/width` | {srv['inputs']['channels']} x {srv['inputs']['height']} x {srv['inputs']['width']} | `input` of `{model_in}` |
| `inputs.scaling` | {srv['inputs']['scaling']} | deploy page: 2**fraction_bits of the potentials; riverops exports them as Q{POTENTIAL_BITS - POTENTIAL_FRAC}.{POTENTIAL_FRAC} |
| `outputs.channels/height/width` | {srv['outputs']['channels']} x {srv['outputs']['height']} x {srv['outputs']['width']} | computed: shape of `{net['output_node']}` propagated through the graph (layer table below) |

The nesting follows the example on the deploy page. Compare it with the file already on
the board before replacing it.

**metavision_camera.json**

| Key | Value | Source |
|---|---|---|
| `downsampling` | {cam['downsampling']} | computed: {SENSOR_SIDE} (native sensor side, deploy page) / {srv['inputs']['width']} (model input) |
| `frequency` | {cam['frequency']} | deploy page default. **Not checked against how the model was trained** |
| `noise_filtering_factor` | {cam['noise_filtering_factor']} | deploy page default |
| `biases` | {cam['biases']} | deploy page default (sensor biases) |

## Checks that passed

- Every node is a `ConvLIF2d`, nodes are in order, and inputs added into one node have
  the same shape.
- Parameter keys are exactly `weights_0..{net['layers'] - 1}`, `bias_0..{net['layers'] - 1}`, then
  `threshold_i, leak_i` pairs, one per node.
- Per layer, weight count = out_channels x in_channels x kernel x kernel and bias count =
  out_channels. Where two edges meet, this confirms they are added, not concatenated.
- Weights and biases fit int{WEIGHT_BITS} (Q1.{WEIGHT_FRAC}); thresholds and leaks fit
  int{POTENTIAL_BITS} (Q{POTENTIAL_BITS - POTENTIAL_FRAC}.{POTENTIAL_FRAC}).
- Thresholds: {thr} (= {', '.join(str(t / (1 << POTENTIAL_FRAC)) for t in thr)}).
  Leaks: {leak} (= {', '.join(f'{v / (1 << POTENTIAL_FRAC):.4f}' for v in leak)} of the potential per step).

## Layers

Parameter index i is taken to be the i-th node of the model file.

| i | node | from | in (CxHxW) | up | k/s | out (CxHxW) | weights | range | threshold | leak |
|---|---|---|---|---|---|---|---|---|---|---|
{rows}

## Not verifiable from these two files

- **Layer order.** Parameter index i is matched to the i-th node, as riverops training
  scripts number them. Counts catch most mistakes, but two layers of the same shape
  could be swapped without any check failing.
- **Neuron reset mode.** The model graph does not record it (riverops docs).
- **What the output channels mean** and how to decode them: that is in the training
  script, not in the export.
- **Time steps and frame rate.** How many steps the server runs per frame, whether the
  neuron state is reset between frames, and the `frequency` the model expects.
- **The server config format** on the real board (the example comes from the docs).

## Checksums (sha256)

| File | sha256 |
|---|---|
{sums}
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model", type=Path, help="River Model Graph JSON (export_for_river)")
    ap.add_argument("parameters", type=Path, help="parameters JSON (export_params per layer)")
    ap.add_argument("--out", type=Path, help="output folder (default: out/spleat/<model stem>)")
    ap.add_argument("--scaling", type=float, default=float(1 << POTENTIAL_FRAC),
                    help="inputs.scaling, must be 2**fraction_bits of the potentials")
    args = ap.parse_args(argv)
    out = args.out or Path("out/spleat") / args.model.stem
    try:
        m = build(args.model, args.parameters, out, args.scaling)
    except BundleError as e:
        ap.exit(1, f"error: {e}\n")
    net = m["network"]
    print(f"{net['layers']} layers, input {'x'.join(map(str, net['input']))}, "
          f"output {'x'.join(map(str, net['output']))}, {net['parameter_values']} values")
    print(f"wrote {out}/: weights.txt config.yaml metavision_camera.json README.md manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
