"""Train, evaluate and export the target-zone SNN for RIVER.

    python -m reflex.river.train --data data/river_synth --out out/river/synth_v1
    python -m reflex.river.train --data data/river --val-sessions sat_pm --test-sessions sun_am \
        --init out/river/synth_v1/best.pt --out out/river/ball_v1

Adapted from Durance's Motion Buddy notebook: same layers, loss (MSE on spike counts),
optimiser, clipping and export. Differences: the task (none + horizontal zones instead of
gestures), clips cut from continuous recordings instead of one clip per recording, class
balancing, left/right mirroring as augmentation, and the export also writes the
spleat-server config for this model.

Writes into --out: best.pt, meta.json, metrics.json, model.json, parameters.json,
weights.txt, config.yaml, metavision_camera.json.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import (FrameConfig, LabelConfig, NONE_KIND, WindowConfig, WindowDataset,
                   find_recordings, load_recording)
from .model import ModelConfig, TargetSNN, count_parameters, readout_counts, spike_targets


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- data

def load_split(roots, targets, frame, lab, val_sessions, test_sessions):
    found = [f for r in roots for f in find_recordings(r)]
    if not found:
        raise SystemExit(f"no recordings under {roots} (expected <root>/<session>/<kind>/*.bin)")
    sessions = sorted({s for _, s, _ in found})
    if not val_sessions and not test_sessions:
        if len(sessions) < 3:
            raise SystemExit(f"need at least 3 sessions for an automatic split, have {sessions}; "
                             f"pass --val-sessions / --test-sessions")
        test_sessions, val_sessions = [sessions[-1]], [sessions[-2]]
    split = {"train": [], "val": [], "test": []}
    for path, session, kind in found:
        if kind != NONE_KIND and kind not in targets:
            continue                       # another target (drone while training ball): skip
        rec = load_recording(path, session, kind, frame, lab, tuple(targets))
        if rec is None:
            continue
        name = "test" if session in test_sessions else "val" if session in val_sessions else "train"
        split[name].append(rec)
    print("sessions:", {k: sorted({r.session for r in v}) for k, v in split.items()})
    return split


def balanced_sampler(ds: WindowDataset, n_classes: int, num_samples: int) -> WeightedRandomSampler:
    y = ds.labels()
    freq = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = 1.0 / np.maximum(freq[y], 1.0)
    return WeightedRandomSampler(torch.as_tensor(w), num_samples=num_samples, replacement=True)


# --------------------------------------------------------------------------- loops

def run_epoch(model, loader, device, n_classes, readout, soft, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, loss_sum = 0, 0.0
    conf = np.zeros((n_classes, n_classes), dtype=np.int64)       # rows: truth, cols: predicted
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device)
            counts = readout_counts(model(x), readout)
            loss = (counts - spike_targets(y, n_classes, readout, soft)).square().mean(1).mean() / readout
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                with torch.no_grad():                 # keep weights inside Q1.7's range
                    for p in model.parameters():
                        p.clamp_(-1.0, 1.0)
            loss_sum += loss.item() * len(y)
            total += len(y)
            np.add.at(conf, (y.cpu().numpy(), counts.argmax(1).cpu().numpy()), 1)
    return loss_sum / max(1, total), conf


@torch.no_grad()
def recalibrate_bn(model, loader, device, batches: int) -> None:
    """Re-estimate BatchNorm running statistics on `batches` training batches.

    The running statistics are an exponential average that lags behind fast-moving weights,
    and the board runs the model with BatchNorm folded from those statistics. Without this,
    eval-mode accuracy fell from 56 % (batch statistics) to 14 % on the same windows after
    two short epochs; recalibrated it was 59 %. Weights are untouched.
    """
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns or batches <= 0:
        return
    momenta = [bn.momentum for bn in bns]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None                     # cumulative average over the calibration batches
    was_training = model.training
    model.train()
    for i, (x, _) in enumerate(loader):
        if i >= batches:
            break
        model(x.to(device, non_blocking=True))
    for bn, m in zip(bns, momenta):
        bn.momentum = m
    model.train(was_training)


def summarize(conf: np.ndarray, names: list[str]) -> dict:
    n = conf.sum()
    recall = conf.diagonal() / np.maximum(conf.sum(1), 1)
    none_row, target_rows = conf[0], conf[1:]
    targets_n = max(1, target_rows.sum())
    return {
        "windows": int(n),
        "accuracy": float(conf.trace() / max(1, n)),
        "balanced_accuracy": float(recall[conf.sum(1) > 0].mean()) if n else 0.0,
        "recall": {k: float(r) for k, r in zip(names, recall)},
        "false_trigger_rate": float(none_row[1:].sum() / max(1, none_row.sum())),   # none -> a zone
        "miss_rate": float(target_rows[:, 0].sum() / targets_n),                      # zone -> none
        "wrong_zone_rate": float((target_rows[:, 1:].sum() - np.trace(conf[1:, 1:])) / targets_n),
        "confusion": conf.tolist(),
    }


def fmt(m: dict) -> str:
    return (f"acc {m['accuracy']:.1%}  bal {m['balanced_accuracy']:.1%}  "
            f"false-trig {m['false_trigger_rate']:.1%}  miss {m['miss_rate']:.1%}  "
            f"wrong-zone {m['wrong_zone_rate']:.1%}")


# --------------------------------------------------------------------------- export

Q1_7_MAX = 127 / 128


def board_weights_text(params: dict) -> str:
    """weights.txt for spleat-server: all weights, all biases, then threshold and leak per
    layer, one integer per line, no trailing newline.

    Weights and biases (8-bit Q1.7) are written as two's complement, unsigned 0-255
    (-16 -> 240), as in Durance's example weights.txt. riverops 0.2.0's export_params gives
    them signed (-128..127); the board reads those wrongly. Checked on the KV260 on 22 Nov
    (docs/fpga-zones3-2026-11-22.md): with two's complement the output spikes match the
    fixed-point emulation neuron by neuron, signed they do not. Threshold and leak (16-bit
    Q4.12, positive) are written as they are.
    """
    values = []
    for key, v in params.items():
        v = v if isinstance(v, list) else [v]
        if key.startswith(("weights_", "bias_")):
            if any(not -128 <= x <= 127 for x in v):
                raise ValueError(f"{key}: values outside 8-bit Q1.7")
            v = [x & 0xFF for x in v]
        values.extend(v)
    return "\n".join(map(str, values))


def fold_for_board(model: TargetSNN, qcfg):
    """The network as the board runs it: BatchNorm folded, then each layer rescaled so its
    folded weights and bias fit Q1.7, then fixed-point emulation.

    Folding BatchNorm can push weights far past Q1.7's [-1, 127/128]: in the first layer
    it multiplies by 1/std of the small event-count inputs (|w| up to 16.5 after two short
    epochs), and to_fixed would clip them. Dividing a layer's weights, bias and threshold by
    the same factor s >= 1 leaves its spikes unchanged (the input is the frame or the
    previous layer's spikes, the leak is linear, the reset is to zero), and the threshold
    is a per-layer value in weights.txt. Returns (model, per-layer scales).
    """
    from riverops.nn import fuse_model, prepare_fixed_point

    fused = fuse_model(model.eval()).eval()
    scales = []
    with torch.no_grad():
        for layer in fused.net:
            w, b = layer.conv.weight, layer.conv.bias
            peak = max(float(w.abs().max()), float(b.abs().max()) if b is not None else 0.0)
            sc = max(1.0, peak / Q1_7_MAX)
            w.div_(sc)
            if b is not None:
                b.div_(sc)
            layer.lif.v_threshold = layer.lif.v_threshold / sc
            scales.append(sc)
    return prepare_fixed_point(fused, qcfg).eval(), scales


def export(model: TargetSNN, example: torch.Tensor, out: Path, frame: FrameConfig,
           n_classes: int, val_loader, device, readout, soft, names) -> dict:
    """Architecture JSON, then the board's parameters: BatchNorm folded, rescaled to fit Q1.7 (see
    fold_for_board), fixed point. Durance's notebook exported without the rescale."""
    from riverops.export import export_for_river
    from riverops.nn import FixedPointFormat, QuantConfig

    model.eval()
    (out / "model.json").write_text(export_for_river(model.net, example))
    frac = 12
    qcfg = QuantConfig(weight_format=FixedPointFormat(total_bits=8, frac_bits=7),
                       potential_format=FixedPointFormat(total_bits=16, frac_bits=frac),
                       enable_weight_quant=True, enable_potential_quant=True)
    hw, scales = fold_for_board(model, qcfg)
    print("per-layer rescale to fit Q1.7:", " ".join(f"{s_:.2f}" for s_ in scales))
    quant = {"layer_scales": scales}
    try:      # fixed-point emulation of what the board computes (approximate, per riverops)
        _, conf = run_epoch(hw, val_loader, device, n_classes, readout, soft)
        quant.update(summarize(conf, names))
        print("fixed-point val (BN folded, as on the board):", fmt(quant))
    except Exception as e:                 # noqa: BLE001 - informative only
        print(f"could not evaluate the fixed-point model ({e!r}); export continues")

    weights, biases, neurons = {}, {}, {}
    for i, layer in enumerate(hw.net):
        p = layer.export_params()
        weights[f"weights_{i}"], biases[f"bias_{i}"] = p["weights"], p["bias"]
        neurons[f"threshold_{i}"], neurons[f"leak_{i}"] = p["threshold"], p["leak"]
    params = {**weights, **biases, **neurons}
    (out / "parameters.json").write_text(json.dumps(params, indent=2) + "\n")
    (out / "weights.txt").write_text(board_weights_text(params))

    s = frame.size
    (out / "config.yaml").write_text(
        "application:\n  host: \"0.0.0.0\"\n  port: 3333\n  weights_file: /root/weights.txt\n"
        "  max_concurrency: 40\n\n"
        f"inputs:\n  channels: 2\n  height: {s}\n  width: {s}\n  scaling: {float(2 ** frac)}\n\n"
        f"outputs:\n  channels: {n_classes}\n  height: 1\n  width: 1\n")
    (out / "metavision_camera.json").write_text(json.dumps({
        "frequency": 1e6 / frame.bin_us, "downsampling": frame.downsample,
        "noise_filtering_factor": 0.5,
        "biases": {"bias_diff": 51, "bias_diff_off": 40, "bias_diff_on": 40,
                   "bias_fo": 29, "bias_hpf": 0, "bias_refr": 82}}, indent=2) + "\n")
    return quant


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", default=["data/river"], help="roots: <root>/<session>/<kind>/*.bin")
    ap.add_argument("--targets", nargs="+", default=["ball"], help="kinds labelled by position")
    ap.add_argument("--val-sessions", nargs="*", default=[])
    ap.add_argument("--test-sessions", nargs="*", default=[])
    ap.add_argument("--out", default="out/river/run")
    ap.add_argument("--init", help="checkpoint to start from (fine-tuning)")
    ap.add_argument("--freeze-trunk", action="store_true", help="train only the last layer (new target)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--samples-per-epoch", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-bins", type=int, default=3)
    ap.add_argument("--min-events", type=int, default=30)
    ap.add_argument("--downsample", type=int, default=4)
    ap.add_argument("--norm", default="mean", choices=["mean", "sum", "binary"])
    ap.add_argument("--flip-x", action="store_true")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--readout", type=int, default=20)
    ap.add_argument("--no-soft", action="store_true", help="no 50 %% target on neighbouring zones")
    ap.add_argument("--channels", type=int, nargs="+", default=[16, 32, 32, 64])
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bn-calib-batches", type=int, default=20,
                    help="training batches to re-estimate BatchNorm statistics before each validation (0: off)")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = pick_device()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    frame = FrameConfig(downsample=a.downsample, norm=a.norm, flip_x=a.flip_x)
    lab = LabelConfig(n_bins=a.n_bins, min_events=a.min_events)
    win = WindowConfig(frames=a.frames, readout=a.readout)
    mcfg = ModelConfig(channels=tuple(a.channels))
    names, n_classes, soft = lab.names, lab.n_classes, not a.no_soft
    print(f"device {device}  classes {names}  input {frame.size}x{frame.size}")

    t0 = time.time()
    split = load_split(a.data, a.targets, frame, lab, a.val_sessions, a.test_sessions)
    train_ds = WindowDataset(split["train"], frame, lab, win, augment=True)
    val_ds = WindowDataset(split["val"], frame, lab, win)
    test_ds = WindowDataset(split["test"], frame, lab, win)
    for k, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        hist = np.bincount(ds.labels(), minlength=n_classes) if len(ds) else np.zeros(n_classes, int)
        print(f"{k}: {len(ds)} windows  " + "  ".join(f"{n}={h}" for n, h in zip(names, hist)))
    print(f"data ready in {time.time() - t0:.0f} s")
    if not len(train_ds) or not len(val_ds):
        raise SystemExit("empty train or val split")

    kw = dict(batch_size=a.batch, num_workers=a.workers, pin_memory=device.type == "cuda",
              persistent_workers=a.workers > 0)
    train_loader = DataLoader(train_ds, sampler=balanced_sampler(train_ds, n_classes, a.samples_per_epoch), **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)

    model = TargetSNN(n_classes, frame.size, mcfg).to(device)
    if a.init:
        state = torch.load(a.init, map_location=device, weights_only=True)["model_state_dict"]
        model.load_state_dict(state)
        print(f"initialised from {a.init}")
    if a.freeze_trunk:
        for layer in list(model.net)[:-1]:
            for p in layer.parameters():
                p.requires_grad_(False)
    print(f"{count_parameters(model)} parameters, final kernel {model.final_kernel}")
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=a.lr, weight_decay=a.weight_decay)

    meta = {"names": names, "frame": asdict(frame), "labels": asdict(lab), "window": asdict(win),
            "model": mcfg.to_dict(), "n_classes": n_classes, "targets": a.targets, "args": vars(a)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    best, best_path = -1.0, out / "best.pt"
    for epoch in range(1, a.epochs + 1):
        te = time.time()
        tr_loss, tr_conf = run_epoch(model, train_loader, device, n_classes, win.readout, soft, opt)
        recalibrate_bn(model, train_loader, device, a.bn_calib_batches)
        va_loss, va_conf = run_epoch(model, val_loader, device, n_classes, win.readout, soft)
        tr, va = summarize(tr_conf, names), summarize(va_conf, names)
        mark = ""
        if va["balanced_accuracy"] > best:
            best = va["balanced_accuracy"]
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val": va, "meta": meta}, best_path)
            mark = "  *saved"
        print(f"epoch {epoch:2d}/{a.epochs} {time.time() - te:5.0f} s | train loss {tr_loss:.4f} {fmt(tr)} "
              f"| val loss {va_loss:.4f} {fmt(va)}{mark}")

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    metrics = {"best_epoch": ckpt["epoch"], "val": ckpt["val"]}
    if len(test_ds):
        _, conf = run_epoch(model, DataLoader(test_ds, shuffle=False, **kw), device, n_classes, win.readout, soft)
        metrics["test"] = summarize(conf, names)
        print("test:", fmt(metrics["test"]))
        print("confusion (rows truth, cols predicted, order", names, "):")
        print(np.array(metrics["test"]["confusion"]))
    if not a.no_export:
        example = val_ds[0][0].unsqueeze(0).to(device)
        metrics["val_quantized"] = export(model, example, out, frame, n_classes, val_loader, device,
                                          win.readout, soft, names)
        print(f"exported model.json, weights.txt, config.yaml to {out}")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
