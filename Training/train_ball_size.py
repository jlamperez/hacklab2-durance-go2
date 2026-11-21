"""Ball position and size at 320x320: ``train_motion_buddy_fast.py`` plus a radius head.

The benchmark has no size labels (only gt_x, gt_y, gt_t), so the radius label
is estimated from the events. For each 1 ms window it is the radius around
the label that holds RADIUS_FRACTION of the event pixels in excess of the
local background, median-smoothed over neighbouring labels of the same
recording. It measures the visible event blob (ball plus 1 ms of motion),
not the physical ball. The crop scale is applied to it exactly as to the
position, so zoomed-in crops get proportionally larger radius labels.

Network: a small spiking U-Net. The encoder goes down to a 10x10 grid so a
cell sees about 200 px (zoomed-in balls reach ~140 px across), the decoder
comes back to the 40x40 grid through nearest upsampling, and additive skips
from the 20x20 and 40x40 encoder layers keep the position precise. The head
has 1 + RADIUS_BINS spiking channels: channel 0 is the position heatmap; the
others vote for log-spaced radius bins and are read at the predicted cell.
Only ops the RIVER exporter lowers are used: square ConvLIFs, upsample
directly followed by a ConvLIF (UpConvLIF) and additions (extra incoming
edges).

Crops, splits, data loading and video writing come from the fast script. The
export writes ``ball_model.json`` and ``ball_parameters.{json,txt}``, with
parameters numbered in layer order (printed at export)::

    python train_ball_size.py
    python train_ball_size.py visualize
    python train_ball_size.py predict data/recording.h5 --window-ms 5
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

import train_motion_buddy as base
import train_motion_buddy_fast as fast
from riverops.export import export_for_river
from riverops.nn import (
    FixedPointFormat, NearestUpsample, QuantConfig, SpikingConvBN2d, prepare_qat,
)

ARCHITECTURE = "unet-v1"
CHECKPOINT = Path("ball_size_best.pt")
SENSOR = fast.SENSOR
HEAT = fast.HEAT
STRIDE = fast.HEATMAP_STRIDE
RADIUS_BINS = 8
RADIUS_RANGE = (1.5, 64.0)
RADIUS_CENTERS = np.geomspace(*RADIUS_RANGE, RADIUS_BINS)
RADIUS_SIGMA = 0.75
RADIUS_WEIGHT = 1.0
RADIUS_SEARCH = 48
RADIUS_FRACTION = 0.8
RADIUS_MIN_EVENTS = 6.0
RADIUS_SMOOTH = 4
RADIUS_FALLBACK = 5.0


def window_radius(pixels, xy):
    """Event-blob radius in source pixels around label ``xy``; NaN if too few events."""
    if not len(pixels):
        return np.nan
    distance = np.hypot(pixels[:, 1] - xy[0], pixels[:, 0] - xy[1])
    ring = np.count_nonzero((distance >= RADIUS_SEARCH) & (distance < 2 * RADIUS_SEARCH))
    density = ring / (3 * np.pi * RADIUS_SEARCH ** 2)
    radii = np.arange(1, RADIUS_SEARCH + 1, dtype=np.float64)
    inside = np.searchsorted(np.sort(distance), radii, side="right")
    excess = inside - density * np.pi * radii ** 2
    total = excess.max()
    if total < RADIUS_MIN_EVENTS:
        return np.nan
    return float(radii[np.argmax(excess >= RADIUS_FRACTION * total)])


def smooth_radii(raw, lengths):
    """Median over +-RADIUS_SMOOTH labels inside each recording; gaps get its median."""
    out = np.full_like(raw, np.nan)
    start = 0
    for length in lengths:
        segment = raw[start:start + length]
        part = out[start:start + length]
        for i in range(length):
            window = segment[max(0, i - RADIUS_SMOOTH):i + RADIUS_SMOOTH + 1]
            window = window[np.isfinite(window)]
            if window.size:
                part[i] = np.median(window)
        finite = segment[np.isfinite(segment)]
        if finite.size:
            part[~np.isfinite(part)] = np.median(finite)
        start += length
    fallback = np.nanmedian(out) if np.isfinite(out).any() else RADIUS_FALLBACK
    out[~np.isfinite(out)] = fallback
    return out


def label_radii(events, offsets, xy, lengths, name):
    """Smoothed radius label (source pixels) for every window."""
    raw = np.array([
        window_radius(events[offsets[i]:offsets[i + 1]], xy[i]) for i in range(len(xy))
    ])
    if sum(lengths) != len(xy):
        print(f"{name}: recording lengths do not match; smoothing across the whole split")
        lengths = [len(xy)]
    radii = smooth_radii(raw, lengths).astype(np.float32)
    print(
        f"{name}: radius labels from {np.isfinite(raw).mean():.0%} of windows, "
        f"median {np.median(radii):.1f} px, 5-95% {np.percentile(radii, 5):.0f}-"
        f"{np.percentile(radii, 95):.0f} px (source pixels)",
        flush=True,
    )
    return radii


def _visible_labels(path):
    with h5py.File(path, "r") as handle:
        x = np.atleast_1d(base._column(handle, "gt_x"))
        y = np.atleast_1d(base._column(handle, "gt_y"))
    return int(np.count_nonzero(
        (x >= 0) & (y >= 0) & (x < fast.SOURCE_WIDTH) & (y < fast.SOURCE_HEIGHT)
    ))


def split_file_lengths(dataset_dir):
    """Labels per recording, in the order ``base._prepare_splits`` concatenates them."""
    projected = base._recordings(dataset_dir / "projected")
    hand = base._recordings(dataset_dir / "hand_labeled")
    bucket = {
        name: [path for path in hand if base._hand_labeled_bucket(path) == name]
        for name in ("train", "validation", "test")
    }
    return tuple(
        [_visible_labels(path) for path in paths]
        for paths in (projected + bucket["train"], bucket["validation"], bucket["test"])
    )


class SizedDataset(fast.SensorDataset):
    """SensorDataset whose targets are (x, y, radius) in sensor pixels; radius 0 = unknown."""

    def __init__(self, events, offsets, positions, radii, crop, noise_seed=None):
        super().__init__(events, offsets, positions, crop, noise_seed)
        self.radii = (
            np.zeros(len(self), dtype=np.float32) if radii is None
            else np.asarray(radii, dtype=np.float32)
        )

    def __getitem__(self, index):
        xy = self.positions[index]
        x0, y0, side = self.crop(index, xy)
        scale = SENSOR / side
        pixels = self.events[self.offsets[index]:self.offsets[index + 1]]
        target = np.array(
            ((xy[0] - x0) * scale, (xy[1] - y0) * scale, self.radii[index] * scale),
            dtype=np.float32,
        )
        flat = fast.resample(pixels, x0, y0, scale, self.noise_rng(index))
        return torch.from_numpy(flat), torch.from_numpy(target)


def benchmark_splits():
    """Train, validation and test SizedDatasets; the crops match the fast script's."""
    dataset_dir = base.ensure_dataset()
    arrays = base._read_splits(dataset_dir)
    lengths = split_file_lengths(dataset_dir)
    crops = (
        fast.random_crop(np.random.default_rng(base.SEED)),
        fast.fixed_crop(base.SEED + 1),
        fast.fixed_crop(base.SEED + 2),
    )
    noise_seeds = (None, base.SEED + 1, base.SEED + 2)
    names = ("train", "validation", "test")
    return tuple(
        SizedDataset(
            events, offsets, xy, label_radii(events, offsets, xy, n, name), crop, noise_seed,
        )
        for (events, offsets, xy), n, crop, noise_seed, name
        in zip(arrays, lengths, crops, noise_seeds, names)
    )


LIF = {
    "initial_tau": 1.0 / (1.0 - 0.93),
    "learnable_leak": False,
    "v_threshold": 1.0,
    "reset_mode": "HARD",
}


def _conv(in_channels, out_channels, kernel_size, stride=1, padding=None, first_layer=False):
    return SpikingConvBN2d(
        in_channels, out_channels, kernel_size=kernel_size, stride=stride,
        padding=(kernel_size - 1) // 2 if padding is None else padding,
        bias=False, first_layer=first_layer, **LIF,
    )


class SizeUNet(nn.Module):
    """Spiking U-Net on [B, T, 1, 320, 320]; returns head spikes [B, T, 1 + RADIUS_BINS, 40, 40].

    Grid (cell size) and receptive field per stage:
      enc1 80x80 (4 px, 8)  enc2-3 40x40 (8 px, 36)  enc4-5 20x20 (16 px, 92)
      enc6-7 10x10 (32 px, 204)  dec1/mix1 20x20  dec2/mix2/head 40x40 (~300).
    Modules are registered in forward order, which is the export order.
    """

    def __init__(self):
        super().__init__()
        self.enc1 = _conv(1, 16, 8, stride=4, padding=2, first_layer=True)
        self.enc2 = _conv(16, 32, 4, stride=2, padding=1)
        self.enc3 = _conv(32, 32, 3)
        self.enc4 = _conv(32, 64, 4, stride=2, padding=1)
        self.enc5 = _conv(64, 64, 3)
        self.enc6 = _conv(64, 64, 4, stride=2, padding=1)
        self.enc7 = _conv(64, 64, 3)
        self.up1 = NearestUpsample((2, 2))
        self.dec1 = _conv(64, 64, 3)
        self.mix1 = _conv(64, 64, 3)
        self.up2 = NearestUpsample((2, 2))
        self.dec2 = _conv(64, 32, 3)
        self.mix2 = _conv(32, 32, 3)
        self.head = _conv(32, 1 + RADIUS_BINS, 1, padding=0)

    def forward(self, x):
        e3 = self.enc3(self.enc2(self.enc1(x)))
        e5 = self.enc5(self.enc4(e3))
        e7 = self.enc7(self.enc6(e5))
        m1 = self.mix1(self.dec1(self.up1(e7)) + e5)
        m2 = self.mix2(self.dec2(self.up2(m1)) + e3)
        return self.head(m2)


class SizedBallDetector(nn.Module):
    """``SizeUNet`` plus the spike count over time (the count is decoded off-chip)."""

    def __init__(self):
        super().__init__()
        self.body = SizeUNet()

    def forward(self, x):
        """Spike counts, [B, 1 + RADIUS_BINS, HEAT, HEAT]."""
        return self.body(x).sum(dim=1)


def export_model(model, example_frames, device):
    """Write ball_model.json and ball_parameters.{json,txt} for the U-Net."""
    model.eval()
    model_json = export_for_river(model.body, example_frames.unsqueeze(0).to(device))
    Path("ball_model.json").write_text(model_json)
    print("Saved architecture to: ball_model.json")

    quant_config = QuantConfig(
        weight_format=FixedPointFormat(total_bits=8, frac_bits=7),
        potential_format=FixedPointFormat(total_bits=16, frac_bits=12),
        enable_weight_quant=True,
        enable_potential_quant=True,
    )
    prepared = prepare_qat(model, quant_config).eval()
    layers = [
        (name, module) for name, module in prepared.body.named_children()
        if isinstance(module, SpikingConvBN2d)
    ]
    weights, biases, neurons = {}, {}, {}
    for index, (name, layer) in enumerate(layers):
        params = layer.export_params()
        weights[f"weights_{index}"] = params["weights"]
        biases[f"bias_{index}"] = params["bias"]
        neurons[f"threshold_{index}"] = params["threshold"]
        neurons[f"leak_{index}"] = params["leak"]
    print("Parameter order:", ", ".join(f"{i}={name}" for i, (name, _) in enumerate(layers)))

    parameters = {**weights, **biases, **neurons}
    Path("ball_parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    values = []
    for value in parameters.values():
        values.extend(value if isinstance(value, list) else [value])
    Path("ball_parameters.txt").write_text("\n".join(map(str, values)))
    print("Saved parameters to: ball_parameters.json and ball_parameters.txt")


def _log_centers(device):
    return torch.log(torch.as_tensor(RADIUS_CENTERS, dtype=torch.float32, device=device))


def radius_logits(counts, rows, cols, time_steps):
    """Radius-bin logits read at one cell per sample, [B, RADIUS_BINS]."""
    batch = torch.arange(counts.shape[0], device=counts.device)
    return counts[batch, 1:, rows, cols] * (fast.LOGIT_SCALE / time_steps)


def compute_loss(counts, targets, time_steps):
    """Heatmap cross-entropy plus radius-bin cross-entropy at the true cell."""
    positions, radii = targets[:, :2], targets[:, 2]
    loss = fast.compute_loss(counts[:, 0], positions, time_steps)
    rows = (positions[:, 1] / STRIDE).long().clamp(0, HEAT - 1)
    cols = (positions[:, 0] / STRIDE).long().clamp(0, HEAT - 1)
    log_centers = _log_centers(counts.device)
    log_radius = radii.clamp(min=1e-3).log().clamp(log_centers[0], log_centers[-1])
    step = float(np.log(RADIUS_CENTERS[1] / RADIUS_CENTERS[0]))
    target = torch.exp(
        -((log_centers[None, :] - log_radius[:, None]) / step).square()
        / (2 * RADIUS_SIGMA ** 2)
    )
    target = target / target.sum(dim=1, keepdim=True)
    log_probs = radius_logits(counts, rows, cols, time_steps).log_softmax(dim=1)
    radius_loss = -(target * log_probs).sum(dim=1)
    known = (radii > 0).to(radius_loss.dtype)
    return loss + RADIUS_WEIGHT * (radius_loss * known).sum() / known.sum().clamp(min=1.0)


def decode(counts, time_steps):
    """(x, y, radius) in sensor pixels per sample."""
    heat = counts[:, 0]
    xy = fast.decode_positions(heat)
    index = heat.flatten(1).argmax(dim=1)
    probs = radius_logits(counts, index // HEAT, index % HEAT, time_steps).softmax(dim=1)
    radius = torch.exp((probs * _log_centers(counts.device)).sum(dim=1))
    return torch.cat((xy, radius[:, None]), dim=1)


def run_epoch(model, loader, device, time_steps, optimizer=None, log_prefix=""):
    training = optimizer is not None
    model.train(training)
    total_loss = torch.zeros((), device=device)
    total_error = torch.zeros((), device=device)
    total_radius = torch.zeros((), device=device)
    total_known = torch.zeros((), device=device)
    total = 0
    batch_count = len(loader)
    with torch.set_grad_enabled(training):
        for step, (flat, targets) in enumerate(loader, start=1):
            if step == 1 or step % 10 == 0 or step == batch_count:
                print(f"{log_prefix} batch {step}/{batch_count}", flush=True)
            targets = targets.to(device, non_blocking=True)
            frames = fast.to_frames(flat, len(targets), device, time_steps)
            if training:
                optimizer.zero_grad()
            counts = model(frames)
            loss = compute_loss(counts, targets, time_steps)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.clamp_(-1.0, 1.0)
            decoded = decode(counts.detach(), time_steps)
            known = targets[:, 2] > 0
            count = len(targets)
            total_loss += loss.detach() * count
            total_error += torch.linalg.vector_norm(decoded[:, :2] - targets[:, :2], dim=1).sum()
            total_radius += (
                (decoded[:, 2] - targets[:, 2]).abs() / targets[:, 2].clamp(min=1e-3) * known
            ).sum()
            total_known += known.sum()
            total += count
    return (
        total_loss.item() / total,
        total_error.item() / total,
        total_radius.item() / max(total_known.item(), 1.0),
    )


def infer(model, dataset, device, batch_size, time_steps):
    """Predicted (x, y, radius) in sensor pixels, shape (N, 3), in dataset order."""
    predicted = []
    model.eval()
    with torch.no_grad():
        for flat, targets in fast.make_loader(dataset, batch_size):
            frames = fast.to_frames(flat, len(targets), device, time_steps)
            predicted.append(decode(model(frames), time_steps).cpu().numpy())
    if not predicted:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(predicted, axis=0)


def train(batch_size, epochs, time_steps):
    torch.manual_seed(base.SEED)
    torch.backends.cudnn.benchmark = True
    device = base.pick_device()
    print("Device:", device, flush=True)

    train_dataset, validation_dataset, test_dataset = benchmark_splits()
    train_loader = fast.make_loader(train_dataset, batch_size, shuffle=True)
    validation_loader = fast.make_loader(validation_dataset, batch_size)
    print("Training frames:", len(train_dataset))
    print("Validation frames:", len(validation_dataset))
    print(
        f"Sensor: {SENSOR}x{SENSOR}, crop sides "
        f"{fast.CROP_SIDES[0]:.0f}-{fast.CROP_SIDES[1]:.0f} px"
    )
    print("Batch size:", batch_size)
    print("Time steps:", time_steps)

    example_frames = base.save_sample_figure(
        [fast.dense_sample(validation_dataset, 0, time_steps)], Path("ball_size_sample.png"),
    )

    model = SizedBallDetector().to(device)
    model.eval()
    with torch.no_grad():
        counts = model(example_frames.unsqueeze(0).to(device))
    if tuple(counts.shape[1:]) != (1 + RADIUS_BINS, HEAT, HEAT):
        raise RuntimeError(f"Unexpected output shape {tuple(counts.shape)}")
    print("Output shape:", tuple(counts.shape), flush=True)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=base.LEARNING_RATE, weight_decay=base.WEIGHT_DECAY,
    )
    best_validation_error = float("inf")

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs} training", flush=True)
        train_loss, train_error, train_radius = run_epoch(
            model, train_loader, device, time_steps, optimizer,
            log_prefix=f"Epoch {epoch} train",
        )
        print(f"Epoch {epoch}/{epochs} validation", flush=True)
        validation_loss, validation_error, validation_radius = run_epoch(
            model, validation_loader, device, time_steps,
            log_prefix=f"Epoch {epoch} validation",
        )
        if validation_error < best_validation_error:
            best_validation_error = validation_error
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "validation_error": validation_error,
                "validation_radius_error": validation_radius,
                "sensor": SENSOR,
                "crop_sides": fast.CROP_SIDES,
                "time_steps": time_steps,
                "heatmap_stride": STRIDE,
                "radius_centers": RADIUS_CENTERS.tolist(),
                "architecture": ARCHITECTURE,
            }, CHECKPOINT)
        print(
            f"Epoch {epoch}/{epochs} | Training loss: {train_loss:.4f} | "
            f"Training error: {train_error:.2f} px, radius {train_radius:.0%} | "
            f"Validation loss: {validation_loss:.4f} | "
            f"Validation error: {validation_error:.2f} px, radius {validation_radius:.0%}"
        )

    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Test", flush=True)
    test_loss, test_error, test_radius = run_epoch(
        model, fast.make_loader(test_dataset, batch_size), device, time_steps,
        log_prefix="Test",
    )
    print("Best epoch:", checkpoint["epoch"])
    print(f"Best validation error: {checkpoint['validation_error']:.2f} px")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test error: {test_error:.2f} px (sensor pixels)")
    print(f"Test radius error: {test_radius:.0%} (mean relative)")

    export_model(model, example_frames, device)


def load_model(checkpoint_path, device):
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path}. Train first with: python train_ball_size.py"
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("sensor") != SENSOR or checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError(
            f"{checkpoint_path} is not a {SENSOR}x{SENSOR} {ARCHITECTURE} checkpoint; retrain."
        )
    model = SizedBallDetector().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    time_steps = int(checkpoint["time_steps"])
    print("Time steps:", time_steps)
    return model, time_steps


def write_video(model, device, dataset, name, output_path, fps, batch_size, time_steps):
    """Render every window with its predicted (and, if labeled, true) position and size."""
    predicted = infer(model, dataset, device, batch_size, time_steps)
    writer = fast.VideoWriter(output_path, fps, SENSOR * fast.DISPLAY_SCALE)
    errors, radius_errors = [], []
    try:
        for index in range(len(dataset)):
            flat, target = dataset[index]
            px, py, pr = (float(v) for v in predicted[index])
            caption = f"{name}  {index + 1}/{len(dataset)}  pred ({px:.0f}, {py:.0f}) r {pr:.1f}"
            truth, truth_radius = None, 6.0
            if dataset.labeled:
                tx, ty, tr = target.tolist()
                truth = (tx, ty)
                error = float(np.hypot(px - tx, py - ty))
                errors.append(error)
                caption += f"  gt ({tx:.0f}, {ty:.0f}) r {tr:.1f}  err {error:.1f}"
                if tr > 0:
                    truth_radius = tr
                    radius_errors.append(abs(pr - tr) / tr)
            writer.write(fast.render_frame(
                flat, (px, py), truth, caption,
                prediction_radius=pr, truth_radius=truth_radius,
            ))
    finally:
        writer.close()
    print(f"Wrote {writer.frames} frames to {Path(output_path).resolve()}")
    if errors:
        print(f"Mean error: {float(np.mean(errors)):.2f} px (sensor pixels)")
    if radius_errors:
        print(f"Mean radius error: {float(np.mean(radius_errors)):.0%}")


def visualize_test(checkpoint_path, output_path, fps, batch_size):
    """Run the checkpoint on the fixed test crops (takes 7 and 8) and write an mp4."""
    device = base.pick_device()
    print("Device:", device, flush=True)
    model, time_steps = load_model(checkpoint_path, device)
    write_video(model, device, benchmark_splits()[2], "test 7-8", output_path, fps,
                batch_size, time_steps)


def predict_recording(recording, checkpoint_path, output_path, fps, batch_size,
                      step_ms, window_ms, use_labels):
    """Run the checkpoint on one HDF5 event file and write an mp4."""
    fast.check_hdf5(recording)
    device = base.pick_device()
    print("Device:", device, flush=True)
    model, time_steps = load_model(checkpoint_path, device)
    windows = fast.recording_dataset(
        recording, step_ms * 1000.0, window_ms * 1000.0, use_labels,
    )
    radii = None
    if windows.labeled:
        radii = label_radii(
            windows.events, windows.offsets, windows.positions, [len(windows)], recording.name,
        )
    dataset = SizedDataset(
        windows.events, windows.offsets, windows.positions if windows.labeled else None,
        radii, windows.crop, noise_seed=0,
    )
    output_path = output_path or Path(f"{recording.stem}_size_pred.mp4")
    write_video(model, device, dataset, recording.name, output_path, fps,
                batch_size, time_steps)


def main():
    parser = argparse.ArgumentParser(description="Spiking ball detector with size, 320x320.")
    parser.add_argument("--batch-size", type=int, default=fast.BATCH_SIZE)
    subparsers = parser.add_subparsers(dest="command")
    train_parser = subparsers.add_parser("train", help="train on the benchmark and export")
    train_parser.add_argument("--epochs", type=int, default=base.EPOCHS)
    train_parser.add_argument("--time-steps", type=int, default=fast.TIME_STEPS)
    view = subparsers.add_parser("visualize", help="write a test-set video")
    view.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    view.add_argument("--output", type=Path, default=Path("ball_size_test.mp4"))
    view.add_argument("--fps", type=int, default=15)
    predict = subparsers.add_parser(
        "predict", help="write a prediction video for an HDF5 event file",
    )
    predict.add_argument("recording", type=Path)
    predict.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    predict.add_argument("--output", type=Path, default=None,
                         help="default: <recording name>_size_pred.mp4 in the current folder")
    predict.add_argument("--fps", type=int, default=30)
    predict.add_argument("--step-ms", type=float, default=5.0,
                         help="window spacing for files without labels")
    predict.add_argument("--window-ms", type=float, default=base.ACC_TIME / 1000.0,
                         help="events per frame; training used 1 ms")
    predict.add_argument("--no-labels", action="store_true",
                         help="ignore gt_* and use evenly spaced windows")
    args = parser.parse_args()
    if args.command == "visualize":
        visualize_test(args.checkpoint, args.output, args.fps, args.batch_size)
    elif args.command == "predict":
        predict_recording(
            args.recording, args.checkpoint, args.output, args.fps, args.batch_size,
            args.step_ms, args.window_ms, not args.no_labels,
        )
    else:
        train(
            args.batch_size,
            getattr(args, "epochs", base.EPOCHS),
            getattr(args, "time_steps", fast.TIME_STEPS),
        )


if __name__ == "__main__":
    main()
