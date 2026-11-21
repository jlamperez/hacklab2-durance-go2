"""Ball position at 320x320 with a spiking U-Net; the ball size is measured, not predicted.

The network predicts only the ball centre. The circle size drawn in the
videos comes from the events: the radius around a point that holds
RADIUS_FRACTION of the event pixels in excess of the local background,
median-smoothed over neighbouring frames. It measures the visible event blob
(ball plus 1 ms of motion), not the physical ball. On labeled data it is
measured around the label (and scaled with the crop like the position) and
used for both circles and for the "within 1 radius" metric; on unlabeled
recordings it is measured around the prediction.

Network: a small spiking U-Net. The encoder goes down to a 10x10 grid so a
cell sees about 200 px (zoomed-in balls reach ~140 px across), the decoder
comes back to an 80x80 grid (4 px cells) through nearest upsampling, and
additive skips from the 20x20, 40x40 and 80x80 encoder layers keep the
position precise. The head is read like this, all on the 80x80 grid:

- channel 0: position heatmap; the peak cell is the coarse position;
- 2 x OFFSET_BINS channels: votes for where the ball sits inside the cell,
  along x and along y, read at the peak cell.

Only ops the RIVER exporter lowers are used: square ConvLIFs, upsample
directly followed by a ConvLIF (UpConvLIF) and additions (extra incoming
edges).

Crops, splits, data loading and video writing come from the fast script. The
export writes ``ball_model.json`` and ``ball_parameters.{json,txt}``, with
parameters numbered in layer order (printed at export)::

    python train_ball_size.py
    python train_ball_size.py export
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

ARCHITECTURE = "unet-position-v1"
CHECKPOINT = Path("ball_size_best.pt")
SENSOR = fast.SENSOR
HEAT = 80
STRIDE = SENSOR // HEAT
HEAT_SIGMA = 1.5
OFFSET_BINS = 4
OFFSET_SIGMA = 0.5
OFFSET_WEIGHT = 1.0
LOGIT_SCALE_RANGE = (1.0, 64.0)
MIN_LEARNING_RATE = 1e-5
SILENT_RATE = 0.001
RADIUS_SEARCH = 48
RADIUS_FRACTION = 0.8
RADIUS_MIN_EVENTS = 6.0
RADIUS_SMOOTH = 4
RADIUS_FALLBACK = 5.0
X_OFFSET_CHANNELS = slice(1, 1 + OFFSET_BINS)
Y_OFFSET_CHANNELS = slice(X_OFFSET_CHANNELS.stop, X_OFFSET_CHANNELS.stop + OFFSET_BINS)
HEAD_CHANNELS = Y_OFFSET_CHANNELS.stop


def window_radius(pixels, xy):
    """Event-blob radius around ``xy`` for (y, x) ``pixels``, in their units; NaN if too few."""
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


class BallUNet(nn.Module):
    """Spiking U-Net on [B, T, 1, 320, 320]; returns head spikes [B, T, HEAD_CHANNELS, 80, 80].

    Grid (cell size) and receptive field per stage:
      enc1 80x80 (4 px, 8)  enc2-3 40x40 (8 px, 36)  enc4-5 20x20 (16 px, 92)
      enc6-7 10x10 (32 px, 204)  dec1/mix1 20x20  dec2/mix2 40x40
      dec3/mix3/head 80x80 (~300).
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
        self.up3 = NearestUpsample((2, 2))
        self.dec3 = _conv(32, 16, 3)
        self.mix3 = _conv(16, 16, 3)
        self.head = _conv(16, HEAD_CHANNELS, 1, padding=0)

    def forward(self, x):
        e1 = self.enc1(x)
        e3 = self.enc3(self.enc2(e1))
        e5 = self.enc5(self.enc4(e3))
        e7 = self.enc7(self.enc6(e5))
        m1 = self.mix1(self.dec1(self.up1(e7)) + e5)
        m2 = self.mix2(self.dec2(self.up2(m1)) + e3)
        m3 = self.mix3(self.dec3(self.up3(m2)) + e1)
        return self.head(m3)


class BallDetector(nn.Module):
    """``BallUNet`` plus the off-chip readout: spike count over time and logit scales.

    ``log_logit_scales`` holds one learnable scale per head group (heatmap,
    offsets) that turns spike rates into logits. It is used only in the loss
    and the off-chip decoding, so it is not part of the exported network.
    """

    def __init__(self):
        super().__init__()
        self.body = BallUNet()
        self.log_logit_scales = nn.Parameter(
            torch.full((2,), float(np.log(fast.LOGIT_SCALE)))
        )

    def forward(self, x):
        """Spike counts, [B, HEAD_CHANNELS, HEAT, HEAT]."""
        return self.body(x).sum(dim=1)

    def logit_scales(self):
        """(heatmap, offset) scales, kept within LOGIT_SCALE_RANGE."""
        low, high = np.log(LOGIT_SCALE_RANGE)
        return self.log_logit_scales.clamp(low, high).exp()


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


def _offset_centers(device):
    """Centre of each offset bin, in pixels from the cell's top-left corner."""
    width = STRIDE / OFFSET_BINS
    return (torch.arange(OFFSET_BINS, device=device, dtype=torch.float32) + 0.5) * width


def cell_logits(counts, channels, rows, cols, time_steps, scale):
    """Logits of a channel group read at one cell per sample, [B, len(channels)]."""
    batch = torch.arange(counts.shape[0], device=counts.device)
    return counts[batch, channels, rows, cols] * (scale / time_steps)


def _bin_targets(position, count, sigma):
    """Normalized Gaussian over ``count`` bins around a fractional bin ``position``."""
    bins = torch.arange(count, device=position.device, dtype=torch.float32)
    target = torch.exp(-(bins[None, :] - position[:, None]).square() / (2 * sigma ** 2))
    return target / target.sum(dim=1, keepdim=True)


def _soft_cross_entropy(logits, target):
    return -(target * logits.log_softmax(dim=1)).sum(dim=1)


def heatmap_targets(positions):
    """Normalized Gaussian (HEAT_SIGMA cells) over the HEAT x HEAT grid, flattened."""
    centers = (torch.arange(HEAT, device=positions.device, dtype=torch.float32) + 0.5) * STRIDE
    dx = (centers[None, None, :] - positions[:, 0, None, None]) / STRIDE
    dy = (centers[None, :, None] - positions[:, 1, None, None]) / STRIDE
    target = torch.exp(-(dx.square() + dy.square()) / (2 * HEAT_SIGMA ** 2))
    return (target / target.sum(dim=(1, 2), keepdim=True)).flatten(1)


def compute_loss(counts, targets, time_steps, scales):
    """Heatmap cross-entropy plus in-cell offset cross-entropy at the true cell.

    ``scales`` are the (heatmap, offset) logit scales; ``targets[:, 2]`` (the
    measured radius) is not used here.
    """
    positions = targets[:, :2]
    heat_scale, offset_scale = scales
    heat_logits = counts[:, 0].flatten(1) * (heat_scale / time_steps)
    loss = _soft_cross_entropy(heat_logits, heatmap_targets(positions)).mean()

    rows = (positions[:, 1] / STRIDE).long().clamp(0, HEAT - 1)
    cols = (positions[:, 0] / STRIDE).long().clamp(0, HEAT - 1)
    bin_width = STRIDE / OFFSET_BINS
    for channels, offset in (
        (X_OFFSET_CHANNELS, positions[:, 0] - cols * STRIDE),
        (Y_OFFSET_CHANNELS, positions[:, 1] - rows * STRIDE),
    ):
        target = _bin_targets(offset / bin_width - 0.5, OFFSET_BINS, OFFSET_SIGMA)
        logits = cell_logits(counts, channels, rows, cols, time_steps, offset_scale)
        loss = loss + OFFSET_WEIGHT * _soft_cross_entropy(logits, target).mean() / 2
    return loss


def decode(counts, time_steps, scales):
    """(x, y) in sensor pixels per sample.

    Peak cell of the heatmap, plus the expected in-cell offset along x and y
    read at that cell.
    """
    _, offset_scale = scales
    index = counts[:, 0].flatten(1).argmax(dim=1)
    rows, cols = index // HEAT, index % HEAT
    offsets = _offset_centers(counts.device)
    x_probs = cell_logits(
        counts, X_OFFSET_CHANNELS, rows, cols, time_steps, offset_scale,
    ).softmax(dim=1)
    y_probs = cell_logits(
        counts, Y_OFFSET_CHANNELS, rows, cols, time_steps, offset_scale,
    ).softmax(dim=1)
    x = cols * STRIDE + (x_probs * offsets).sum(dim=1)
    y = rows * STRIDE + (y_probs * offsets).sum(dim=1)
    return torch.stack((x, y), dim=1)


def summarize(loss_sum, total, errors, radii):
    """Epoch metrics from per-sample position errors and measured radii (sensor pixels)."""
    errors = torch.cat(errors)
    radii = torch.cat(radii)
    known = radii > 0
    return {
        "loss": loss_sum.item() / total,
        "mean": errors.mean().item(),
        "median": errors.median().item(),
        "within_8px": (errors <= 8.0).float().mean().item(),
        "within_radius": (
            (errors[known] <= radii[known]).float().mean().item() if known.any()
            else float("nan")
        ),
    }


def describe(metrics):
    return (
        f"loss {metrics['loss']:.3f} | error mean {metrics['mean']:.1f} px, "
        f"median {metrics['median']:.1f} px, {metrics['within_8px']:.0%} within 8 px, "
        f"{metrics['within_radius']:.0%} within 1 radius"
    )


def baselines(dataset):
    """Always predicting the frame centre, to read the real metrics against."""
    targets = torch.stack([dataset[i][1] for i in range(len(dataset))])
    center = torch.full_like(targets[:, :2], SENSOR / 2)
    errors = torch.linalg.vector_norm(center - targets[:, :2], dim=1)
    known = targets[:, 2] > 0
    return (
        f"always centre -> median {errors.median().item():.1f} px; "
        f"measured radius median {targets[known, 2].median().item():.1f} px"
    )


def run_epoch(model, loader, device, time_steps, optimizer=None, log_prefix=""):
    training = optimizer is not None
    model.train(training)
    loss_sum = torch.zeros((), device=device)
    errors, radii = [], []
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
            scales = model.logit_scales()
            loss = compute_loss(counts, targets, time_steps, scales)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                with torch.no_grad():
                    for parameter in model.body.parameters():
                        parameter.clamp_(-1.0, 1.0)
            decoded = decode(counts.detach(), time_steps, scales.detach())
            loss_sum += loss.detach() * len(targets)
            errors.append(torch.linalg.vector_norm(decoded - targets[:, :2], dim=1))
            radii.append(targets[:, 2])
            total += len(targets)
    return summarize(loss_sum, total, errors, radii)


def firing_rates(model, loader, device, time_steps, batches=2):
    """Mean spike probability per neuron and time step for every layer."""
    rates, hooks = {}, []
    for name, module in model.body.named_children():
        if isinstance(module, SpikingConvBN2d):
            hooks.append(module.register_forward_hook(
                lambda _module, _inputs, output, name=name:
                rates.setdefault(name, []).append(output.float().mean())
            ))
    model.eval()
    try:
        with torch.no_grad():
            for step, (flat, targets) in enumerate(loader):
                if step >= batches:
                    break
                model(fast.to_frames(flat, len(targets), device, time_steps))
    finally:
        for hook in hooks:
            hook.remove()
    return {name: torch.stack(values).mean().item() for name, values in rates.items()}


def describe_rates(rates):
    silent = [name for name, rate in rates.items() if rate < SILENT_RATE]
    text = ", ".join(f"{name} {rate:.1%}" for name, rate in rates.items())
    return f"Firing rates: {text}" + (f" | SILENT: {', '.join(silent)}" if silent else "")


def infer(model, dataset, device, batch_size, time_steps):
    """Predicted (x, y) in sensor pixels, shape (N, 2), in dataset order."""
    predicted = []
    model.eval()
    with torch.no_grad():
        scales = model.logit_scales()
        for flat, targets in fast.make_loader(dataset, batch_size):
            frames = fast.to_frames(flat, len(targets), device, time_steps)
            predicted.append(decode(model(frames), time_steps, scales).cpu().numpy())
    if not predicted:
        return np.zeros((0, 2), dtype=np.float32)
    return np.concatenate(predicted, axis=0)


def train(batch_size, epochs, time_steps, learning_rate, min_learning_rate):
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
    print("Baselines on validation:", baselines(validation_dataset), flush=True)

    example_frames = base.save_sample_figure(
        [fast.dense_sample(validation_dataset, 0, time_steps)], Path("ball_size_sample.png"),
    )

    model = BallDetector().to(device)
    model.eval()
    with torch.no_grad():
        counts = model(example_frames.unsqueeze(0).to(device))
    if tuple(counts.shape[1:]) != (HEAD_CHANNELS, HEAT, HEAT):
        raise RuntimeError(f"Unexpected output shape {tuple(counts.shape)}")
    print("Output shape:", tuple(counts.shape), flush=True)

    optimizer = torch.optim.Adam([
        {"params": model.body.parameters(), "weight_decay": base.WEIGHT_DECAY},
        {"params": [model.log_logit_scales], "weight_decay": 0.0},
    ], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=min_learning_rate,
    )
    print(f"Learning rate: {learning_rate:g} -> {min_learning_rate:g} (cosine over {epochs} epochs)")
    best_validation_error = float("inf")

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs} training, lr {scheduler.get_last_lr()[0]:.2e}", flush=True)
        train_metrics = run_epoch(
            model, train_loader, device, time_steps, optimizer,
            log_prefix=f"Epoch {epoch} train",
        )
        scheduler.step()
        print(f"Epoch {epoch}/{epochs} validation", flush=True)
        validation_metrics = run_epoch(
            model, validation_loader, device, time_steps,
            log_prefix=f"Epoch {epoch} validation",
        )
        validation_error = validation_metrics["median"]
        saved = validation_error < best_validation_error
        if saved:
            best_validation_error = validation_error
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "validation_error": validation_error,
                "validation_metrics": validation_metrics,
                "sensor": SENSOR,
                "crop_sides": fast.CROP_SIDES,
                "time_steps": time_steps,
                "heatmap_stride": STRIDE,
                "offset_bins": OFFSET_BINS,
                "architecture": ARCHITECTURE,
            }, CHECKPOINT)
        scales = model.logit_scales().tolist()
        print(f"Epoch {epoch}/{epochs} | Training   {describe(train_metrics)}")
        print(
            f"Epoch {epoch}/{epochs} | Validation {describe(validation_metrics)}"
            + ("  [saved]" if saved else "")
        )
        print(
            f"Epoch {epoch}/{epochs} | Logit scales: heatmap {scales[0]:.1f}, "
            f"offset {scales[1]:.1f} | "
            + describe_rates(firing_rates(model, validation_loader, device, time_steps)),
            flush=True,
        )

    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Test", flush=True)
    test_metrics = run_epoch(
        model, fast.make_loader(test_dataset, batch_size), device, time_steps,
        log_prefix="Test",
    )
    print("Best epoch:", checkpoint["epoch"])
    print(f"Best validation median error: {checkpoint['validation_error']:.2f} px")
    print(f"Test {describe(test_metrics)} (sensor pixels)")

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
    model = BallDetector().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    time_steps = int(checkpoint["time_steps"])
    print("Time steps:", time_steps)
    return model, time_steps


def export_checkpoint(checkpoint_path):
    """Export a saved checkpoint (by default the best one) without retraining."""
    device = base.pick_device()
    model, time_steps = load_model(checkpoint_path, device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    print(
        f"Exporting {checkpoint_path}: epoch {checkpoint['epoch']}, "
        f"validation median error {checkpoint['validation_error']:.2f} px"
    )
    export_model(model, torch.zeros(time_steps, 1, SENSOR, SENSOR), device)


def measured_radii(dataset, predicted):
    """Circle radius per frame, in sensor pixels, measured from the events.

    Labeled: the label radius (measured around the label, scaled with the
    crop). Unlabeled: measured around the prediction in each frame, then
    median-smoothed over neighbouring frames like the labels.
    """
    if dataset.labeled:
        return np.array([float(dataset[i][1][2]) for i in range(len(dataset))])
    raw = []
    for index in range(len(dataset)):
        flat = dataset[index][0].numpy()
        pixels = np.stack((flat // SENSOR, flat % SENSOR), axis=1)
        raw.append(window_radius(pixels, predicted[index]))
    return smooth_radii(np.array(raw, dtype=np.float64), [len(raw)])


def write_video(model, device, dataset, name, output_path, fps, batch_size, time_steps):
    """Render every window with its predicted (and, if labeled, true) position.

    Both circles are drawn with the radius measured from the events.
    """
    predicted = infer(model, dataset, device, batch_size, time_steps)
    radii = measured_radii(dataset, predicted)
    writer = fast.VideoWriter(output_path, fps, SENSOR * fast.DISPLAY_SCALE)
    errors, hits = [], []
    try:
        for index in range(len(dataset)):
            flat, target = dataset[index]
            px, py = (float(v) for v in predicted[index])
            radius = float(radii[index])
            caption = (
                f"{name}  {index + 1}/{len(dataset)}  pred ({px:.0f}, {py:.0f})  r {radius:.1f}"
            )
            truth = None
            if dataset.labeled:
                tx, ty = target[0].item(), target[1].item()
                truth = (tx, ty)
                error = float(np.hypot(px - tx, py - ty))
                errors.append(error)
                hits.append(error <= radius)
                caption += f"  gt ({tx:.0f}, {ty:.0f})  err {error:.1f}"
            writer.write(fast.render_frame(
                flat, (px, py), truth, caption,
                prediction_radius=radius, truth_radius=radius,
            ))
    finally:
        writer.close()
    print(f"Wrote {writer.frames} frames to {Path(output_path).resolve()}")
    if errors:
        print(
            f"Mean error: {float(np.mean(errors)):.2f} px, median {float(np.median(errors)):.2f} px, "
            f"{float(np.mean(hits)):.0%} within 1 radius (sensor pixels)"
        )


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
    parser = argparse.ArgumentParser(description="Spiking U-Net ball detector, 320x320.")
    parser.add_argument("--batch-size", type=int, default=fast.BATCH_SIZE)
    subparsers = parser.add_subparsers(dest="command")
    train_parser = subparsers.add_parser("train", help="train on the benchmark and export")
    train_parser.add_argument("--epochs", type=int, default=base.EPOCHS)
    train_parser.add_argument("--time-steps", type=int, default=fast.TIME_STEPS)
    train_parser.add_argument("--lr", type=float, default=base.LEARNING_RATE,
                              help="start of the cosine schedule")
    train_parser.add_argument("--min-lr", type=float, default=MIN_LEARNING_RATE,
                              help="end of the cosine schedule (reached at the last epoch)")
    export = subparsers.add_parser("export", help="export a checkpoint (default: the best)")
    export.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
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
    if args.command == "export":
        export_checkpoint(args.checkpoint)
    elif args.command == "visualize":
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
            getattr(args, "lr", base.LEARNING_RATE),
            getattr(args, "min_lr", MIN_LEARNING_RATE),
        )


if __name__ == "__main__":
    main()
