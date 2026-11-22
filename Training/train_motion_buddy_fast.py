"""Heatmap variant of ``train_motion_buddy.py`` at the 320x320 camera resolution.

Same benchmark data and splits as the original. Changes:

1. Every sample is a SENSOR x SENSOR (320x320) frame, the resolution of our
   camera recordings (``reflex-events-v1`` files). Training takes a random
   square crop of the 1280x720 benchmark frame that contains the ball, with a
   side drawn log-uniformly from CROP_SIDES, and resamples it to 320x320.
   Validation and test crops are fixed per sample. The varying side doubles as
   scale augmentation, since our camera's lens and ball size are unknown:
   sides below 320 px zoom in (the ball looks up to 4x bigger), sides above
   zoom out. A zoomed-in pixel fills its footprint on the sensor, but each
   filled pixel is only kept with probability UPSAMPLE_KEEP, so enlarged
   objects are not unrealistically solid.
2. The dataset yields sparse pixel indices. The dense frame is built on the
   device and repeated over time with ``expand`` instead of real copies.
3. Larger batches, and loss and error are accumulated on the device.
4. 4 time steps instead of 8 (``--time-steps``). The same frame is fed at
   every step; T only sets the spike-count resolution and the cost.
5. Fully convolutional spiking network. The last layer is a 40x40 map of
   spiking neurons, one per 8x8 px cell; its spike counts are the ball
   heatmap. Training uses cross-entropy over the cells against a Gaussian
   around the true position. Decoding (argmax cell, then a spike-weighted
   centroid over its 3x3 neighbourhood) runs off-chip. All kernels and strides
   are square, as the RIVER exporter requires.

Checkpoints are not interchangeable with the original. The export still writes
``ball_model.json`` and ``ball_parameters.{json,txt}``::

    python train_motion_buddy_fast.py
    python train_motion_buddy_fast.py visualize
    python train_motion_buddy_fast.py predict data/recording.h5
"""

import argparse
import subprocess
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset

import train_motion_buddy as base
from riverops.nn import SpikingConvBN2d

BATCH_SIZE = 64
TIME_STEPS = 4
CHECKPOINT = Path("ball_heatmap_best.pt")
SENSOR = 320
INPUT_CHANNELS = 1
SOURCE_WIDTH = base.CAMERA_WIDTH
SOURCE_HEIGHT = base.CAMERA_HEIGHT
CROP_SIDES = (80.0, 720.0)
CROP_MARGIN = 16.0
UPSAMPLE_KEEP = 0.75
HEATMAP_STRIDE = 8
HEAT = SENSOR // HEATMAP_STRIDE
TARGET_SIGMA = 1.0
LOGIT_SCALE = 16.0
DISPLAY_SCALE = 2


def _crop_origin(rng, center, side, limit):
    """Crop start along one axis so ``center`` lies inside, CROP_MARGIN from the edge."""
    low = max(0.0, center - side + CROP_MARGIN)
    high = min(limit - side, center - CROP_MARGIN)
    if low < high:
        return float(rng.uniform(low, high))
    return float(np.clip(center - side / 2, 0.0, limit - side))


def random_crop(rng):
    """New crop for every call: (x0, y0, side) in benchmark pixels."""
    def crop(index, xy):
        side = float(np.exp(rng.uniform(*np.log(CROP_SIDES))))
        return (
            _crop_origin(rng, xy[0], side, SOURCE_WIDTH),
            _crop_origin(rng, xy[1], side, SOURCE_HEIGHT),
            side,
        )
    return crop


def fixed_crop(seed):
    """Same random crop for a given index on every call."""
    def crop(index, xy):
        return random_crop(np.random.default_rng((seed, index)))(index, xy)
    return crop


def fit_crop(width, height):
    """Whole width x height frame, centered and scaled to fit the sensor."""
    side = float(max(width, height))
    return lambda index, xy: ((width - side) / 2, (height - side) / 2, side)


def resample(pixels, x0, y0, scale, rng=None, keep=UPSAMPLE_KEEP):
    """(y, x, p) source pixels -> unique flat indices into [INPUT_CHANNELS, SENSOR, SENSOR].

    With INPUT_CHANNELS = 2, OFF events go to channel 0 and ON events to
    channel 1; with 1, polarity is merged. Zooming out maps each pixel to one
    sensor pixel. Zooming in fills the sensor pixels a source pixel covers, so
    objects are not dotted; with an ``rng``, each filled pixel is kept with
    probability ``keep``.
    """
    x = (pixels[:, 1].astype(np.float32) - x0) * scale
    y = (pixels[:, 0].astype(np.float32) - y0) * scale
    channel = (
        pixels[:, 2].astype(np.int64) if INPUT_CHANNELS > 1
        else np.zeros(len(pixels), dtype=np.int64)
    )
    if scale <= 1.0:
        xs, ys, cs = np.floor(x), np.floor(y), channel
    else:
        step = np.arange(int(np.ceil(scale)) + 1, dtype=np.float32)
        xs = np.floor(x)[:, None, None] + step[None, None, :]
        ys = np.floor(y)[:, None, None] + step[None, :, None]
        inside = (xs < np.ceil(x + scale)[:, None, None]) & (ys < np.ceil(y + scale)[:, None, None])
        xs = np.broadcast_to(xs, inside.shape)[inside]
        ys = np.broadcast_to(ys, inside.shape)[inside]
        cs = np.broadcast_to(channel[:, None, None], inside.shape)[inside]
    valid = (xs >= 0) & (xs < SENSOR) & (ys >= 0) & (ys < SENSOR)
    flat = np.unique(
        (cs[valid] * SENSOR + ys[valid].astype(np.int64)) * SENSOR + xs[valid].astype(np.int64)
    )
    if scale > 1.0 and rng is not None and keep < 1.0:
        flat = flat[rng.random(len(flat)) < keep]
    return flat


class SensorDataset(Dataset):
    """Windows of (y, x) source pixels resampled into the SENSOR x SENSOR frame.

    ``crop(index, xy)`` returns the square (x0, y0, side), in source pixels,
    that is mapped onto the sensor. Items are (flat indices y * SENSOR + x,
    position in sensor pixels). ``positions=None`` means unlabeled. Without
    ``noise_seed`` the upsampling dropout changes on every call (training);
    with it, it is fixed per index (evaluation).
    """

    def __init__(self, events, offsets, positions, crop, noise_seed=None):
        self.noise_seed = noise_seed
        self.rng = np.random.default_rng(base.SEED)
        self.events = np.ascontiguousarray(events)
        self.offsets = np.ascontiguousarray(offsets)
        self.labeled = positions is not None
        count = len(self.offsets) - 1
        self.positions = (
            np.ascontiguousarray(positions, dtype=np.float32) if self.labeled
            else np.zeros((count, 2), dtype=np.float32)
        )
        self.crop = crop

    def __len__(self):
        return int(self.positions.shape[0])

    def noise_rng(self, index):
        if self.noise_seed is None:
            return self.rng
        return np.random.default_rng((self.noise_seed, index))

    def __getitem__(self, index):
        xy = self.positions[index]
        x0, y0, side = self.crop(index, xy)
        scale = SENSOR / side
        pixels = self.events[self.offsets[index]:self.offsets[index + 1]]
        flat = resample(pixels, x0, y0, scale, self.noise_rng(index))
        position = np.array(((xy[0] - x0) * scale, (xy[1] - y0) * scale), dtype=np.float32)
        return torch.from_numpy(flat), torch.from_numpy(position)


def collate(batch):
    pixels, positions = zip(*batch)
    flat = torch.cat([p + i * INPUT_CHANNELS * SENSOR * SENSOR for i, p in enumerate(pixels)])
    return flat, torch.stack(positions)


def make_loader(dataset, batch_size, shuffle=False):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
        collate_fn=collate, pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(base.SEED) if shuffle else None,
    )


def to_frames(flat, batch_size, device, time_steps):
    """Scatter flat indices into [B, T, INPUT_CHANNELS, SENSOR, SENSOR] on the device; T is a view."""
    frames = torch.zeros(batch_size * INPUT_CHANNELS * SENSOR * SENSOR, device=device)
    frames[flat.to(device, non_blocking=True)] = 1.0
    return frames.view(batch_size, 1, INPUT_CHANNELS, SENSOR, SENSOR).expand(
        -1, time_steps, -1, -1, -1,
    )


def dense_sample(dataset, index, time_steps):
    flat, position = dataset[index]
    frame = torch.zeros(INPUT_CHANNELS * SENSOR * SENSOR)
    frame[flat] = 1.0
    return frame.view(INPUT_CHANNELS, SENSOR, SENSOR).repeat(time_steps, 1, 1, 1), position


class HeatmapBallDetector(nn.Module):
    """Spatial sizes: 320x320 -> 80x80 -> 40x40 (x4 layers), one output channel."""

    def __init__(self):
        super().__init__()
        lif = {
            "initial_tau": 1.0 / (1.0 - 0.93),
            "learnable_leak": False,
            "v_threshold": 1.0,
            "reset_mode": "HARD",
        }
        self.net = nn.Sequential(
            SpikingConvBN2d(1, 16, kernel_size=8, stride=4, padding=2,
                            bias=False, first_layer=True, **lif),
            SpikingConvBN2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False, **lif),
            SpikingConvBN2d(32, 32, kernel_size=3, padding=1, bias=False, **lif),
            SpikingConvBN2d(32, 32, kernel_size=3, padding=1, bias=False, **lif),
            SpikingConvBN2d(32, 32, kernel_size=3, padding=1, bias=False, **lif),
            SpikingConvBN2d(32, 1, kernel_size=1, bias=False, **lif),
        )

    def forward(self, x):
        """Spike counts per heatmap cell, [B, HEAT, HEAT]."""
        return self.net(x).sum(dim=1)[:, 0]


def _cell_centers(device):
    return (torch.arange(HEAT, device=device, dtype=torch.float32) + 0.5) * HEATMAP_STRIDE


def heatmap_targets(positions):
    """Normalized Gaussian over the cells around each (x, y), flattened."""
    centers = _cell_centers(positions.device)
    dx = (centers[None, None, :] - positions[:, 0, None, None]) / HEATMAP_STRIDE
    dy = (centers[None, :, None] - positions[:, 1, None, None]) / HEATMAP_STRIDE
    target = torch.exp(-(dx.square() + dy.square()) / (2 * TARGET_SIGMA ** 2))
    return (target / target.sum(dim=(1, 2), keepdim=True)).flatten(1)


def compute_loss(counts, positions, time_steps):
    logits = counts.flatten(1) * (LOGIT_SCALE / time_steps)
    return -(heatmap_targets(positions) * logits.log_softmax(dim=1)).sum(dim=1).mean()


def decode_positions(counts):
    """Argmax cell, refined by the spike-weighted centroid of its 3x3 neighbours."""
    batch = counts.shape[0]
    index = counts.flatten(1).argmax(dim=1)
    offsets = torch.arange(-1, 2, device=counts.device)
    rows = ((index // HEAT)[:, None] + offsets).clamp(0, HEAT - 1)
    cols = ((index % HEAT)[:, None] + offsets).clamp(0, HEAT - 1)
    batch_index = torch.arange(batch, device=counts.device)[:, None, None]
    weights = counts[batch_index, rows[:, :, None], cols[:, None, :]] + 1e-6
    centers = _cell_centers(counts.device)
    y = (centers[rows][:, :, None] * weights).sum(dim=(1, 2)) / weights.sum(dim=(1, 2))
    x = (centers[cols][:, None, :] * weights).sum(dim=(1, 2)) / weights.sum(dim=(1, 2))
    return torch.stack((x, y), dim=1)


def position_error(counts, positions):
    return torch.linalg.vector_norm(decode_positions(counts) - positions, dim=1)


def run_epoch(model, loader, device, time_steps, optimizer=None, log_prefix=""):
    training = optimizer is not None
    model.train(training)
    total_loss = torch.zeros((), device=device)
    total_error = torch.zeros((), device=device)
    total = 0
    batch_count = len(loader)
    with torch.set_grad_enabled(training):
        for step, (flat, positions) in enumerate(loader, start=1):
            if step == 1 or step % 10 == 0 or step == batch_count:
                print(f"{log_prefix} batch {step}/{batch_count}", flush=True)
            positions = positions.to(device, non_blocking=True)
            frames = to_frames(flat, len(positions), device, time_steps)
            if training:
                optimizer.zero_grad()
            counts = model(frames)
            loss = compute_loss(counts, positions, time_steps)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.clamp_(-1.0, 1.0)
            count = len(positions)
            total_loss += loss.detach() * count
            total_error += position_error(counts.detach(), positions).sum()
            total += count
    return total_loss.item() / total, total_error.item() / total


def infer_positions(model, dataset, device, batch_size, time_steps):
    """Return predicted sensor pixels, shape (N, 2), in dataset order."""
    predicted = []
    model.eval()
    with torch.no_grad():
        for flat, positions in make_loader(dataset, batch_size):
            frames = to_frames(flat, len(positions), device, time_steps)
            predicted.append(decode_positions(model(frames)).cpu().numpy())
    if not predicted:
        return np.zeros((0, 2), dtype=np.float32)
    return np.concatenate(predicted, axis=0)


def benchmark_splits():
    """Train, validation and test SensorDatasets over the benchmark."""
    train_arrays, validation_arrays, test_arrays = base._read_splits(base.ensure_dataset())
    return (
        SensorDataset(*train_arrays, random_crop(np.random.default_rng(base.SEED))),
        SensorDataset(*validation_arrays, fixed_crop(base.SEED + 1), noise_seed=base.SEED + 1),
        SensorDataset(*test_arrays, fixed_crop(base.SEED + 2), noise_seed=base.SEED + 2),
    )


def train(batch_size, epochs, time_steps):
    torch.manual_seed(base.SEED)
    torch.backends.cudnn.benchmark = True
    device = base.pick_device()
    print("Device:", device, flush=True)

    train_dataset, validation_dataset, test_dataset = benchmark_splits()
    train_loader = make_loader(train_dataset, batch_size, shuffle=True)
    validation_loader = make_loader(validation_dataset, batch_size)
    print("Training frames:", len(train_dataset))
    print("Validation frames:", len(validation_dataset))
    print(f"Sensor: {SENSOR}x{SENSOR}, crop sides {CROP_SIDES[0]:.0f}-{CROP_SIDES[1]:.0f} px")
    print("Batch size:", batch_size)
    print("Time steps:", time_steps)

    example_frames = base.save_sample_figure(
        [dense_sample(validation_dataset, 0, time_steps)], Path("ball_sample.png"),
    )

    model = HeatmapBallDetector().to(device)
    model.eval()
    print("Running one forward pass", flush=True)
    with torch.no_grad():
        counts = model(example_frames.unsqueeze(0).to(device))
    if tuple(counts.shape[1:]) != (HEAT, HEAT):
        raise RuntimeError(f"Unexpected output shape {tuple(counts.shape)}")
    prediction = decode_positions(counts)[0].cpu().tolist()
    print("Output shape:", tuple(counts.shape), flush=True)
    print(f"Sample prediction (x, y): ({prediction[0]:.0f}, {prediction[1]:.0f})", flush=True)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=base.LEARNING_RATE, weight_decay=base.WEIGHT_DECAY,
    )
    best_validation_error = float("inf")

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs} training", flush=True)
        train_loss, train_error = run_epoch(
            model, train_loader, device, time_steps, optimizer,
            log_prefix=f"Epoch {epoch} train",
        )
        print(f"Epoch {epoch}/{epochs} validation", flush=True)
        validation_loss, validation_error = run_epoch(
            model, validation_loader, device, time_steps,
            log_prefix=f"Epoch {epoch} validation",
        )
        if validation_error < best_validation_error:
            best_validation_error = validation_error
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "validation_error": validation_error,
                "sensor": SENSOR,
                "crop_sides": CROP_SIDES,
                "time_steps": time_steps,
                "heatmap_stride": HEATMAP_STRIDE,
            }, CHECKPOINT)
        print(
            f"Epoch {epoch}/{epochs} | Training loss: {train_loss:.4f} | "
            f"Training error: {train_error:.2f} px | "
            f"Validation loss: {validation_loss:.4f} | "
            f"Validation error: {validation_error:.2f} px"
        )

    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Test", flush=True)
    test_loss, test_error = run_epoch(
        model, make_loader(test_dataset, batch_size), device, time_steps, log_prefix="Test",
    )
    print("Best epoch:", checkpoint["epoch"])
    print(f"Best validation error: {checkpoint['validation_error']:.2f} px")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test error: {test_error:.2f} px (sensor pixels)")

    base.export_model(model, example_frames, device)


def load_model(checkpoint_path, device):
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path}. Train first with: "
            "python train_motion_buddy_fast.py"
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("sensor") != SENSOR:
        raise ValueError(
            f"{checkpoint_path} was not trained at {SENSOR}x{SENSOR}; retrain."
        )
    model = HeatmapBallDetector().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    time_steps = int(checkpoint["time_steps"])
    print("Time steps:", time_steps)
    return model, time_steps


def check_hdf5(path):
    """Fail early, with a hint, when ``path`` is not a readable HDF5 file."""
    if not path.is_file():
        raise FileNotFoundError(f"No recording at {path}")
    if h5py.is_hdf5(path):
        return
    with path.open("rb") as handle:
        head = handle.read(64)
    hints = {
        b"version https://git-lfs": "it is a Git LFS pointer; run `git lfs pull` in its repo",
        b"% evt": "it looks like a Prophesee .raw file",
        b"% format": "it looks like a Prophesee .raw file",
        b"#!AER-DAT": "it looks like an iniVation AEDAT file",
        b"PK": "it looks like a zip or .npz archive",
    }
    hint = next(
        (text for prefix, text in hints.items() if head.startswith(prefix)),
        f"first bytes: {head[:16]!r}",
    )
    raise ValueError(
        f"{path} ({path.stat().st_size} bytes) is not an HDF5 file: {hint}."
    )


def read_recording(path):
    """Return (x, y, t, p, width, height, labels) from either supported layout.

    ``reflex-events-v1``: ``events/{x,y,t,p}`` with ``width``/``height`` attributes.
    Benchmark: ``event_{x,y,t,p}`` on a 1280x720 sensor, optional ``gt_{x,y,t}``.
    Times are in µs. ``labels`` is (gt_x, gt_y, gt_t) or None.
    """
    with h5py.File(path, "r") as handle:
        if isinstance(handle.get("events"), h5py.Group):
            unit = handle.attrs.get("t_unit", "us")
            if unit != "us":
                raise ValueError(f"{path}: t_unit is {unit!r}, expected 'us'")
            group = handle["events"]
            x, y, t, p = (group[key][()] for key in "xytp")
            width = int(handle.attrs.get("width", int(x.max()) + 1))
            height = int(handle.attrs.get("height", int(y.max()) + 1))
            return x, y, t, p, width, height, None
        missing = {"event_x", "event_y", "event_t", "event_p"} - set(handle.keys())
        if missing:
            raise KeyError(
                f"{path}: neither events/{{x,y,t,p}} nor {', '.join(sorted(missing))}"
            )
        x, y, t, p = (base._column(handle, f"event_{key}") for key in "xytp")
        labels = None
        if {"gt_x", "gt_y", "gt_t"} <= set(handle.keys()):
            labels = tuple(base._column(handle, f"gt_{key}").astype(np.float64) for key in "xyt")
        return x, y, t, p, SOURCE_WIDTH, SOURCE_HEIGHT, labels


def recording_dataset(path, step_us, window_us, use_labels=True):
    """One window per label, or one every ``step_us``, fitted to the sensor."""
    x, y, t, p, width, height, labels = read_recording(path)
    if t.size == 0:
        raise ValueError(f"{path} has no events")
    if width > SOURCE_WIDTH or height > SOURCE_HEIGHT:
        raise ValueError(f"{path}: {width}x{height} sensors are not supported")
    t = t.astype(np.float64)
    order = np.argsort(t, kind="mergesort")
    x, y, t, p = x[order], y[order], t[order], p[order]
    xy = None
    if use_labels and labels is not None:
        gx, gy, gt = labels
        visible = (gx >= 0) & (gy >= 0) & (gx < width) & (gy < height)
        order = np.argsort(gt[visible], kind="mergesort")
        ends = gt[visible][order]
        xy = np.stack((gx[visible][order], gy[visible][order]), axis=1)
    else:
        ends = np.arange(t[0] + window_us, t[-1] + step_us, step_us)
    starts = np.searchsorted(t, ends - window_us, side="left")
    stops = np.searchsorted(t, ends, side="right")
    pixels = [base._window_pixels(x[a:b], y[a:b], p[a:b]) for a, b in zip(starts, stops)]
    offsets = np.zeros(len(pixels) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(window) for window in pixels])
    events = np.concatenate(pixels, axis=0) if offsets[-1] else np.zeros((0, 3), np.uint16)
    print(
        f"{path.name}: {width}x{height}, {len(pixels)} windows of "
        f"{window_us / 1000:g} ms, median {int(np.median(np.diff(offsets)))} px per window",
        flush=True,
    )
    return SensorDataset(events, offsets, xy, fit_crop(width, height), noise_seed=0)


class VideoWriter(base.VideoWriter):
    """``base.VideoWriter`` for square frames of any size."""

    def __init__(self, path, fps, size):
        self.path = Path(path)
        self.proc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{size}x{size}",
                "-r", str(fps),
                "-i", "-",
                "-an",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(self.path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.frames = 0
        self._closed = False


def render_frame(flat, prediction, truth, caption, prediction_radius=6.0, truth_radius=6.0):
    """Sensor frame, upscaled; red prediction ring and, if given, green truth ring.

    Radii are in sensor pixels. ``prediction=None`` draws no red ring.
    """
    image = np.zeros((SENSOR * SENSOR, 3), dtype=np.uint8)
    image[flat.numpy() % (SENSOR * SENSOR)] = 210
    image = image.reshape(SENSOR, SENSOR, 3)
    image = image.repeat(DISPLAY_SCALE, axis=0).repeat(DISPLAY_SCALE, axis=1)
    picture = Image.fromarray(image)
    draw = ImageDraw.Draw(picture)
    size = SENSOR * DISPLAY_SCALE

    def ring(xy, radius, color):
        x, y = float(xy[0]) * DISPLAY_SCALE, float(xy[1]) * DISPLAY_SCALE
        radius = max(float(radius), 2.0) * DISPLAY_SCALE
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=3)

    if truth is not None:
        ring(truth, truth_radius, (80, 220, 80))
    if prediction is not None:
        ring(prediction, prediction_radius, (240, 60, 60))
    draw.rectangle((0, 0, size, 28), fill=(0, 0, 0))
    draw.text((8, 6), caption, fill=(255, 255, 255))
    legend = "green ground truth    red prediction" if truth is not None else "red prediction"
    draw.text((8, size - 22), legend, fill=(255, 255, 255))
    return np.asarray(picture)


def write_video(model, device, dataset, name, output_path, fps, batch_size, time_steps):
    """Render every window of ``dataset`` with its prediction into an mp4."""
    predicted = infer_positions(model, dataset, device, batch_size, time_steps)
    writer = VideoWriter(output_path, fps, SENSOR * DISPLAY_SCALE)
    errors = []
    try:
        for index in range(len(dataset)):
            flat, position = dataset[index]
            caption = (
                f"{name}  {index + 1}/{len(dataset)}  "
                f"pred ({predicted[index, 0]:.0f}, {predicted[index, 1]:.0f})"
            )
            truth = position.numpy() if dataset.labeled else None
            if truth is not None:
                error = float(np.linalg.norm(predicted[index] - truth))
                errors.append(error)
                caption += f"  gt ({truth[0]:.0f}, {truth[1]:.0f})  err {error:.1f}"
            writer.write(render_frame(flat, predicted[index], truth, caption))
    finally:
        writer.close()
    print(f"Wrote {writer.frames} frames to {Path(output_path).resolve()}")
    if errors:
        print(f"Mean error: {float(np.mean(errors)):.2f} px (sensor pixels)")


def visualize_test(checkpoint_path, output_path, fps, batch_size):
    """Run the checkpoint on the fixed test crops (takes 7 and 8) and write an mp4."""
    device = base.pick_device()
    print("Device:", device, flush=True)
    model, time_steps = load_model(checkpoint_path, device)
    test_dataset = benchmark_splits()[2]
    write_video(model, device, test_dataset, "test 7-8", output_path, fps,
                batch_size, time_steps)


def predict_recording(recording, checkpoint_path, output_path, fps, batch_size,
                      step_ms, window_ms, use_labels):
    """Run the checkpoint on one HDF5 event file and write an mp4."""
    check_hdf5(recording)
    device = base.pick_device()
    print("Device:", device, flush=True)
    model, time_steps = load_model(checkpoint_path, device)
    dataset = recording_dataset(recording, step_ms * 1000.0, window_ms * 1000.0, use_labels)
    output_path = output_path or Path(f"{recording.stem}_pred.mp4")
    write_video(model, device, dataset, recording.name, output_path, fps,
                batch_size, time_steps)


def main():
    parser = argparse.ArgumentParser(description="Heatmap spiking ball detector, 320x320.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    subparsers = parser.add_subparsers(dest="command")
    train_parser = subparsers.add_parser("train", help="train on the benchmark and export")
    train_parser.add_argument("--epochs", type=int, default=base.EPOCHS)
    train_parser.add_argument("--time-steps", type=int, default=TIME_STEPS)
    view = subparsers.add_parser("visualize", help="write a test-set video")
    view.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    view.add_argument("--output", type=Path, default=Path("ball_heatmap_test.mp4"))
    view.add_argument("--fps", type=int, default=15)
    predict = subparsers.add_parser(
        "predict", help="write a prediction video for an HDF5 event file",
    )
    predict.add_argument("recording", type=Path)
    predict.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    predict.add_argument("--output", type=Path, default=None,
                         help="default: <recording name>_pred.mp4 in the current folder")
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
            getattr(args, "time_steps", TIME_STEPS),
        )


if __name__ == "__main__":
    main()
