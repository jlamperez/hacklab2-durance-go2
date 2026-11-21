"""Heatmap variant of ``train_motion_buddy.py``: same data, splits and export.

Changes:

1. The dataset yields sparse pixel indices. The dense frame is built on the
   device and repeated over time with ``expand`` instead of real copies.
2. The input is the 720x1280 sensor, not a zero-padded 1280x1280 square.
3. Larger batches, and loss and error are accumulated on the device, so there
   is no host sync per batch.
4. 4 time steps instead of 8 (``--time-steps``). The same frame is still fed at
   every step; T only sets the spike-count resolution and the cost.
5. Fully convolutional spiking network. The last layer is a 45x80 map of
   spiking neurons, one per 16x16 px cell; its spike counts are the ball
   heatmap. Training uses cross-entropy over the cells against a Gaussian
   around the true position. Decoding (argmax cell, then a spike-weighted
   centroid over its 3x3 neighbourhood) runs off-chip. All kernels and strides
   are square, as the RIVER exporter requires.

Checkpoints are not interchangeable with the original. The export still writes
``ball_model.json`` and ``ball_parameters.{json,txt}``::

    python train_motion_buddy_fast.py
    python train_motion_buddy_fast.py visualize
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import train_motion_buddy as base
from riverops.nn import SpikingConvBN2d

BATCH_SIZE = 16
TIME_STEPS = 4
CHECKPOINT = Path("ball_heatmap_best.pt")
HEIGHT = base.CAMERA_HEIGHT
WIDTH = base.CAMERA_WIDTH
HEATMAP_STRIDE = 16
HEAT_H = HEIGHT // HEATMAP_STRIDE
HEAT_W = WIDTH // HEATMAP_STRIDE
TARGET_SIGMA = 1.0
LOGIT_SCALE = 16.0


class SparseBallDataset(Dataset):
    """Yields the flat sensor indices (y * WIDTH + x) hit in each 1 ms window."""

    def __init__(self, events, offsets, positions):
        flat = events[:, 0].astype(np.int32) * WIDTH + events[:, 1].astype(np.int32)
        self.flat = torch.from_numpy(flat)
        self.offsets = np.ascontiguousarray(offsets)
        self.positions = torch.from_numpy(np.ascontiguousarray(positions)).to(torch.float32)

    def __len__(self):
        return int(self.positions.shape[0])

    def __getitem__(self, index):
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        return self.flat[start:end], self.positions[index]


def collate(batch):
    pixels, positions = zip(*batch)
    flat = torch.cat([
        p.to(torch.int64) + i * HEIGHT * WIDTH for i, p in enumerate(pixels)
    ])
    return flat, torch.stack(positions)


def make_loader(dataset, batch_size, shuffle=False):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
        collate_fn=collate, pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(base.SEED) if shuffle else None,
    )


def to_frames(flat, batch_size, device, time_steps):
    """Scatter flat indices into [B, T, 1, H, W] on the device; T is a view."""
    frames = torch.zeros(batch_size * HEIGHT * WIDTH, device=device)
    frames[flat.to(device, non_blocking=True)] = 1.0
    return frames.view(batch_size, 1, 1, HEIGHT, WIDTH).expand(
        -1, time_steps, -1, -1, -1,
    )


def dense_sample(dataset, index, time_steps):
    flat, position = dataset[index]
    frame = torch.zeros(HEIGHT * WIDTH)
    frame[flat.to(torch.int64)] = 1.0
    return frame.view(1, HEIGHT, WIDTH).repeat(time_steps, 1, 1, 1), position


class HeatmapBallDetector(nn.Module):
    """Spatial sizes (H x W): 720x1280 -> 180x320 -> 90x160 -> 45x80 (x3 layers)."""

    def __init__(self):
        super().__init__()
        lif = {
            "initial_tau": 1.0 / (1.0 - 0.93),
            "learnable_leak": False,
            "v_threshold": 1.0,
            "reset_mode": "HARD",
        }
        self.net = nn.Sequential(
            SpikingConvBN2d(1, 8, kernel_size=8, stride=4, padding=2,
                            bias=False, first_layer=True, **lif),
            SpikingConvBN2d(8, 16, kernel_size=4, stride=2, padding=1, bias=False, **lif),
            SpikingConvBN2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False, **lif),
            SpikingConvBN2d(32, 32, kernel_size=3, padding=1, bias=False, **lif),
            SpikingConvBN2d(32, 32, kernel_size=3, padding=1, bias=False, **lif),
            SpikingConvBN2d(32, 1, kernel_size=1, bias=False, **lif),
        )

    def forward(self, x):
        """Spike counts per heatmap cell, [B, HEAT_H, HEAT_W]."""
        return self.net(x).sum(dim=1)[:, 0]


def _cell_centers(device):
    rows = (torch.arange(HEAT_H, device=device, dtype=torch.float32) + 0.5) * HEATMAP_STRIDE
    cols = (torch.arange(HEAT_W, device=device, dtype=torch.float32) + 0.5) * HEATMAP_STRIDE
    return rows, cols


def heatmap_targets(positions):
    """Normalized Gaussian over the cells around each (x, y), flattened."""
    rows, cols = _cell_centers(positions.device)
    dx = (cols[None, None, :] - positions[:, 0, None, None]) / HEATMAP_STRIDE
    dy = (rows[None, :, None] - positions[:, 1, None, None]) / HEATMAP_STRIDE
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
    rows = ((index // HEAT_W)[:, None] + offsets).clamp(0, HEAT_H - 1)
    cols = ((index % HEAT_W)[:, None] + offsets).clamp(0, HEAT_W - 1)
    batch_index = torch.arange(batch, device=counts.device)[:, None, None]
    weights = counts[batch_index, rows[:, :, None], cols[:, None, :]] + 1e-6
    centers_y, centers_x = _cell_centers(counts.device)
    y = (centers_y[rows][:, :, None] * weights).sum(dim=(1, 2)) / weights.sum(dim=(1, 2))
    x = (centers_x[cols][:, None, :] * weights).sum(dim=(1, 2)) / weights.sum(dim=(1, 2))
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
    """Return predicted camera pixels, shape (N, 2), in dataset order."""
    predicted = []
    model.eval()
    with torch.no_grad():
        for flat, positions in make_loader(dataset, batch_size):
            frames = to_frames(flat, len(positions), device, time_steps)
            predicted.append(decode_positions(model(frames)).cpu().numpy())
    if not predicted:
        return np.zeros((0, 2), dtype=np.float32)
    return np.concatenate(predicted, axis=0)


def train(batch_size, epochs, time_steps):
    torch.manual_seed(base.SEED)
    torch.backends.cudnn.benchmark = True
    device = base.pick_device()
    print("Device:", device, flush=True)

    dataset = base.ensure_dataset()
    train_arrays, validation_arrays, test_arrays = base._read_splits(dataset)
    train_dataset = SparseBallDataset(*train_arrays)
    validation_dataset = SparseBallDataset(*validation_arrays)
    train_loader = make_loader(train_dataset, batch_size, shuffle=True)
    validation_loader = make_loader(validation_dataset, batch_size)
    print("Training frames:", len(train_dataset))
    print("Validation frames:", len(validation_dataset))
    print("Batch size:", batch_size)
    print("Time steps:", time_steps)

    example_frames = base.save_sample_figure(
        [dense_sample(train_dataset, 0, time_steps)], Path("ball_sample.png"),
    )

    model = HeatmapBallDetector().to(device)
    model.eval()
    print("Running one forward pass", flush=True)
    with torch.no_grad():
        counts = model(example_frames.unsqueeze(0).to(device))
    if tuple(counts.shape[1:]) != (HEAT_H, HEAT_W):
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
                "camera_width": WIDTH,
                "camera_height": HEIGHT,
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
    test_loader = make_loader(SparseBallDataset(*test_arrays), batch_size)
    print("Test", flush=True)
    test_loss, test_error = run_epoch(
        model, test_loader, device, time_steps, log_prefix="Test",
    )
    print("Best epoch:", checkpoint["epoch"])
    print(f"Best validation error: {checkpoint['validation_error']:.2f} px")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test error: {test_error:.2f} px")

    base.export_model(model, example_frames, device)


def visualize_test(checkpoint_path, output_path, fps, batch_size):
    """Run the checkpoint on hand-labeled takes 7 and 8 and write an mp4."""
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path}. Train first with: "
            "python train_motion_buddy_fast.py"
        )
    device = base.pick_device()
    print("Device:", device, flush=True)
    dataset_dir = base.ensure_dataset()
    model = HeatmapBallDetector().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    time_steps = int(checkpoint["time_steps"])
    print("Time steps:", time_steps)

    test_paths = [
        path for path in base._recordings(dataset_dir / "hand_labeled")
        if base._hand_labeled_bucket(path) == "test"
    ]
    if not test_paths:
        raise FileNotFoundError("No hand-labeled test recordings (takes 7 and 8).")

    writer = base.VideoWriter(output_path, fps)
    errors = []
    try:
        for path in test_paths:
            print("Visualizing", path.name, flush=True)
            events, offsets, xy = base.recording_to_samples(path)
            predicted = infer_positions(
                model, SparseBallDataset(events, offsets, xy), device, batch_size,
                time_steps,
            )
            for index in range(len(xy)):
                start, end = int(offsets[index]), int(offsets[index + 1])
                error = float(np.linalg.norm(predicted[index] - xy[index]))
                errors.append(error)
                caption = (
                    f"{path.name}  {index + 1}/{len(xy)}  "
                    f"pred ({predicted[index, 0]:.0f}, {predicted[index, 1]:.0f})  "
                    f"gt ({xy[index, 0]:.0f}, {xy[index, 1]:.0f})  "
                    f"error {error:.1f} px"
                )
                writer.write(base.render_detection(
                    events[start:end], predicted[index], xy[index], caption,
                ))
    finally:
        writer.close()
    print(f"Wrote {writer.frames} frames to {output_path.resolve()}")
    if errors:
        print(f"Test error: {float(np.mean(errors)):.2f} px")


def main():
    parser = argparse.ArgumentParser(description="Heatmap spiking ball detector training.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    subparsers = parser.add_subparsers(dest="command")
    train_parser = subparsers.add_parser("train", help="train on the benchmark and export")
    train_parser.add_argument("--epochs", type=int, default=base.EPOCHS)
    train_parser.add_argument("--time-steps", type=int, default=TIME_STEPS)
    view = subparsers.add_parser("visualize", help="write a test-set video")
    view.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    view.add_argument("--output", type=Path, default=Path("ball_heatmap_test.mp4"))
    view.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    if args.command == "visualize":
        visualize_test(args.checkpoint, args.output, args.fps, args.batch_size)
    else:
        train(
            args.batch_size,
            getattr(args, "epochs", base.EPOCHS),
            getattr(args, "time_steps", TIME_STEPS),
        )


if __name__ == "__main__":
    main()
