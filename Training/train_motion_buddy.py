"""Train a small spiking network on the Tübingen event-camera ball dataset.

Dataset: https://github.com/cogsys-tuebingen/snn-edge-benchmark
Paper: Ziegler et al., Detection of Fast-Moving Objects with Neuromorphic Hardware.

Each sample is the full 1280×720 sensor for 1 ms (polarity discarded), not a
crop. The network classifies the ball's camera pixel as 1280 x-bins followed
by 720 y-bins. All projected recordings are training data. Hand-labeled takes
0–4 train, 5–6 validate, and 7–8 are held out.

The clone is stored in ``data/snn-edge-benchmark`` (override with
``SNN_EDGE_ROOT``). The HDF5 files are Git LFS objects.

Train, then write a test-set video (green ground truth, red prediction)::

    python train_motion_buddy.py
    python train_motion_buddy.py visualize
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset

from riverops.export import export_for_river
from riverops.nn import FixedPointFormat, QuantConfig, SpikingConvBN2d, prepare_qat

SEED = 42
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
TIME_STEPS = 8
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
X_BINS = CAMERA_WIDTH
Y_BINS = CAMERA_HEIGHT
INPUT_SIZE = CAMERA_WIDTH
ACC_TIME = 1000
CACHE_VERSION = 2

REPO_URL = "https://github.com/cogsys-tuebingen/snn-edge-benchmark.git"
DATASET_ROOT = Path(os.environ.get("SNN_EDGE_ROOT", "data/snn-edge-benchmark"))


def _is_lfs_pointer(path):
    with path.open("rb") as handle:
        return handle.read(48).startswith(b"version https://git-lfs.github.com")


def _require_git_lfs():
    try:
        result = subprocess.run(
            ["git", "lfs", "version"],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("git is required to fetch the benchmark dataset.") from exc
    if result.returncode != 0:
        raise RuntimeError(
            "The recordings are Git LFS files. Install git-lfs, then re-run this script."
        )


def ensure_dataset():
    """Clone the benchmark repo and pull the HDF5 recordings if they are missing."""
    dataset = DATASET_ROOT / "dataset"
    if not dataset.is_dir():
        DATASET_ROOT.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(DATASET_ROOT)],
            check=True,
        )
    recordings = sorted(dataset.rglob("*.hdf5"))
    if not recordings:
        raise FileNotFoundError(f"No HDF5 recordings under {dataset}")
    if any(_is_lfs_pointer(path) for path in recordings):
        _require_git_lfs()
        subprocess.run(["git", "lfs", "install"], cwd=DATASET_ROOT, check=True)
        subprocess.run(["git", "lfs", "pull"], cwd=DATASET_ROOT, check=True)
    pending = [path.name for path in recordings if _is_lfs_pointer(path)]
    if pending:
        listed = ", ".join(pending[:5])
        raise RuntimeError(
            f"These recordings are still Git LFS pointers ({listed}). "
            f"Run `git lfs pull` in {DATASET_ROOT.resolve()}."
        )
    print("Dataset folder:", dataset.resolve())
    return dataset


def _column(handle, key):
    return np.squeeze(np.asarray(handle[key]))


def _window_pixels(xs, ys):
    """Unique sensor pixels hit by one 1 ms event window, as (y, x)."""
    xi = np.floor(xs).astype(np.int32)
    yi = np.floor(ys).astype(np.int32)
    selected = (
        (xi >= 0) & (xi < CAMERA_WIDTH) & (yi >= 0) & (yi < CAMERA_HEIGHT)
    )
    if not np.any(selected):
        return np.zeros((0, 2), dtype=np.uint16)
    packed = yi[selected] * CAMERA_WIDTH + xi[selected]
    packed = np.unique(packed)
    return np.stack((packed // CAMERA_WIDTH, packed % CAMERA_WIDTH), axis=1).astype(np.uint16)


def recording_to_samples(path):
    """Build full-sensor event lists and camera-pixel (x, y) labels for one file.

    Every event in the 1 ms before a label is kept. Nothing is cropped.
    Polarity is ignored.
    """
    with h5py.File(path, "r") as handle:
        gt_x = _column(handle, "gt_x").astype(np.float64)
        gt_y = _column(handle, "gt_y").astype(np.float64)
        gt_t = _column(handle, "gt_t").astype(np.float64)
        ev_x = _column(handle, "event_x").astype(np.float64)
        ev_y = _column(handle, "event_y").astype(np.float64)
        ev_t = _column(handle, "event_t").astype(np.float64)

    empty_events = np.zeros((0, 2), dtype=np.uint16)
    empty_offsets = np.zeros(1, dtype=np.int64)
    empty_xy = np.zeros((0, 2), dtype=np.float32)
    if gt_t.size == 0:
        return empty_events, empty_offsets, empty_xy

    order = np.argsort(gt_t, kind="mergesort")
    gt_x, gt_y, gt_t = gt_x[order], gt_y[order], gt_t[order]
    visible = (
        (gt_x >= 0) & (gt_y >= 0)
        & (gt_x < CAMERA_WIDTH) & (gt_y < CAMERA_HEIGHT)
    )
    gt_x, gt_y, gt_t = gt_x[visible], gt_y[visible], gt_t[visible]
    if gt_t.size == 0:
        return empty_events, empty_offsets, empty_xy

    if ev_t.size:
        event_order = np.argsort(ev_t, kind="mergesort")
        ev_x, ev_y, ev_t = ev_x[event_order], ev_y[event_order], ev_t[event_order]
    starts = np.searchsorted(ev_t, gt_t - ACC_TIME, side="left")
    ends = np.searchsorted(ev_t, gt_t, side="right")

    pixels = [
        _window_pixels(ev_x[start:end], ev_y[start:end])
        for start, end in zip(starts, ends)
    ]
    offsets = np.zeros(len(pixels) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(window) for window in pixels])
    events = np.concatenate(pixels, axis=0) if offsets[-1] else empty_events
    xy = np.stack((gt_x, gt_y), axis=1).astype(np.float32)
    return events, offsets, xy


def _recordings(folder):
    return sorted(
        path for path in folder.glob("*.hdf5")
        if not path.name.startswith(".")
    )


def _hand_labeled_bucket(path):
    """Split takes by the trailing index so both cameras stay together."""
    take = path.stem.rsplit("_", 1)[-1]
    if take in {"7", "8"}:
        return "test"
    if take in {"5", "6"}:
        return "validation"
    return "train"


def _merge_chunks(chunks):
    events = []
    offsets = [0]
    positions = []
    cursor = 0
    for file_events, file_offsets, file_xy in chunks:
        events.append(file_events)
        offsets.extend((file_offsets[1:] + cursor).tolist())
        cursor += int(file_offsets[-1])
        positions.append(file_xy)
    merged_events = (
        np.concatenate(events, axis=0)
        if cursor else np.zeros((0, 2), dtype=np.uint16)
    )
    return (
        merged_events,
        np.asarray(offsets, dtype=np.int64),
        np.concatenate(positions, axis=0),
    )


def _load_recordings(paths):
    chunks = []
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] {path.parent.name}/{path.name}", flush=True)
        events, offsets, xy = recording_to_samples(path)
        if len(xy):
            chunks.append((events, offsets, xy))
    if not chunks:
        raise ValueError(f"No labeled frames in {paths[0].parent}")
    return _merge_chunks(chunks)


def _cache_stamp(paths):
    return [
        [str(path), path.stat().st_mtime_ns, path.stat().st_size]
        for path in paths
    ]


def _read_splits(dataset):
    projected = _recordings(dataset / "projected")
    hand_labeled = _recordings(dataset / "hand_labeled")
    name = "prepared_cache" if ACC_TIME == 1000 else f"prepared_cache_{ACC_TIME}us"
    cache_path = DATASET_ROOT / f"{name}.npz"
    stamp_path = DATASET_ROOT / f"{name}_stamp.json"
    stamp = _cache_stamp(projected + hand_labeled)
    if cache_path.exists() and stamp_path.exists():
        meta = json.loads(stamp_path.read_text())
        if meta.get("version") == CACHE_VERSION and meta.get("stamp") == stamp:
            print("Loaded prepared frames from:", cache_path)
            with np.load(cache_path) as cached:
                return tuple(
                    (
                        cached[f"{name}_events"].copy(),
                        cached[f"{name}_offsets"].copy(),
                        cached[f"{name}_xy"].copy(),
                    )
                    for name in ("train", "validation", "test")
                )
    return _prepare_splits(projected, hand_labeled, cache_path, stamp_path, stamp)


def _prepare_splits(projected, hand_labeled, cache_path, stamp_path, stamp):
    buckets = {"train": [], "validation": [], "test": []}
    for path in hand_labeled:
        buckets[_hand_labeled_bucket(path)].append(path)
    if not projected or any(not paths for paths in buckets.values()):
        raise FileNotFoundError(
            "Expected projected recordings and hand-labeled takes 0-8."
        )

    print("Preparing projected recordings")
    projected = _load_recordings(projected)
    print("Preparing hand-labeled recordings")
    hand = {
        name: _load_recordings(paths) for name, paths in buckets.items()
    }
    splits = (
        _merge_chunks((projected, hand["train"])),
        hand["validation"],
        hand["test"],
    )
    np.savez_compressed(
        cache_path,
        train_events=splits[0][0],
        train_offsets=splits[0][1],
        train_xy=splits[0][2],
        validation_events=splits[1][0],
        validation_offsets=splits[1][1],
        validation_xy=splits[1][2],
        test_events=splits[2][0],
        test_offsets=splits[2][1],
        test_xy=splits[2][2],
    )
    stamp_path.write_text(json.dumps({"version": CACHE_VERSION, "stamp": stamp}))
    print("Saved prepared frames to:", cache_path)
    return splits


class BallDataset(Dataset):
    def __init__(self, events, offsets, positions):
        self.events = np.ascontiguousarray(events)
        self.offsets = np.ascontiguousarray(offsets)
        self.positions = torch.from_numpy(np.ascontiguousarray(positions)).to(torch.float32)

    def __len__(self):
        return int(self.positions.shape[0])

    def __getitem__(self, index):
        frame = torch.zeros(INPUT_SIZE, INPUT_SIZE, dtype=torch.float32)
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        if start != end:
            pixels = self.events[start:end]
            frame[pixels[:, 0], pixels[:, 1]] = 1
        sequence = frame.view(1, INPUT_SIZE, INPUT_SIZE).repeat(TIME_STEPS, 1, 1, 1)
        return sequence, self.positions[index]


class BallDetectorModel(nn.Module):
    """Conv stack over the full sensor. Outputs one score per camera pixel axis."""

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
            SpikingConvBN2d(8, 16, kernel_size=8, stride=4, padding=2, bias=False, **lif),
            SpikingConvBN2d(16, 32, kernel_size=8, stride=4, padding=2, bias=False, **lif),
            SpikingConvBN2d(32, 64, kernel_size=8, stride=4, padding=2, bias=False, **lif),
            SpikingConvBN2d(64, 128, kernel_size=5, stride=1, padding=0, bias=False, **lif),
            SpikingConvBN2d(128, X_BINS + Y_BINS, kernel_size=1, bias=False, **lif),
        )

    def forward(self, x):
        return self.net(x).sum(dim=1).flatten(start_dim=1)


def make_position_targets(positions):
    """Peak of 1 on the true camera pixel and 0.5 on its two neighbors, per axis."""
    targets = torch.zeros(
        positions.shape[0], X_BINS + Y_BINS,
        device=positions.device, dtype=torch.float32,
    )
    rows = torch.arange(positions.shape[0], device=positions.device)
    offset = 0
    for axis, count in enumerate((X_BINS, Y_BINS)):
        index = positions[:, axis].floor().long().clamp(0, count - 1)
        low = (index - 1).clamp(0, count - 1)
        high = (index + 1).clamp(0, count - 1)
        targets[rows, offset + low] = 0.5
        targets[rows, offset + high] = 0.5
        targets[rows, offset + index] = 1.0
        offset += count
    return targets


def predict_positions(scores):
    x = scores[:, :X_BINS].argmax(dim=1).to(torch.float32)
    y = scores[:, X_BINS:].argmax(dim=1).to(torch.float32)
    return torch.stack((x, y), dim=1)


def position_error(scores, positions):
    delta = predict_positions(scores) - positions
    return torch.linalg.vector_norm(delta, dim=1)


def compute_loss(scores, targets, num_steps):
    return (scores / num_steps - targets).square().mean()


def run_epoch(model, loader, device, optimizer=None, log_prefix=""):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_error = 0.0
    total = 0
    batch_count = len(loader)
    with torch.set_grad_enabled(training):
        for step, (frames, positions) in enumerate(loader, start=1):
            if step == 1 or step % 10 == 0 or step == batch_count:
                print(f"{log_prefix} batch {step}/{batch_count}", flush=True)
            frames = frames.to(device=device, dtype=torch.float32)
            positions = positions.to(device=device, dtype=torch.float32)
            num_steps = frames.shape[1]
            targets = make_position_targets(positions)
            if training:
                optimizer.zero_grad()
            scores = model(frames)
            loss = compute_loss(scores, targets, num_steps)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.clamp_(-1.0, 1.0)
                optimizer.step()
            count = len(positions)
            total_loss += loss.item() * count
            total_error += position_error(scores, positions).sum().item()
            total += count
    return total_loss / total, total_error / total


def save_sample_figure(dataset, path):
    sequence, position = dataset[0]
    frame = sequence[0, 0, :CAMERA_HEIGHT, :CAMERA_WIDTH].numpy()
    xy = position.numpy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.imshow(frame, cmap="gray", vmin=0, vmax=1, aspect="equal")
    ax.scatter([xy[0]], [xy[1]], c="red", s=30)
    ax.set_title(f"x={xy[0]:.1f}, y={xy[1]:.1f}")
    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("Network input shape:", tuple(sequence.shape))
    print("Saved sample figure to:", path)
    return sequence


def export_model(model, example_frames, device):
    model.eval()
    example_input = example_frames.unsqueeze(0).to(device)
    model_json = export_for_river(model.net, example_input)
    export_path = Path("ball_model.json")
    export_path.write_text(model_json)
    print("Saved architecture to:", export_path)

    quant_config = QuantConfig(
        weight_format=FixedPointFormat(total_bits=8, frac_bits=7),
        potential_format=FixedPointFormat(total_bits=16, frac_bits=12),
        enable_weight_quant=True,
        enable_potential_quant=True,
    )
    prepared = prepare_qat(model, quant_config).eval()
    weights, biases, neurons = {}, {}, {}
    for index, layer in enumerate(prepared.net):
        params = layer.export_params()
        weights[f"weights_{index}"] = params["weights"]
        biases[f"bias_{index}"] = params["bias"]
        neurons[f"threshold_{index}"] = params["threshold"]
        neurons[f"leak_{index}"] = params["leak"]

    parameters = {**weights, **biases, **neurons}
    json_path = Path("ball_parameters.json")
    json_path.write_text(json.dumps(parameters, indent=2) + "\n")

    values = []
    for value in parameters.values():
        values.extend(value if isinstance(value, list) else [value])
    text_path = Path("ball_parameters.txt")
    text_path.write_text("\n".join(map(str, values)))
    print("Saved parameters to:", json_path, "and", text_path)


def pick_device():
    cuda = torch.cuda.is_available()
    print(f"CUDA available: {cuda}", flush=True)
    if cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("MPS available: True", flush=True)
        return torch.device("mps")
    return torch.device("cpu")


def infer_positions(model, dataset, device):
    """Return predicted camera pixels, shape (N, 2), in dataset order."""
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)
    predicted = []
    model.eval()
    with torch.no_grad():
        for frames, _positions in loader:
            frames = frames.to(device=device, dtype=torch.float32)
            predicted.append(predict_positions(model(frames)).cpu().numpy())
    if not predicted:
        return np.zeros((0, 2), dtype=np.float32)
    return np.concatenate(predicted, axis=0)


def render_detection(events, prediction, truth, caption):
    """Draw the event frame with a green ground-truth ring and a red prediction."""
    image = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
    if len(events):
        image[events[:, 0], events[:, 1]] = 210
    picture = Image.fromarray(image)
    draw = ImageDraw.Draw(picture)

    def ring(xy, color):
        x, y = float(xy[0]), float(xy[1])
        radius = 12
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=3)

    ring(truth, (80, 220, 80))
    ring(prediction, (240, 60, 60))
    draw.rectangle((0, 0, CAMERA_WIDTH, 28), fill=(0, 0, 0))
    draw.text((8, 6), caption, fill=(255, 255, 255))
    draw.text(
        (8, CAMERA_HEIGHT - 22),
        "green ground truth    red prediction",
        fill=(255, 255, 255),
    )
    return np.asarray(picture)


class VideoWriter:
    def __init__(self, path, fps):
        self.path = Path(path)
        self.proc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}",
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

    def write(self, frame):
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        self.frames += 1

    def close(self):
        if self._closed:
            return
        self._closed = True
        stdin = self.proc.stdin
        self.proc.stdin = None
        stdin.close()
        stderr = self.proc.stderr.read()
        return_code = self.proc.wait()
        if self.frames and return_code != 0:
            message = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed: {message}")


def visualize_test(checkpoint_path, output_path, fps):
    """Run the checkpoint on hand-labeled takes 7 and 8 and write an mp4."""
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path}. Train first with: python train_motion_buddy.py"
        )
    device = pick_device()
    print("Device:", device, flush=True)
    dataset_dir = ensure_dataset()
    model = BallDetectorModel().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_paths = [
        path for path in _recordings(dataset_dir / "hand_labeled")
        if _hand_labeled_bucket(path) == "test"
    ]
    if not test_paths:
        raise FileNotFoundError("No hand-labeled test recordings (takes 7 and 8).")

    writer = VideoWriter(output_path, fps)
    errors = []
    frames_written = 0
    try:
        for path in test_paths:
            print("Visualizing", path.name, flush=True)
            events, offsets, xy = recording_to_samples(path)
            predicted = infer_positions(model, BallDataset(events, offsets, xy), device)
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
                writer.write(render_detection(
                    events[start:end], predicted[index], xy[index], caption,
                ))
                frames_written += 1
    finally:
        writer.close()
    print(f"Wrote {frames_written} frames to {output_path.resolve()}")
    if errors:
        print(f"Test error: {float(np.mean(errors)):.2f} px")


def train():
    torch.manual_seed(SEED)
    device = pick_device()
    print("Device:", device, flush=True)

    dataset = ensure_dataset()
    train_arrays, validation_arrays, test_arrays = _read_splits(dataset)
    train_dataset = BallDataset(*train_arrays)
    validation_dataset = BallDataset(*validation_arrays)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(SEED),
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=BATCH_SIZE, num_workers=0,
    )
    print("Training frames:", len(train_dataset))
    print("Validation frames:", len(validation_dataset))

    example_frames = save_sample_figure(train_dataset, Path("ball_sample.png"))

    model = BallDetectorModel().to(device)
    model.eval()
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    print("Running one forward pass", flush=True)
    with torch.no_grad():
        scores = model(example_frames.unsqueeze(0).to(device))
    prediction = predict_positions(scores)[0].detach().cpu().tolist()
    print("Output shape:", tuple(scores.shape), flush=True)
    print(f"Sample prediction (x, y): ({prediction[0]:.0f}, {prediction[1]:.0f})", flush=True)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    checkpoint_path = Path("ball_best.pt")
    best_validation_error = float("inf")

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS} training", flush=True)
        train_loss, train_error = run_epoch(
            model, train_loader, device, optimizer, log_prefix=f"Epoch {epoch} train",
        )
        print(f"Epoch {epoch}/{EPOCHS} validation", flush=True)
        validation_loss, validation_error = run_epoch(
            model, validation_loader, device, log_prefix=f"Epoch {epoch} validation",
        )
        if validation_error < best_validation_error:
            best_validation_error = validation_error
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "validation_error": validation_error,
                "camera_width": CAMERA_WIDTH,
                "camera_height": CAMERA_HEIGHT,
                "time_steps": TIME_STEPS,
            }, checkpoint_path)
        print(
            f"Epoch {epoch}/{EPOCHS} | Training loss: {train_loss:.4f} | "
            f"Training error: {train_error:.2f} px | "
            f"Validation loss: {validation_loss:.4f} | "
            f"Validation error: {validation_error:.2f} px"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_dataset = BallDataset(*test_arrays)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0)
    print("Test", flush=True)
    test_loss, test_error = run_epoch(model, test_loader, device, log_prefix="Test")
    print("Best epoch:", checkpoint["epoch"])
    print(f"Best validation error: {checkpoint['validation_error']:.2f} px")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test error: {test_error:.2f} px")

    export_model(model, example_frames, device)


def main():
    parser = argparse.ArgumentParser(description="Train or visualize the full-frame ball detector.")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("train", help="train on the benchmark and export the network")
    view = subparsers.add_parser(
        "visualize",
        help="write a test-set video with inferred ball positions",
    )
    view.add_argument("--checkpoint", type=Path, default=Path("ball_best.pt"))
    view.add_argument("--output", type=Path, default=Path("ball_test.mp4"))
    view.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    if args.command == "visualize":
        visualize_test(args.checkpoint, args.output, args.fps)
    else:
        train()


if __name__ == "__main__":
    main()
