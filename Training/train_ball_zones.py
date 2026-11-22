"""Ball zone classifier: which sixth of the frame the ball is in, or no ball.

Built on train_ball_small.py: same input ([2, 320, 320], one binary channel per
polarity), same int8 fake-quantized training, same data, crops, windows and
no-ball recordings. Instead of a position heatmap the network answers with a
vector of CLASSES spike counts, one output neuron per class:

    upper_left   upper_middle   upper_right      y < 160
    lower_left   lower_middle   lower_right      y >= 160
    none                                         no ball in the frame

Columns split the 320 px frame into thirds. The label comes from the ball's
centre, so a ball on a border belongs to the zone its centre is in.

Network: merge and the encoder of the small U-Net (enc1-enc7, down to a 10x10
grid), then ``reduce``, a 4x4 stride-2 ConvLIF like the encoder's (10x10 ->
5x5), and ``classify``, a ConvLIF whose 5x5 kernel covers the whole 5x5 grid,
so it acts as a fully connected layer and outputs [T, 7, 1, 1]. No kernel is
larger than enc1's 8x8. No decoder.
Off-chip: logits = spike count / T * logit_scale, class = argmax; an output
with no spikes at all counts as "none".

The "none" class needs recordings without a ball (``--no-ball FOLDER``);
without them it has no training examples. The test set (takes 7-8) holds ball
frames only.

``visualize`` writes a test-set video and ``predict`` one for any HDF5
recording: the frame with the zone grid, the predicted zone in red, the true
zone in green when the frame is labeled::

    python train_ball_zones.py train --no-ball data/no_ball
    python train_ball_zones.py train --epochs 30 --no-ball data/no_ball
    python train_ball_zones.py export
    python train_ball_zones.py visualize
    python train_ball_zones.py predict data/recording.h5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn

import train_ball_small as small
import train_ball_size as size
import train_motion_buddy as base
import train_motion_buddy_fast as fast
from riverops.nn import prepare_qat

CLASSES = (
    "upper_left", "upper_middle", "upper_right",
    "lower_left", "lower_middle", "lower_right",
    "none",
)
ROWS, COLUMNS = 2, 3
NONE = CLASSES.index("none")
GRID = fast.SENSOR // 64
ARCHITECTURE = "zones-small-polarity-v2"
CHECKPOINT = Path("ball_zones_best.pt")
size.EXPORT_PREFIX = "ball_zones"
SAMPLE_FIGURE = Path("ball_zones_sample.png")
TEST_VIDEO = Path("ball_zones_test.mp4")
PREDICT_SUFFIX = "_zones_pred.mp4"


class ZoneNet(nn.Module):
    """[B, T, 2, 320, 320] -> class spikes [B, T, len(CLASSES), 1, 1].

    Grids: merge 320x320, enc1 80x80, enc2-3 40x40, enc4-5 20x20, enc6-7 10x10,
    reduce 5x5 (GRID), classify 1x1. Modules are registered in forward order,
    which is the export order.
    """

    def __init__(self, widths=small.WIDTHS):
        super().__init__()
        c1, c2, c3, c4 = widths
        conv = size._conv
        self.merge = conv(small.POLARITIES, 1, 1, padding=0, first_layer=True)
        self.enc1 = conv(1, c1, 8, stride=4, padding=2)
        self.enc2 = conv(c1, c2, 4, stride=2, padding=1)
        self.enc3 = conv(c2, c2, 3)
        self.enc4 = conv(c2, c3, 4, stride=2, padding=1)
        self.enc5 = conv(c3, c3, 3)
        self.enc6 = conv(c3, c4, 4, stride=2, padding=1)
        self.enc7 = conv(c4, c4, 3)
        self.reduce = conv(c4, c4, 4, stride=2, padding=1)
        self.classify = conv(c4, len(CLASSES), GRID, padding=0)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ZoneDetector(nn.Module):
    """``ZoneNet`` plus the off-chip readout: spike counts over time and one logit scale."""

    def __init__(self):
        super().__init__()
        body = ZoneNet()
        self.body = (
            prepare_qat(body, size.QUANT_CONFIG, fused_bn=True) if size.TRAIN_INT8 else body
        )
        self.log_logit_scale = nn.Parameter(torch.tensor(float(np.log(fast.LOGIT_SCALE))))

    def forward(self, x):
        """Spike counts per class, [B, len(CLASSES)]."""
        return self.body(x).sum(dim=1).flatten(1)

    def logit_scale(self):
        low, high = np.log(size.LOGIT_SCALE_RANGE)
        return self.log_logit_scale.clamp(low, high).exp()


def zone_labels(targets):
    """Class index per (x, y, radius) target in sensor pixels; x < 0 means no ball."""
    x, y = targets[:, 0], targets[:, 1]
    cols = (x / (fast.SENSOR / COLUMNS)).long().clamp(0, COLUMNS - 1)
    rows = (y / (fast.SENSOR / ROWS)).long().clamp(0, ROWS - 1)
    return torch.where(x >= 0, rows * COLUMNS + cols, torch.full_like(cols, NONE))


def decide(counts, logits):
    """Argmax class; an output without a single spike is "none"."""
    predicted = logits.argmax(dim=1)
    predicted[counts.sum(dim=1) == 0] = NONE
    return predicted


def summarize(loss, confusion):
    """Accuracy, balanced accuracy (mean recall over classes present) and recall per class."""
    confusion = confusion.double()
    present = confusion.sum(dim=1)
    recall = torch.where(present > 0, confusion.diag() / present.clamp(min=1), torch.nan)
    return {
        "loss": loss,
        "accuracy": (confusion.diag().sum() / confusion.sum()).item(),
        "balanced_accuracy": float(np.nanmean(recall.numpy())),
        "recall": dict(zip(CLASSES, recall.tolist())),
        "frames": dict(zip(CLASSES, present.long().tolist())),
    }


def describe(metrics):
    recall = ", ".join(
        f"{name} {value:.0%}" for name, value in metrics["recall"].items()
        if metrics["frames"][name]
    )
    return (
        f"loss {metrics['loss']:.3f} | accuracy {metrics['accuracy']:.0%} | "
        f"balanced {metrics['balanced_accuracy']:.0%} | recall: {recall}"
    )


def describe_confusion(confusion):
    """Rows: true class, columns: predicted class."""
    width = max(map(len, CLASSES)) + 2
    lines = ["true \\ predicted".ljust(width) + "".join(name[:9].rjust(10) for name in CLASSES)]
    for name, row in zip(CLASSES, confusion.tolist()):
        lines.append(name.ljust(width) + "".join(f"{value:10d}" for value in row))
    return "\n".join(lines)


def run_epoch(model, loader, device, time_steps, optimizer=None, log_prefix=""):
    training = optimizer is not None
    model.train(training)
    loss_sum = torch.zeros((), device=device)
    classes = len(CLASSES)
    confusion = torch.zeros(classes, classes, dtype=torch.long, device=device)
    total = 0
    batch_count = len(loader)
    with torch.set_grad_enabled(training):
        for step, (flat, targets) in enumerate(loader, start=1):
            if step == 1 or step % 10 == 0 or step == batch_count:
                print(f"{log_prefix} batch {step}/{batch_count}", flush=True)
            labels = zone_labels(targets.to(device, non_blocking=True))
            frames = fast.to_frames(flat, len(labels), device, time_steps)
            if training:
                optimizer.zero_grad()
            counts = model(frames)
            logits = counts * (model.logit_scale() / time_steps)
            loss = nn.functional.cross_entropy(logits, labels)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                with torch.no_grad():
                    for parameter in model.body.parameters():
                        parameter.clamp_(-1.0, 1.0)
            predicted = decide(counts.detach(), logits.detach())
            confusion += torch.bincount(
                labels * classes + predicted, minlength=classes * classes,
            ).view(classes, classes)
            loss_sum += loss.detach() * len(labels)
            total += len(labels)
    return summarize(loss_sum.item() / total, confusion.cpu()), confusion.cpu()


def export_model(model, example_frames, device):
    """Write ball_zones_model.json, _parameters.{json,txt} and _readout.json."""
    model.eval()
    size.export_network(model.body, example_frames, device)
    readout = {
        "time_steps": int(example_frames.shape[0]),
        "classes": list(CLASSES),
        "zones": {"rows": ROWS, "columns": COLUMNS, "sensor": fast.SENSOR},
        "logit_scale": model.logit_scale().item(),
        "logits": "spike count / time_steps * logit_scale",
        "decision": "argmax of the logits; no spikes at all -> none",
    }
    path = Path(f"{size.EXPORT_PREFIX}_readout.json")
    path.write_text(json.dumps(readout, indent=2) + "\n")
    print(f"Saved off-chip readout constants to: {path}")


def train(batch_size, epochs, time_steps, learning_rate, min_learning_rate,
          no_ball_folders, no_ball_step_ms, no_ball_window_ms, no_ball_share, window_ms):
    torch.manual_seed(base.SEED)
    torch.backends.cudnn.benchmark = True
    device = base.pick_device()
    print("Device:", device, flush=True)
    base.ACC_TIME = round(window_ms * 1000)
    no_ball_window_ms = no_ball_window_ms or window_ms
    print(f"Window: {window_ms:g} ms of events per frame (no-ball frames {no_ball_window_ms:g} ms)")
    if not no_ball_folders:
        print("WARNING: no --no-ball folders, so the 'none' class has no training examples")

    train_dataset, validation_dataset, test_dataset, _, validation_ball = size.training_data(
        no_ball_folders, no_ball_step_ms, no_ball_window_ms, no_ball_share,
    )
    train_loader = fast.make_loader(train_dataset, batch_size, shuffle=True)
    validation_loader = fast.make_loader(validation_dataset, batch_size)
    print("Classes:", ", ".join(f"{i}={name}" for i, name in enumerate(CLASSES)))
    print("Batch size:", batch_size)
    print("Time steps:", time_steps)
    print(
        "Quantization: int8 weights and 16-bit potentials during training"
        if size.TRAIN_INT8 else "Quantization: float training, int8 only at export",
        flush=True,
    )

    example_frames = base.save_sample_figure(
        [fast.dense_sample(validation_ball, 0, time_steps)], SAMPLE_FIGURE,
    )
    model = ZoneDetector().to(device)
    model.eval()
    with torch.no_grad():
        counts = model(example_frames.unsqueeze(0).to(device))
    if tuple(counts.shape[1:]) != (len(CLASSES),):
        raise RuntimeError(f"Unexpected output shape {tuple(counts.shape)}")
    print("Output shape:", tuple(counts.shape), flush=True)

    optimizer = torch.optim.Adam([
        {"params": model.body.parameters(), "weight_decay": base.WEIGHT_DECAY},
        {"params": [model.log_logit_scale], "weight_decay": 0.0},
    ], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=min_learning_rate,
    )
    print(f"Learning rate: {learning_rate:g} -> {min_learning_rate:g} (cosine over {epochs} epochs)")
    print("Checkpoint kept by validation balanced accuracy", flush=True)
    best = -1.0

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs} training, lr {scheduler.get_last_lr()[0]:.2e}", flush=True)
        train_metrics, _ = run_epoch(
            model, train_loader, device, time_steps, optimizer,
            log_prefix=f"Epoch {epoch} train",
        )
        scheduler.step()
        print(f"Epoch {epoch}/{epochs} validation", flush=True)
        validation_metrics, validation_confusion = run_epoch(
            model, validation_loader, device, time_steps,
            log_prefix=f"Epoch {epoch} validation",
        )
        score = validation_metrics["balanced_accuracy"]
        saved = score > best
        if saved:
            best = score
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "validation_score": score,
                "validation_metrics": validation_metrics,
                "sensor": fast.SENSOR,
                "crop_sides": fast.CROP_SIDES,
                "time_steps": time_steps,
                "window_ms": window_ms,
                "classes": CLASSES,
                "architecture": ARCHITECTURE,
                "train_int8": size.TRAIN_INT8,
            }, CHECKPOINT)
        print(f"Epoch {epoch}/{epochs} | Training   {describe(train_metrics)}")
        print(
            f"Epoch {epoch}/{epochs} | Validation {describe(validation_metrics)}"
            + ("  [saved]" if saved else "")
        )
        print(
            f"Epoch {epoch}/{epochs} | Logit scale {model.logit_scale().item():.1f} | "
            + size.describe_rates(size.firing_rates(model, validation_loader, device, time_steps)),
            flush=True,
        )
        print(describe_confusion(validation_confusion), flush=True)

    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Test (ball frames only)", flush=True)
    test_metrics, test_confusion = run_epoch(
        model, fast.make_loader(test_dataset, batch_size), device, time_steps, log_prefix="Test",
    )
    print("Best epoch:", checkpoint["epoch"])
    print(f"Best validation balanced accuracy {checkpoint['validation_score']:.0%}")
    print(f"Test {describe(test_metrics)}")
    print(describe_confusion(test_confusion))
    export_model(model, example_frames, device)


def load_model(checkpoint_path, device):
    """(ZoneDetector with the checkpoint's weights, the checkpoint dict)."""
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path}. Train first with: python train_ball_zones.py train"
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError(f"{checkpoint_path} is not a {ARCHITECTURE} checkpoint; retrain.")
    model = ZoneDetector().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"Loaded {checkpoint_path}: epoch {checkpoint['epoch']}, "
        f"time steps {checkpoint['time_steps']}, window {checkpoint['window_ms']:g} ms"
    )
    return model, checkpoint


def export_checkpoint(checkpoint_path):
    """Export a saved checkpoint (by default the best one) without retraining."""
    device = base.pick_device()
    model, checkpoint = load_model(checkpoint_path, device)
    time_steps = int(checkpoint["time_steps"])
    export_model(
        model, torch.zeros(time_steps, fast.INPUT_CHANNELS, fast.SENSOR, fast.SENSOR), device,
    )


def infer(model, dataset, device, batch_size, time_steps):
    """Predicted class (N,) and class probabilities (N, len(CLASSES)), in dataset order."""
    classes, probabilities = [], []
    model.eval()
    with torch.no_grad():
        scale = model.logit_scale()
        for flat, targets in fast.make_loader(dataset, batch_size):
            counts = model(fast.to_frames(flat, len(targets), device, time_steps))
            logits = counts * (scale / time_steps)
            classes.append(decide(counts, logits).cpu())
            probabilities.append(logits.softmax(dim=1).cpu())
    if not classes:
        return np.zeros(0, dtype=np.int64), np.zeros((0, len(CLASSES)), dtype=np.float32)
    return torch.cat(classes).numpy(), torch.cat(probabilities).numpy()


def render_zones(flat, predicted, truth, caption):
    """Sensor frame with the zone grid; red box = prediction, green box = truth (if given)."""
    scale = fast.DISPLAY_SCALE
    side = fast.SENSOR * scale
    image = np.zeros((fast.SENSOR * fast.SENSOR, 3), dtype=np.uint8)
    image[flat.numpy() % (fast.SENSOR * fast.SENSOR)] = 210
    image = image.reshape(fast.SENSOR, fast.SENSOR, 3).repeat(scale, axis=0).repeat(scale, axis=1)
    picture = Image.fromarray(image)
    draw = ImageDraw.Draw(picture)
    width, height = side / COLUMNS, side / ROWS

    def box(index):
        row, col = divmod(index, COLUMNS)
        return (col * width, row * height, (col + 1) * width - 1, (row + 1) * height - 1)

    for col in range(1, COLUMNS):
        draw.line((col * width, 0, col * width, side), fill=(90, 90, 90), width=1)
    for row in range(1, ROWS):
        draw.line((0, row * height, side, row * height), fill=(90, 90, 90), width=1)
    if truth is not None and truth != NONE:
        draw.rectangle(box(truth), outline=(80, 220, 80), width=7)
    if predicted != NONE:
        draw.rectangle(box(predicted), outline=(240, 60, 60), width=3)
    draw.rectangle((0, 0, side, 28), fill=(0, 0, 0))
    draw.text((8, 6), caption, fill=(255, 255, 255))
    legend = "green true zone    red predicted zone" if truth is not None else "red predicted zone"
    draw.text((8, side - 22), legend, fill=(255, 255, 255))
    return np.asarray(picture)


def write_video(model, device, dataset, name, output_path, fps, batch_size, time_steps):
    """Render every window of ``dataset`` with its predicted (and, if labeled, true) zone."""
    predicted, probabilities = infer(model, dataset, device, batch_size, time_steps)
    writer = fast.VideoWriter(output_path, fps, fast.SENSOR * fast.DISPLAY_SCALE)
    classes = len(CLASSES)
    confusion = np.zeros((classes, classes), dtype=np.int64)
    try:
        for index in range(len(dataset)):
            flat, target = dataset[index]
            guess = int(predicted[index])
            caption = (
                f"{name}  {index + 1}/{len(dataset)}  "
                f"pred {CLASSES[guess]} {probabilities[index, guess]:.2f}"
            )
            truth = None
            if dataset.labeled:
                truth = int(zone_labels(target[None])[0])
                confusion[truth, guess] += 1
                caption += f"  true {CLASSES[truth]}"
            writer.write(render_zones(flat, guess, truth, caption))
    finally:
        writer.close()
    print(f"Wrote {writer.frames} frames to {Path(output_path).resolve()}")
    print("Predicted:", ", ".join(
        f"{name} {np.mean(predicted == i):.0%}" for i, name in enumerate(CLASSES)
    ))
    if dataset.labeled and len(dataset):
        print(describe(summarize(float("nan"), torch.from_numpy(confusion))))
        print(describe_confusion(torch.from_numpy(confusion)))


def visualize_test(checkpoint_path, output_path, fps, batch_size):
    """Run the checkpoint on the fixed test crops (takes 7 and 8) and write an mp4."""
    device = base.pick_device()
    print("Device:", device, flush=True)
    model, checkpoint = load_model(checkpoint_path, device)
    base.ACC_TIME = round(float(checkpoint["window_ms"]) * 1000)
    write_video(model, device, size.benchmark_splits()[2], "test 7-8", output_path, fps,
                batch_size, int(checkpoint["time_steps"]))


def predict_recording(recording, checkpoint_path, output_path, fps, batch_size,
                      step_ms, window_ms, use_labels):
    """Run the checkpoint on one HDF5 event file and write an mp4."""
    fast.check_hdf5(recording)
    device = base.pick_device()
    print("Device:", device, flush=True)
    model, checkpoint = load_model(checkpoint_path, device)
    window_ms = window_ms or float(checkpoint["window_ms"])
    dataset = fast.recording_dataset(recording, step_ms * 1000.0, window_ms * 1000.0, use_labels)
    output_path = output_path or Path(f"{recording.stem}{PREDICT_SUFFIX}")
    write_video(model, device, dataset, recording.name, output_path, fps,
                batch_size, int(checkpoint["time_steps"]))


def main():
    parser = argparse.ArgumentParser(description="Spiking ball zone classifier, 320x320.")
    parser.add_argument("--batch-size", type=int, default=fast.BATCH_SIZE)
    subparsers = parser.add_subparsers(dest="command")
    train_parser = subparsers.add_parser("train", help="train on the benchmark and export")
    train_parser.add_argument("--epochs", type=int, default=base.EPOCHS)
    train_parser.add_argument("--time-steps", type=int, default=fast.TIME_STEPS)
    train_parser.add_argument("--lr", type=float, default=base.LEARNING_RATE,
                              help="start of the cosine schedule")
    train_parser.add_argument("--min-lr", type=float, default=size.MIN_LEARNING_RATE,
                              help="end of the cosine schedule (reached at the last epoch)")
    train_parser.add_argument("--no-ball", type=Path, nargs="+", default=[],
                              help="folders with recordings where no ball is visible "
                                   "(.h5/.hdf5, searched recursively); the 'none' class")
    train_parser.add_argument("--no-ball-step-ms", type=float, default=size.NO_BALL_STEP_MS,
                              help="spacing of the windows cut from no-ball recordings")
    train_parser.add_argument("--window-ms", type=float, default=size.WINDOW_MS,
                              help="events per ball frame, ending at each label")
    train_parser.add_argument("--no-ball-window-ms", type=float, default=None,
                              help="events per no-ball frame; default: --window-ms")
    train_parser.add_argument("--no-ball-share", type=float, default=size.NO_BALL_SHARE,
                              help="share of training frames without a ball (resampled); "
                                   "0 = use every no-ball window once")
    export = subparsers.add_parser("export", help="export a checkpoint (default: the best)")
    export.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    view = subparsers.add_parser("visualize", help="write a test-set video")
    view.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    view.add_argument("--output", type=Path, default=TEST_VIDEO)
    view.add_argument("--fps", type=int, default=15)
    predict = subparsers.add_parser(
        "predict", help="write a prediction video for an HDF5 event file",
    )
    predict.add_argument("recording", type=Path)
    predict.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    predict.add_argument("--output", type=Path, default=None,
                         help=f"default: <recording name>{PREDICT_SUFFIX} in the current folder")
    predict.add_argument("--fps", type=int, default=30)
    predict.add_argument("--step-ms", type=float, default=5.0,
                         help="window spacing for files without labels")
    predict.add_argument("--window-ms", type=float, default=None,
                         help="events per frame; default: the window the checkpoint was trained on")
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
            getattr(args, "min_lr", size.MIN_LEARNING_RATE),
            getattr(args, "no_ball", []),
            getattr(args, "no_ball_step_ms", size.NO_BALL_STEP_MS),
            getattr(args, "no_ball_window_ms", None),
            getattr(args, "no_ball_share", size.NO_BALL_SHARE),
            getattr(args, "window_ms", size.WINDOW_MS),
        )


if __name__ == "__main__":
    main()
