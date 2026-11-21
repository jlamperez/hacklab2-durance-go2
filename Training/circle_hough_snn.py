"""Training-free circle detector on event frames: a Hough transform as a spiking network.

A circle Hough transform is a convolution of the edge image with a ring per
radius; the peak over (radius, y, x) is the circle. Event frames already are
edge images, and a convolution followed by spiking neurons is what the RIVER
hardware runs, so the detector is two hand-set spiking layers, no training:

1. ``pool``: POOL x POOL OR-pooling, 320x320 -> 160x160. Every weight is
   POOL_WEIGHT and the threshold is below it, so a neuron fires at every
   step if any pixel of its block is active.
2. ``rings``: one output channel per radius in RADII (pooled pixels). Each
   kernel is a 1-pixel ring (weights summing to +1) minus two bands at
   +-BAND_OFFSET (each summing to -INHIBIT). Uniform texture and straight
   edges hit ring and bands alike and cancel; a circle of the right radius
   fills only the ring. The input current is the filled share of the ring.

With threshold 1 (times WEIGHT_GAIN, see below), almost no leak and a soft
reset, a neuron's spike count over T steps is about T x that share. The
decoding runs off-chip: the peak of count / T over (radius, y, x) is the
circle if it is at least MIN_SCORE; its centre and radius are refined with
the counts of the neighbouring cells and radii.

Kernels are scaled by WEIGHT_GAIN (largest weight -> 0.99, threshold scaled
alike) so that the small ring weights survive 8-bit quantization better.
The RIVER IR does not record the reset type, so whether the board uses the
soft reset assumed here is unverified (``--reset HARD`` to compare).

Evaluation uses the benchmark's hand-labeled takes (validation 5-6, test 7-8)
as 320x320 crops at native scale (``--crop-side`` to zoom), with the event
radius measured around each label (see ``train_ball_size.py``)::

    python circle_hough_snn.py                     # evaluate on validation and test
    python circle_hough_snn.py evaluate --quantized
    python circle_hough_snn.py visualize
    python circle_hough_snn.py predict data/recording.h5 --window-ms 5
    python circle_hough_snn.py export
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

import train_motion_buddy as base
import train_motion_buddy_fast as fast
import train_ball_size as sized
from riverops.export import export_for_river
from riverops.nn import FixedPointFormat, QuantConfig, SpikingConv2d, prepare_qat

SENSOR = fast.SENSOR
POOL = 2
GRID = SENSOR // POOL
RADII = np.arange(2, 13)
RING_HALF_WIDTH = 0.7
BAND_OFFSET = 2.0
INHIBIT = 0.5
KERNEL = 2 * int(RADII.max() + BAND_OFFSET + 1) + 1
POOL_WEIGHT = 0.99
POOL_THRESHOLD = 0.5
RING_THRESHOLD = 1.0
LEAK_TAU = 4096.0
TIME_STEPS = 8
MIN_SCORE = 0.3
BATCH_SIZE = 16
QUANT_CONFIG = QuantConfig(
    weight_format=FixedPointFormat(total_bits=8, frac_bits=7),
    potential_format=FixedPointFormat(total_bits=16, frac_bits=12),
    enable_weight_quant=True,
    enable_potential_quant=True,
)


def ring_kernels():
    """[len(RADII), 1, KERNEL, KERNEL]: ring (+1 in total) minus two bands (-INHIBIT each)."""
    half = KERNEL // 2
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    distance = np.hypot(xx, yy)
    kernels = []
    for radius in RADII:
        ring = np.abs(distance - radius) <= RING_HALF_WIDTH
        inner = np.abs(distance - max(radius - BAND_OFFSET, 0.0)) <= RING_HALF_WIDTH
        outer = np.abs(distance - (radius + BAND_OFFSET)) <= RING_HALF_WIDTH
        kernel = (
            ring / ring.sum()
            - INHIBIT * inner / inner.sum()
            - INHIBIT * outer / outer.sum()
        )
        kernels.append(kernel)
    return torch.tensor(np.stack(kernels)[:, None], dtype=torch.float32)


KERNELS = ring_kernels()
WEIGHT_GAIN = float(0.99 / KERNELS.abs().max())


class HoughCircleSNN(nn.Module):
    """Two hand-set spiking layers; forward returns spike counts [B, len(RADII), GRID, GRID]."""

    def __init__(self, reset_mode="SOFT"):
        super().__init__()
        neuron = {"initial_tau": LEAK_TAU, "learnable_leak": False}
        self.net = nn.Sequential(
            SpikingConv2d(1, 1, kernel_size=POOL, stride=POOL, bias=False, first_layer=True,
                          v_threshold=POOL_THRESHOLD, reset_mode="HARD", **neuron),
            SpikingConv2d(1, len(RADII), kernel_size=KERNEL, padding=KERNEL // 2, bias=False,
                          v_threshold=RING_THRESHOLD * WEIGHT_GAIN, reset_mode=reset_mode,
                          **neuron),
        )
        with torch.no_grad():
            self.net[0].conv.weight.fill_(POOL_WEIGHT)
            self.net[1].conv.weight.copy_(KERNELS * WEIGHT_GAIN)
        self.requires_grad_(False)

    def forward(self, x):
        return self.net(x).sum(dim=1)


def _centroid(values, centers, index, limit):
    """Weighted mean of ``centers[index-1 .. index+1]`` with non-negative ``values`` weights."""
    offsets = torch.arange(-1, 2, device=values.device)
    neighbours = (index[:, None] + offsets).clamp(0, limit - 1)
    weights = values.gather(1, neighbours).clamp(min=0) + 1e-6
    return (centers[neighbours] * weights).sum(dim=1) / weights.sum(dim=1)


def detect(counts, time_steps):
    """Off-chip decoding: (x, y, radius) in sensor pixels, score, found, per sample."""
    score = counts / time_steps
    batch = torch.arange(score.shape[0], device=score.device)
    best = score.flatten(1).argmax(dim=1)
    peak = score.flatten(1).max(dim=1).values
    r_index = best // (GRID * GRID)
    row = (best % (GRID * GRID)) // GRID
    col = best % GRID
    plane = score[batch, r_index]
    cells = torch.arange(GRID, device=score.device, dtype=torch.float32)
    x = _centroid(plane[batch, row, :], cells, col, GRID)
    y = _centroid(plane[batch, :, col], cells, row, GRID)
    radii = torch.as_tensor(RADII, device=score.device, dtype=torch.float32)
    radius = _centroid(score[batch, :, row, col], radii, r_index, len(RADII))
    xy_radius = torch.stack(((x + 0.5) * POOL, (y + 0.5) * POOL, radius * POOL), dim=1)
    return xy_radius, peak, peak >= MIN_SCORE


def detect_all(model, dataset, device, batch_size, time_steps):
    """Detections for a whole dataset: (N, 3) circles, (N,) scores, (N,) found."""
    circles, scores, found = [], [], []
    model.eval()
    with torch.no_grad():
        for flat, targets in fast.make_loader(dataset, batch_size):
            frames = fast.to_frames(flat, len(targets), device, time_steps)
            circle, score, hit = detect(model(frames), time_steps)
            circles.append(circle.cpu())
            scores.append(score.cpu())
            found.append(hit.cpu())
    return torch.cat(circles).numpy(), torch.cat(scores).numpy(), torch.cat(found).numpy()


def native_crop(seed, side):
    """Square crop of ``side`` benchmark pixels around the ball, fixed per index."""
    def crop(index, xy):
        rng = np.random.default_rng((seed, index))
        return (
            fast._crop_origin(rng, xy[0], side, fast.SOURCE_WIDTH),
            fast._crop_origin(rng, xy[1], side, fast.SOURCE_HEIGHT),
            float(side),
        )
    return crop


def benchmark_sets(crop_side):
    """Validation and test SizedDatasets (targets x, y, measured radius) at one crop size."""
    dataset_dir = base.ensure_dataset()
    _, validation, test = base._read_splits(dataset_dir)
    _, validation_lengths, test_lengths = sized.split_file_lengths(dataset_dir)
    sets = {}
    for name, (events, offsets, xy), lengths, seed in (
        ("validation", validation, validation_lengths, base.SEED + 1),
        ("test", test, test_lengths, base.SEED + 2),
    ):
        radii = sized.label_radii(events, offsets, xy, lengths, name)
        sets[name] = sized.SizedDataset(
            events, offsets, xy, radii, native_crop(seed, crop_side), noise_seed=seed,
        )
    return sets


def metrics(circles, scores, found, targets):
    """Detection rate, position error and radius agreement against labels."""
    errors = np.hypot(circles[:, 0] - targets[:, 0], circles[:, 1] - targets[:, 1])
    radius = targets[:, 2]
    on_ball = found & (errors <= radius)
    ratio = np.abs(np.log(circles[found, 2] / radius[found])) if found.any() else np.array([])
    return (
        f"found {found.mean():.0%} | on the ball (found, within 1 radius) {on_ball.mean():.0%} | "
        f"error median {np.median(errors):.1f} px, mean {errors.mean():.1f} px "
        f"(found only: median {np.median(errors[found]) if found.any() else float('nan'):.1f} px) | "
        f"radius typically x{np.exp(ratio.mean()) if ratio.size else float('nan'):.2f} off | "
        f"score median {np.median(scores):.2f}"
    )


def build_model(reset_mode, quantized, device):
    model = HoughCircleSNN(reset_mode)
    if quantized:
        model = prepare_qat(model, QUANT_CONFIG)
    return model.to(device).eval()


def describe_model(reset_mode, quantized, time_steps):
    print(
        f"Pool {POOL}x{POOL} -> {GRID}x{GRID}; radii {RADII[0] * POOL}-{RADII[-1] * POOL} sensor px "
        f"({len(RADII)} channels, kernel {KERNEL}x{KERNEL}); weight gain {WEIGHT_GAIN:.2f}; "
        f"reset {reset_mode}; T={time_steps}; min score {MIN_SCORE}; "
        f"{'8-bit quantized' if quantized else 'float'}",
        flush=True,
    )


def evaluate(crop_side, reset_mode, quantized, time_steps, batch_size):
    device = base.pick_device()
    describe_model(reset_mode, quantized, time_steps)
    model = build_model(reset_mode, quantized, device)
    for name, dataset in benchmark_sets(crop_side).items():
        targets = np.stack([dataset[i][1].numpy() for i in range(len(dataset))])
        circles, scores, found = detect_all(model, dataset, device, batch_size, time_steps)
        print(f"{name} ({len(dataset)} frames, crop side {crop_side:.0f}): "
              f"{metrics(circles, scores, found, targets)}", flush=True)


def write_video(model, device, dataset, name, output_path, fps, batch_size, time_steps):
    """Red ring: detected circle (none if below MIN_SCORE). Green: label with measured radius."""
    circles, scores, found = detect_all(model, dataset, device, batch_size, time_steps)
    writer = fast.VideoWriter(output_path, fps, SENSOR * fast.DISPLAY_SCALE)
    try:
        for index in range(len(dataset)):
            flat, target = dataset[index]
            x, y, radius = (float(v) for v in circles[index])
            caption = f"{name}  {index + 1}/{len(dataset)}  score {scores[index]:.2f}  "
            caption += f"circle ({x:.0f}, {y:.0f}) r {radius:.1f}" if found[index] else "no circle"
            truth, truth_radius = None, 6.0
            if dataset.labeled:
                truth, truth_radius = (target[0].item(), target[1].item()), target[2].item()
                caption += f"  gt ({truth[0]:.0f}, {truth[1]:.0f}) r {truth_radius:.1f}"
            writer.write(fast.render_frame(
                flat, (x, y) if found[index] else None, truth, caption,
                prediction_radius=radius, truth_radius=truth_radius,
            ))
    finally:
        writer.close()
    print(f"Wrote {writer.frames} frames to {Path(output_path).resolve()}")
    if dataset.labeled:
        targets = np.stack([dataset[i][1].numpy() for i in range(len(dataset))])
        print(metrics(circles, scores, found, targets))


def visualize(output_path, fps, crop_side, reset_mode, quantized, time_steps, batch_size):
    device = base.pick_device()
    describe_model(reset_mode, quantized, time_steps)
    model = build_model(reset_mode, quantized, device)
    dataset = benchmark_sets(crop_side)["test"]
    write_video(model, device, dataset, "test 7-8", output_path, fps, batch_size, time_steps)


def predict(recording, output_path, fps, step_ms, window_ms, use_labels, reset_mode,
            quantized, time_steps, batch_size):
    fast.check_hdf5(recording)
    device = base.pick_device()
    describe_model(reset_mode, quantized, time_steps)
    model = build_model(reset_mode, quantized, device)
    windows = fast.recording_dataset(recording, step_ms * 1000.0, window_ms * 1000.0, use_labels)
    radii = None
    if windows.labeled:
        radii = sized.label_radii(
            windows.events, windows.offsets, windows.positions, [len(windows)], recording.name,
        )
    dataset = sized.SizedDataset(
        windows.events, windows.offsets, windows.positions if windows.labeled else None,
        radii, windows.crop, noise_seed=0,
    )
    output_path = output_path or Path(f"{recording.stem}_circles.mp4")
    write_video(model, device, dataset, recording.name, output_path, fps, batch_size, time_steps)


def export(reset_mode, time_steps):
    """Write circle_model.json and circle_parameters.{json,txt} (8-bit fixed point)."""
    model = HoughCircleSNN(reset_mode).eval()
    example = torch.zeros(1, time_steps, 1, SENSOR, SENSOR)
    Path("circle_model.json").write_text(export_for_river(model.net, example))
    prepared = prepare_qat(model, QUANT_CONFIG).eval()
    parameters = {}
    for index, layer in enumerate(prepared.net):
        params = layer.export_params()
        parameters[f"weights_{index}"] = params["weights"]
        parameters[f"bias_{index}"] = params["bias"]
        parameters[f"threshold_{index}"] = params["threshold"]
        parameters[f"leak_{index}"] = params["leak"]
    Path("circle_parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    values = []
    for value in parameters.values():
        values.extend(value if isinstance(value, list) else [value])
    Path("circle_parameters.txt").write_text("\n".join(map(str, values)))
    print("Saved circle_model.json, circle_parameters.json and circle_parameters.txt")
    print(f"Reset mode assumed: {reset_mode} (not recorded in the exported graph)")


def main():
    parser = argparse.ArgumentParser(description="Training-free spiking circle (Hough) detector.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--time-steps", type=int, default=TIME_STEPS)
    parser.add_argument("--reset", choices=("SOFT", "HARD"), default="SOFT",
                        help="reset of the ring neurons; SOFT gives counts ~ T x ring share")
    parser.add_argument("--quantized", action="store_true",
                        help="simulate 8-bit weights and 16-bit potentials")
    subparsers = parser.add_subparsers(dest="command")
    evaluate_parser = subparsers.add_parser("evaluate", help="metrics on validation and test")
    evaluate_parser.add_argument("--crop-side", type=float, default=float(SENSOR),
                                 help="benchmark pixels mapped onto 320 (320 = native scale)")
    view = subparsers.add_parser("visualize", help="write a test-set video")
    view.add_argument("--output", type=Path, default=Path("circle_test.mp4"))
    view.add_argument("--fps", type=int, default=15)
    view.add_argument("--crop-side", type=float, default=float(SENSOR))
    predict_parser = subparsers.add_parser("predict", help="video for an HDF5 event file")
    predict_parser.add_argument("recording", type=Path)
    predict_parser.add_argument("--output", type=Path, default=None,
                                help="default: <recording name>_circles.mp4")
    predict_parser.add_argument("--fps", type=int, default=30)
    predict_parser.add_argument("--step-ms", type=float, default=5.0)
    predict_parser.add_argument("--window-ms", type=float, default=base.ACC_TIME / 1000.0)
    predict_parser.add_argument("--no-labels", action="store_true")
    subparsers.add_parser("export", help="write the RIVER graph and fixed-point parameters")
    args = parser.parse_args()
    common = (args.reset, args.quantized, args.time_steps, args.batch_size)
    if args.command == "visualize":
        visualize(args.output, args.fps, args.crop_side, *common)
    elif args.command == "predict":
        predict(args.recording, args.output, args.fps, args.step_ms, args.window_ms,
                not args.no_labels, *common)
    elif args.command == "export":
        export(args.reset, args.time_steps)
    else:
        evaluate(getattr(args, "crop_side", float(SENSOR)), *common)


if __name__ == "__main__":
    main()
