"""The SNN ball detector from feat/training as a Target source (reflex/runtime/target.py).

The network and its checkpoints belong to Theodor's training scripts in `Training/`
(`train_ball_small.py`, `train_ball_size.py`); this module only loads a checkpoint
with them and turns the output into a `Target`, so the two cannot drift apart.

What the network expects (from the training scripts):

  input    320x320, 1 where any event fired in the last `window_ms` (stored in the
           checkpoint, 2 ms for ball_small_best.pt), polarity ignored. The same frame
           is fed for each of the checkpoint's `time_steps`.
  output   (x, y) in sensor pixels and P(ball), from an 80x80 heatmap with in-cell
           offsets plus a learned "no ball" class.

Our 50 ms / 80x80 EventWindow is far too coarse for that, so the event camera node
publishes a second, full-resolution window on /reflex/events_fullres when someone
listens (`fullres_ms`). `frame_from_window` turns it into the input.

Window length. The network was trained on 2 ms of the snn-edge-benchmark camera
(1280x720, cropped and zoomed to 320x320). Our GenX320 recordings are far sparser:
1-3 k events/s, so a 2 ms frame is almost always empty. Measured on
data/river/sat_follow2 (21 Nov, frames every 40 ms, >= 20 event pixels), share of frames
with a ball reported:

    window   P(ball) >=   ball takes   none takes   drone take
     10 ms      0.55         18 %         13 %         35 %
     20 ms      0.50         33 %         19 %         56 %
     20 ms      0.55         28 %         16 %         43 %

So 20 ms is the default (`fullres_ms`) and 0.55 the threshold. The network reacts to
moving blobs in general, not only to the ball: the still takes give nothing, but some
"none" takes and the drone take trigger it. Ball takes also contain stretches without
the ball, so 28 % is not a recall. Frames with only a few events give P(ball) ~0.5 with
the peak in the top-left corner: hence `min_pixels`.

Needs PyTorch, `riverops` (Durance's SDK; the unpacked wheel in `riverops-0/` is
used if it is not installed) and the scripts in `Training/`.

Run it over recordings, no ROS:

    python -m reflex.runtime.ball_snn data/river/sat_follow2 --step-ms 20
"""
from __future__ import annotations

import importlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from reflex.data.windows import WindowConfig, make_window
from reflex.runtime.target import Target, no_target

REPO = Path(__file__).resolve().parents[2]

# Architecture name stored in the checkpoint -> the training script that defines it.
SCRIPTS = {
    "unet-position-small-v1": "train_ball_small",
    "unet-position-v2": "train_ball_size",
}


@dataclass
class BallSNNConfig:
    checkpoint: str = "out/models/ball_small_best.pt"
    training_dir: str = "Training"      # Theodor's scripts (git checkout origin/feat/training -- Training)
    riverops_dir: str = "riverops-0"    # unpacked riverops wheel, used only if riverops is not installed
    min_p: float = 0.55                 # P(ball) below this -> confidence 0 ("nothing seen"). Training
                                        # uses 0.5; on our GenX320 recordings 0.55 cuts false alarms
    min_pixels: int = 20                # fewer event pixels in the frame -> nothing seen, network not run.
                                        # Near-empty frames give P(ball) ~0.5 with the peak in a corner
    zone_top: float = 0.0               # blank image rows above this fraction before the network, as
                                        # the centroid tracker ignores them (people, the room). 0: all
    device: str = "cpu"


def _resolve(p: str) -> Path:
    path = Path(p)
    if path.is_absolute() or path.exists():
        return path
    return REPO / path


def _add_to_path(directory: Path) -> None:
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


def frame_from_window(window: np.ndarray) -> np.ndarray:
    """(T, C, H, W) event counts -> (H, W) uint8, 1 where any event fired: the network input."""
    return (window.reshape(-1, *window.shape[-2:]).sum(axis=0) > 0).astype(np.uint8)


def fullres_config(window_ms: float, height: int = 320, width: int = 320) -> WindowConfig:
    """One time bin, no downsampling: what /reflex/events_fullres carries."""
    return WindowConfig(window_ms=window_ms, n_bins=1, downsample=1, height=height, width=width)


class BallSNNTracker:
    """Checkpoint in, `Target` out. `predict(frame)` takes the 320x320 occupancy frame."""

    def __init__(self, cfg: BallSNNConfig | None = None):
        import torch

        self.cfg = cfg = cfg or BallSNNConfig()
        path = _resolve(cfg.checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"no checkpoint at {path}")
        info = torch.load(path, map_location="cpu", weights_only=True)
        arch = info.get("architecture")
        if arch not in SCRIPTS:
            raise ValueError(f"{path}: architecture {arch!r} unknown; known: {', '.join(SCRIPTS)}")
        try:
            import riverops  # noqa: F401
        except ImportError:
            _add_to_path(_resolve(cfg.riverops_dir))
        training = _resolve(cfg.training_dir)
        if not (training / f"{SCRIPTS[arch]}.py").is_file():
            raise FileNotFoundError(
                f"{training / SCRIPTS[arch]}.py missing: git checkout origin/feat/training -- Training")
        _add_to_path(training)
        importlib.import_module(SCRIPTS[arch])     # the small script patches train_ball_size on import
        self._size = importlib.import_module("train_ball_size")
        self.model, self.time_steps = self._size.load_model(path, cfg.device)
        self.model.eval()
        self.sensor = int(info["sensor"])
        self.window_ms = float(info.get("window_ms", 1.0))
        self.architecture = arch
        self.validation = info.get("validation_metrics", {})
        self._torch = torch
        self.last: dict = {}     # details of the last prediction, for the debug topic

    def predict(self, frame: np.ndarray, t_us: int = 0) -> Target:
        torch, s = self._torch, self.sensor
        if frame.shape != (s, s):
            raise ValueError(f"frame is {frame.shape}, the network wants ({s}, {s})")
        if self.cfg.zone_top > 0:
            frame = frame.copy()
            frame[:int(self.cfg.zone_top * s)] = 0
        pixels = int(np.count_nonzero(frame))
        if pixels < self.cfg.min_pixels:
            self.last = {"pixels": pixels, "p_ball": None, "ms": 0.0}
            return no_target(t_us)
        start = time.perf_counter()
        x = torch.from_numpy(np.ascontiguousarray(frame > 0, dtype=np.float32)).to(self.cfg.device)
        x = x.view(1, 1, 1, s, s).expand(1, self.time_steps, 1, s, s)
        with torch.no_grad():
            counts = self.model(x)
            px, py, p = self._size.decode(counts, self.time_steps, self.model.logit_scales(),
                                          self.model.no_ball_logit)[0].tolist()
        u, v = px / s * 2.0 - 1.0, py / s * 2.0 - 1.0
        self.last = {"x_px": round(px, 1), "y_px": round(py, 1), "p_ball": round(p, 3),
                     "pixels": pixels, "ms":round((time.perf_counter() - start) * 1000, 1)}
        if p < self.cfg.min_p:
            return no_target(t_us)
        return Target(u, v, p, t_us)


def main() -> None:
    import argparse

    from reflex.events.sources import find_takes, read_take

    ap = argparse.ArgumentParser(description="Run the SNN ball detector over recordings.")
    ap.add_argument("paths", nargs="+", help=".h5 files or session folders")
    ap.add_argument("--checkpoint", default=BallSNNConfig.checkpoint)
    ap.add_argument("--step-ms", type=float, default=20.0, help="one frame every step (camera node: 50 Hz)")
    ap.add_argument("--window-ms", type=float, default=20.0,
                    help="as the event camera's fullres_ms (0: the checkpoint's, 2 ms)")
    ap.add_argument("--min-p", type=float, default=BallSNNConfig.min_p)
    ap.add_argument("--zone-top", type=float, default=0.0, help="blank rows above this fraction")
    ap.add_argument("--every", type=int, default=0, help="print every Nth frame (0: summary only)")
    a = ap.parse_args()

    tracker = BallSNNTracker(BallSNNConfig(checkpoint=a.checkpoint, min_p=a.min_p, zone_top=a.zone_top))
    window_ms = a.window_ms or tracker.window_ms
    print(f"{tracker.architecture}, {tracker.time_steps} time steps, {window_ms:g} ms windows")
    wcfg = fullres_config(window_ms, tracker.sensor, tracker.sensor)
    for take in find_takes(a.paths):
        ev = read_take(take.path)
        if len(ev) == 0:
            continue
        ends = np.arange(int(ev["t"][0] + window_ms * 1000), int(ev["t"][-1]), int(a.step_ms * 1000))
        found, us, vs, ms = 0, [], [], []
        for i, t_end in enumerate(ends):
            target = tracker.predict(frame_from_window(make_window(ev, int(t_end), wcfg)), int(t_end))
            if tracker.last["p_ball"] is not None:
                ms.append(tracker.last["ms"])
            if target.found:
                found += 1
                us.append(target.u)
                vs.append(target.v)
            if a.every and i % a.every == 0:
                p_ball = tracker.last["p_ball"]
                print(f"  {(t_end - ev['t'][0]) / 1e6:6.2f} s  p={'-' if p_ball is None else f'{p_ball:.2f}'}  "
                      f"u={target.u:+.2f} v={target.v:+.2f}  {tracker.last['pixels']} px")
        where = f"  median u {np.median(us):+.2f} v {np.median(vs):+.2f}" if us else ""
        print(f"{take.path.parent.name}/{take.path.name}: ball in {found}/{len(ends)} frames"
              f" ({found / max(len(ends), 1):.0%}){where}  ({np.median(ms) if ms else 0:.0f} ms per frame run)")


if __name__ == "__main__":
    main()
