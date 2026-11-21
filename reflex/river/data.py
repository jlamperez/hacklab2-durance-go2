"""River recordings -> labelled frame windows.

Recording format (Durance's `.bin`, as read in their Motion Buddy notebook): a 4-byte
big-endian payload size, then records of (x: >u2, y: >u2, polarity: u1, timestamp: >i8 µs).
ASSUMPTION: `river-tools capture` writes the same format. `read_river_bin` raises if the
header does not match, so a different format fails loudly instead of silently.
Recordings converted to HDF5 (`format: reflex-events-v1`, events/t,x,y,p) load too.

Folder layout (one recording session = one folder, kind = subfolder):

    data/river/<session>/<kind>/*.bin      kind: ball, drone, none (or any target name)

`none` recordings are labelled `none` everywhere: people walking, hands, the camera
moving, an empty room. Target recordings are labelled per frame: from `<name>.gt.npz`
when it exists (synthetic data), otherwise automatically by the tracker in `autolabel`
(record those with one moving target and a still camera, and one `none/empty*.bin` take
of the still scene per session for the noise calibration). Per-take corrections live in
`<session>/overrides.json` (see `autolabel`). Split train / val / test by session,
never by window.

Labels are in the dog frame: 0 = none, 1..n_bins = zones from the dog's left to its right.
With the camera looking forward and not mirrored, image left is the dog's left;
`flip_x` handles a mirrored mount.
"""
from __future__ import annotations

import argparse
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

RIVER_DTYPE = np.dtype([("x", ">u2"), ("y", ">u2"), ("p", "u1"), ("t", ">i8")])
EVENT_DTYPE = np.dtype([("t", "<i8"), ("x", "<u2"), ("y", "<u2"), ("p", "u1")])  # as reflex.events.emu
NONE_KIND = "none"


# --------------------------------------------------------------------------- I/O

def read_river_bin(path: str | Path) -> np.ndarray:
    raw = Path(path).read_bytes()
    if len(raw) < 4:
        raise ValueError(f"{path}: too short for a river recording")
    payload = struct.unpack("!I", raw[:4])[0]
    if payload != len(raw) - 4 or payload % RIVER_DTYPE.itemsize:
        raise ValueError(f"{path}: header says {payload} bytes, file has {len(raw) - 4}; "
                         f"not the river .bin format this reader expects")
    rec = np.frombuffer(raw, dtype=RIVER_DTYPE, offset=4)
    ev = np.empty(len(rec), dtype=EVENT_DTYPE)
    for k in ("t", "x", "y", "p"):
        ev[k] = rec[k]
    if len(ev) and (np.diff(ev["t"]) < 0).any():
        ev = ev[np.argsort(ev["t"], kind="stable")]
    return ev


def read_h5(path: str | Path) -> np.ndarray:
    """HDF5 converted from a river .bin (root attr format 'reflex-events-v1', datasets
    events/t (µs), events/x, events/y, events/p)."""
    import h5py

    with h5py.File(path, "r") as f:
        fmt = f.attrs.get("format", "")
        if fmt != "reflex-events-v1":
            raise ValueError(f"{path}: unknown HDF5 event format {fmt!r}")
        g = f["events"]
        ev = np.empty(len(g["t"]), dtype=EVENT_DTYPE)
        for k in ("t", "x", "y", "p"):
            ev[k] = g[k][:]
    if len(ev) and (np.diff(ev["t"]) < 0).any():
        ev = ev[np.argsort(ev["t"], kind="stable")]
    return ev


EXTENSIONS = (".bin", ".h5")


def read_events(path: str | Path) -> np.ndarray:
    return read_h5(path) if Path(path).suffix == ".h5" else read_river_bin(path)


def write_river_bin(path: str | Path, ev: np.ndarray) -> None:
    rec = np.empty(len(ev), dtype=RIVER_DTYPE)
    for k in ("t", "x", "y", "p"):
        rec[k] = ev[k]
    payload = rec.tobytes()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(struct.pack("!I", len(payload)) + payload)


# --------------------------------------------------------------------------- frames

@dataclass
class FrameConfig:
    """How events become network input. Must match what the board feeds the SNN
    (metavision_camera.json: `frequency` 100 Hz -> bin_us 10_000, `downsampling` 4)."""
    sensor: int = 320
    bin_us: int = 10_000
    downsample: int = 4
    # Value of one input pixel. ASSUMPTION to confirm with Durance: what despot writes.
    #   mean: counts / downsample**2 (the notebook's area interpolation)
    #   sum: raw counts, binary: 1 if any event
    norm: str = "mean"
    flip_x: bool = False

    @property
    def size(self) -> int:
        return self.sensor // self.downsample


def frame_counts(ev: np.ndarray, cfg: FrameConfig, t0: int | None = None,
                 n_frames: int | None = None) -> np.ndarray:
    """Event counts per (frame, polarity, y, x) at the downsampled size, uint8 (clipped)."""
    t0 = int(ev["t"][0]) if t0 is None else t0
    idx = (ev["t"].astype(np.int64) - t0) // cfg.bin_us
    if n_frames is None:
        n_frames = int(idx.max()) + 1 if len(idx) else 0
    s = cfg.size
    keep = (idx >= 0) & (idx < n_frames) & (ev["x"] < cfg.sensor) & (ev["y"] < cfg.sensor) & (ev["p"] < 2)
    idx = idx[keep]
    x = ev["x"][keep].astype(np.int64) // cfg.downsample
    if cfg.flip_x:
        x = s - 1 - x
    y = ev["y"][keep].astype(np.int64) // cfg.downsample
    p = ev["p"][keep].astype(np.int64)
    out = np.zeros((n_frames, 2, s, s), dtype=np.uint8)
    chunk = 500                                   # frames per bincount, bounds memory
    for f0 in range(0, n_frames, chunk):
        f1 = min(n_frames, f0 + chunk)
        a, b = np.searchsorted(idx, [f0, f1])     # idx is sorted because t is
        flat = (((idx[a:b] - f0) * 2 + p[a:b]) * s + y[a:b]) * s + x[a:b]
        counts = np.bincount(flat, minlength=(f1 - f0) * 2 * s * s)
        out[f0:f1] = np.minimum(counts, 255).reshape(f1 - f0, 2, s, s)
    return out


def normalize(counts: np.ndarray, cfg: FrameConfig) -> np.ndarray:
    c = counts.astype(np.float32)
    if cfg.norm == "mean":
        return c / cfg.downsample ** 2
    if cfg.norm == "binary":
        return (c > 0).astype(np.float32)
    if cfg.norm == "sum":
        return c
    raise ValueError(f"unknown norm {cfg.norm!r}")


# --------------------------------------------------------------------------- labels

@dataclass
class LabelConfig:
    n_bins: int = 3
    method: str = "blob"        # blob: calibrated tracker (autolabel); median: median x of all events
    min_events: int = 30        # median: events per frame, full resolution, to call the target visible
    margin: float = 0.0         # ignore this fraction at each image edge when binning
    fill_gap_frames: int = 10   # bridge shorter gaps (a ball slowing at the top of a bounce)
    # blob tracker, see reflex/river/autolabel.py. Sizes in cells of the input grid (4 px).
    min_blob: int = 8           # events in a patch: floor of the calibrated threshold
    kernel: int = 11            # patch size: 44 px, about one ball
    integrate: int = 1          # frames summed (centred) before detecting; 3 finds slow balls but more clutter
    min_compact: float = 0.55   # share of the events of a 3x longer strip (either axis) the patch must hold
    gate_frames: int = 20       # how long a lost target is searched for near its last position
    gate_cells: float = 6.0     # search radius right after the last detection
    gate_speed: float = 3.0     # added radius per frame since (3 cells = 12 px per 10 ms)
    min_run: int = 3            # shorter runs of detections are dropped as noise
    hot_frac: float = 0.2       # pixel busy in more frames of the empty takes: masked
    clutter_frac: float = 0.3   # 4x4 cell busy in more frames of one recording: masked there
    clutter_min_frames: int = 300  # only for recordings of at least 3 s
    noise_factor: float = 2.0   # threshold = noise_factor x noise_pct percentile of empty patches
    noise_pct: float = 99.0
    # Moving camera: target kinds tracked by their motion relative to the scene (autolabel step 6).
    # A drone take has the camera moving; ball takes keep the tracker above unchanged.
    ego_kinds: tuple = ("drone",)
    ego_kernel: int = 5         # patch size for these kinds: 20 px, about one drone at a few metres
    ego_lag: int = 20           # frames over which target and scene motion are compared (200 ms)
    ego_search: int = 30        # largest scene shift per ego_lag, cells (120 px per 200 ms)
    ego_own: int = 10           # largest target motion relative to the scene per ego_lag, cells
    ego_min_cells: int = 150    # active cells (of 80x80) from which the background counts as firing
    ego_min_match: float = 0.3  # share of active cells the scene shift must explain to be trusted
    ego_patch: int = 7          # cells around a candidate compared with ego_lag frames earlier
    ego_scene_match: float = 0.35  # patch found this well where the scene shift puts it: background
    ego_self_match: float = 0.5    # patch must be found at least this well elsewhere: it moved itself
    ego_min_rel: float = 2.0    # cells per ego_lag the target must move relative to the scene
    ego_hold: int = 30          # frames a track may go on without own motion while the scene fires
    ego_candidates: int = 6     # strongest patches per frame tested

    @property
    def names(self) -> list[str]:
        if self.n_bins == 3:
            return ["none", "left", "center", "right"]
        return ["none"] + [f"bin{i}" for i in range(self.n_bins)]

    @property
    def n_classes(self) -> int:
        return self.n_bins + 1


def centroid_track(ev: np.ndarray, cfg: FrameConfig, lab: LabelConfig, t0: int,
                   n_frames: int) -> np.ndarray:
    """Per frame: median event x in [0, 1] (image coordinates), NaN if too few events."""
    idx = (ev["t"].astype(np.int64) - t0) // cfg.bin_us
    keep = (idx >= 0) & (idx < n_frames)
    idx, x = idx[keep], ev["x"][keep].astype(np.float32)
    out = np.full(n_frames, np.nan, dtype=np.float32)
    if not len(idx):
        return out
    bounds = np.searchsorted(idx, np.arange(n_frames + 1))
    for f in range(n_frames):
        a, b = bounds[f], bounds[f + 1]
        if b - a >= lab.min_events:
            out[f] = np.median(x[a:b]) / cfg.sensor
    return fill_gaps(out, lab.fill_gap_frames)


def fill_gaps(xn: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly interpolate NaN runs of at most max_gap frames that have a value on both sides."""
    out = xn.copy()
    ok = np.flatnonzero(~np.isnan(xn))
    for a, b in zip(ok[:-1], ok[1:]):
        if 1 < b - a <= max_gap + 1:
            out[a + 1:b] = np.interp(np.arange(a + 1, b), [a, b], [xn[a], xn[b]])
    return out


def gt_track(gt: dict, cfg: FrameConfig, t0: int, n_frames: int) -> np.ndarray:
    """Ground truth (t_us, x_px, visible) resampled at the end of each frame; NaN = none."""
    t_end = t0 + (np.arange(n_frames) + 1) * cfg.bin_us - 1
    j = np.clip(np.searchsorted(gt["t_us"], t_end, side="right") - 1, 0, len(gt["t_us"]) - 1)
    x = gt["x"][j].astype(np.float32) / cfg.sensor
    x[~gt["visible"][j].astype(bool)] = np.nan
    return x


def bin_labels(xn: np.ndarray, cfg: FrameConfig, lab: LabelConfig) -> np.ndarray:
    """Normalised image x (NaN = none) -> class per frame (0 none, 1.. from the dog's left)."""
    x = 1.0 - xn if cfg.flip_x else xn
    x = (x - lab.margin) / max(1e-6, 1.0 - 2 * lab.margin)
    b = np.clip(np.floor(x * lab.n_bins), 0, lab.n_bins - 1)
    out = np.zeros(len(xn), dtype=np.int64)
    ok = ~np.isnan(xn)
    out[ok] = b[ok].astype(np.int64) + 1
    return out


# --------------------------------------------------------------------------- recordings

@dataclass
class Recording:
    path: Path
    session: str
    kind: str
    counts: np.ndarray          # (T, 2, S, S) uint8
    labels: np.ndarray          # (T,) int64
    track: object = None        # autolabel.Track when the tracker labelled it (before trim)
    source: str = "none"        # where the labels came from: none, gt, median, blob
    trim: tuple = (0, 0)        # frames kept, as (start, end) of the untrimmed recording
    overrides: dict = field(default_factory=dict)


def find_recordings(root: str | Path) -> list[tuple[Path, str, str]]:
    """(path, session, kind) for every root/<session>/<kind>/*.bin or *.h5. When both
    `x.bin` and its converted `x.h5` exist, only the .bin is used."""
    out = []
    for p in sorted(Path(root).glob("*/*/*")):
        if p.name.startswith(".") or p.suffix not in EXTENSIONS:
            continue
        if p.suffix == ".h5" and p.with_suffix(".bin").exists():
            continue
        out.append((p, p.parent.parent.name, p.parent.name))
    return out


def load_recording(path: Path, session: str, kind: str, cfg: FrameConfig,
                   lab: LabelConfig, targets: tuple[str, ...], use_gt: bool = True) -> Recording | None:
    """Frames and labels of one take; None if it is empty or dropped in overrides.json.
    `use_gt=False` ignores `.gt.npz` (to measure the auto-labels against it)."""
    from . import autolabel

    path = Path(path)
    ov = autolabel.take_overrides(path.parent.parent, path, lab)
    if ov.get("drop"):
        return None
    ev = read_events(path)
    if len(ev) == 0:
        return None
    t0 = int(ev["t"][0])
    counts = frame_counts(ev, cfg, t0)
    n = len(counts)
    gt_path = path.with_suffix(".gt.npz")
    track, source = None, "none"
    if kind not in targets:                 # none, and targets not trained in this run
        labels = np.zeros(n, dtype=np.int64)
    elif use_gt and gt_path.exists():
        g = np.load(gt_path)
        labels, source = bin_labels(gt_track({k: g[k] for k in g.files}, cfg, t0, n), cfg, lab), "gt"
    elif lab.method == "median":
        labels, source = bin_labels(centroid_track(ev, cfg, lab, t0, n), cfg, lab), "median"
    elif lab.method == "blob":
        track = autolabel.track_recording(path, cfg, lab, ev, t0, n, ov)
        labels, source = bin_labels(track.x, cfg, lab), "blob"
    else:
        raise ValueError(f"unknown label method {lab.method!r}")
    if "zone" in ov and kind in targets:
        labels[labels > 0] = lab.names.index(ov["zone"])
    a, b = autolabel.trim_frames(ov, cfg, n)
    return Recording(path, session, kind, counts[a:b], labels[a:b], track, source, (a, b), ov)


# --------------------------------------------------------------------------- windows

@dataclass
class WindowConfig:
    frames: int = 40           # input clip length (400 ms at 10 ms frames)
    readout: int = 20          # spikes are counted over the last `readout` frames
    label_frames: int = 5      # label = majority over the last frames of the clip
    stride: int = 5


def window_label(labels: np.ndarray, end: int, k: int) -> int:
    seg = labels[max(0, end - k):end]
    return int(np.bincount(seg).argmax())


def make_index(recs: list[Recording], win: WindowConfig) -> list[tuple[int, int, int]]:
    """(recording index, end frame exclusive, label) for every window."""
    idx = []
    for r, rec in enumerate(recs):
        for end in range(win.frames, len(rec.counts) + 1, win.stride):
            idx.append((r, end, window_label(rec.labels, end, win.label_frames)))
    return idx


def mirror_label(label: int, n_bins: int) -> int:
    return 0 if label == 0 else n_bins + 1 - label


try:
    import torch
    from torch.utils.data import Dataset

    class WindowDataset(Dataset):
        def __init__(self, recs: list[Recording], cfg: FrameConfig, lab: LabelConfig,
                     win: WindowConfig, augment: bool = False, max_shift: int = 2):
            self.recs, self.cfg, self.lab, self.win = recs, cfg, lab, win
            self.index = make_index(recs, win)
            self.augment, self.max_shift = augment, max_shift

        def __len__(self) -> int:
            return len(self.index)

        def labels(self) -> np.ndarray:
            return np.array([i[2] for i in self.index], dtype=np.int64)

        def __getitem__(self, i):
            r, end, label = self.index[i]
            x = normalize(self.recs[r].counts[end - self.win.frames:end], self.cfg)
            if self.augment:
                if np.random.rand() < 0.5:          # left/right mirror: labels mirror too
                    x = x[..., ::-1]
                    label = mirror_label(label, self.lab.n_bins)
                if self.max_shift:
                    s = self.max_shift
                    dy, dx = np.random.randint(-s, s + 1, size=2)
                    x = np.roll(x, (dy, dx), axis=(-2, -1))
                    if dy > 0: x[..., :dy, :] = 0
                    if dy < 0: x[..., dy:, :] = 0
                    if dx > 0: x[..., :, :dx] = 0
                    if dx < 0: x[..., :, dx:] = 0
            return torch.from_numpy(np.ascontiguousarray(x)), label
except ImportError:          # data inspection works without PyTorch
    pass


# --------------------------------------------------------------------------- CLI

def _check(paths: list[str], cfg: FrameConfig, lab: LabelConfig) -> None:
    """Print duration, event rate and auto-label histogram of each recording."""
    files = []
    for p in map(Path, paths):
        files += sorted(f for f in p.rglob("*") if f.suffix in EXTENSIONS) if p.is_dir() else [p]
    if lab.method == "blob":
        from .autolabel import session_calibration
        for sdir in sorted({f.parent.parent for f in files}):
            print(f"{sdir}: {session_calibration(sdir, cfg, lab).summary()}")
    for f in files:
        kind = f.parent.name
        rec = load_recording(f, f.parent.parent.name, kind, cfg, lab,
                             targets=() if kind == NONE_KIND else (kind,))
        if rec is None:
            print(f"{f}: empty or dropped in overrides.json")
            continue
        n = len(rec.counts)
        hist = np.bincount(rec.labels, minlength=lab.n_classes) / max(1, n)
        ev_rate = rec.counts.sum() / max(1e-9, n * cfg.bin_us * 1e-6)
        print(f"{f}  kind={kind}  {n * cfg.bin_us / 1e6:.1f} s  ~{ev_rate / 1e3:.0f}k ev/s  "
              + "  ".join(f"{name}={h:.0%}" for name, h in zip(lab.names, hist)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect river recordings and their auto-labels")
    ap.add_argument("paths", nargs="+", help=".bin files or folders")
    ap.add_argument("--n-bins", type=int, default=3)
    ap.add_argument("--method", default="blob", choices=["blob", "median"])
    ap.add_argument("--min-events", type=int, default=LabelConfig.min_events, help="median method")
    ap.add_argument("--min-blob", type=int, default=LabelConfig.min_blob, help="blob method: threshold floor")
    ap.add_argument("--flip-x", action="store_true")
    a = ap.parse_args()
    _check(a.paths, FrameConfig(flip_x=a.flip_x),
           LabelConfig(n_bins=a.n_bins, method=a.method, min_events=a.min_events, min_blob=a.min_blob))


if __name__ == "__main__":
    main()
