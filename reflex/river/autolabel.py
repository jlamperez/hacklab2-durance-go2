"""Automatic labels for target recordings: where is the target in each 10 ms frame?

A target take (kind folder other than `none`) is labelled in five steps:

1. Session calibration from the empty takes, `none/empty*.bin` (the scene with nothing
   moving, recorded first in every session): pixels that fire in more than `hot_frac` of
   the frames are masked, and the detection threshold is set above the busiest patch of
   pure noise (`noise_factor` x the `noise_pct` percentile of each frame's peak patch).
   Without an empty take, hot pixels come from all `none` takes and the threshold is the
   fixed floor `min_blob`.
2. Per recording, cells of the input grid (4x4 px) with events in more than
   `clutter_frac` of its frames are masked as well (a flickering lamp, a screen, a hand
   waving in one spot). A moving ball passes through a cell; it does not stay.
3. Per frame, events are counted on the network's input grid (80x80), summed over
   `integrate` frames centred on it (a slow ball gives few events per 10 ms) and over a
   `kernel` x `kernel` patch. A candidate is a patch with at least the threshold count that
   is compact: it holds at least `min_compact` of the events of the 3x longer strip through
   it, vertical and horizontal. A ball is compact; the edge of a person or of a door frame
   sweeping past during a camera pan runs on beyond the patch.
4. A tracker follows the target. While it was seen in the last `gate_frames`, only
   candidates within reach of its last position count (`gate_cells` plus `gate_speed`
   cells per frame since); after that, the strongest candidate anywhere is taken.
5. Runs of detections shorter than `min_run` frames are dropped as noise, gaps of up to
   `fill_gap_frames` are bridged, and the x position is binned into zones.
6. Kinds in `ego_kinds` (a drone, filmed while the camera moves) are tracked by their
   motion relative to the scene instead of by strength alone. When the camera moves, the
   whole background fires and its corners are as compact as the target. Per frame, the
   shift of the whole scene over the last `ego_lag` frames is found by cross-correlating
   the active cells (`scene_motion`). A candidate is the target only if its patch is not
   where the scene's shift puts it `ego_lag` frames earlier, but is found well somewhere
   else, at least `ego_min_rel` cells away from there: it moved on its own. While the
   background is silent (camera still) the target is found as in steps 3-4. A track's
   search area moves with the scene. A candidate without motion of its own continues the
   track only within `gate_cells` of where the scene's shift puts the target (a hovering
   drone), and for at most `ego_hold` frames while the scene fires. Limits: the scene shift is one
   translation for the whole image (a pan or tilt; moving forward is not a translation),
   and a target that hangs still in the world while the camera moves is held only
   `ego_hold` frames. Patches are `ego_kernel` cells (a drone is smaller than a ball), and a
   patch that just came in over the image border is not counted as moving on its own.

Corrections per session go in `<session>/overrides.json`, keyed by the path relative to
the session folder, with "*" for every take of the session:

    {"*": {"ignore": [[280, 0, 320, 320]]},
     "ball/clip_003.bin": {"drop": true},
     "ball/clip_007.bin": {"trim": [1.0, null], "zone": "right"}}

- drop: leave the take out.
- trim: [start_s, end_s] to keep (null = open end). Cuts frames and labels alike.
- ignore: rectangles [x0, y0, x1, y1] in sensor pixels, image coordinates (before
  flip_x), whose events are not used to find the target. The network still sees them.
- zone: every frame in which the target is detected gets this label (a take filed
  under the wrong side, or a mount that confused the tracker).
- ego_motion: true / false to switch step 6 on or off for a take, whatever its kind.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .data import (EXTENSIONS, NONE_KIND, FrameConfig, LabelConfig, fill_gaps, frame_counts,
                   read_events)

EMPTY_PREFIX = "empty"
OVERRIDES_FILE = "overrides.json"
OVERRIDE_KEYS = {"drop", "trim", "ignore", "zone", "ego_motion"}
CHUNK = 512                     # frames per vectorised step, bounds memory


# --------------------------------------------------------------------------- calibration

@dataclass
class Calibration:
    hot: np.ndarray             # (sensor, sensor) bool, indexed [y, x]; True = masked
    min_blob: float             # events in one kernel patch to count as a detection
    source: str

    def summary(self) -> str:
        return (f"{int(self.hot.sum())} hot pixels masked, threshold {self.min_blob:.1f} "
                f"events per patch ({self.source})")


def pixel_activity(ev: np.ndarray, cfg: FrameConfig, t0: int, n_frames: int) -> np.ndarray:
    """Number of frames in which each pixel fired, (sensor, sensor) indexed [y, x]."""
    s = cfg.sensor
    f = (ev["t"].astype(np.int64) - t0) // cfg.bin_us
    ok = (f >= 0) & (f < n_frames) & (ev["x"] < s) & (ev["y"] < s)
    pix = ev["y"][ok].astype(np.int64) * s + ev["x"][ok]
    key = np.unique(f[ok] * (s * s) + pix)
    return np.bincount(key % (s * s), minlength=s * s).reshape(s, s)


def _n_frames(ev: np.ndarray, cfg: FrameConfig, t0: int) -> int:
    return int((int(ev["t"][-1]) - t0) // cfg.bin_us) + 1


def _in_sensor(ev: np.ndarray, cfg: FrameConfig) -> np.ndarray:
    return ev[(ev["x"] < cfg.sensor) & (ev["y"] < cfg.sensor)]


def tracking_grid(ev: np.ndarray, cfg: FrameConfig, t0: int, n_frames: int) -> np.ndarray:
    """(T, S, S) int32 event counts, both polarities, image coordinates (never flipped)."""
    return frame_counts(ev, replace(cfg, flip_x=False), t0, n_frames).sum(1, dtype=np.int32)


def integral(a: np.ndarray) -> np.ndarray:
    """Summed-area table of (T, H, W), shape (T, H + 1, W + 1)."""
    T, H, W = a.shape
    c = np.zeros((T, H + 1, W + 1), dtype=np.int32)
    c[:, 1:, 1:] = a.astype(np.int32).cumsum(1).cumsum(2)
    return c


def box_sum(a: np.ndarray, kh: int, kw: int | None = None, table: np.ndarray | None = None) -> np.ndarray:
    """Sum over the kh x kw patch centred on each cell (zero outside the image), a: (T, H, W)."""
    kw = kh if kw is None else kw
    c = integral(a) if table is None else table
    H, W = c.shape[1] - 1, c.shape[2] - 1
    i0, i1 = np.clip(np.arange(H) - kh // 2, 0, H), np.clip(np.arange(H) + kh - kh // 2, 0, H)
    j0, j1 = np.clip(np.arange(W) - kw // 2, 0, W), np.clip(np.arange(W) + kw - kw // 2, 0, W)
    return (c[:, i1][:, :, j1] - c[:, i0][:, :, j1] - c[:, i1][:, :, j0] + c[:, i0][:, :, j0])


def patch_maps(ev: np.ndarray, cfg: FrameConfig, lab: LabelConfig, t0: int, f0: int, nf: int,
               n_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For frames f0 .. f0 + nf: the grid summed over `integrate` frames centred on each,
    its kernel patch sums and its compactness-filtered score (0 where not compact)."""
    h = lab.integrate // 2
    a, b = max(0, f0 - h), min(n_frames, f0 + nf + (lab.integrate - 1 - h))
    g = tracking_grid(ev, cfg, t0 + a * cfg.bin_us, b - a)
    c = np.concatenate([np.zeros((1,) + g.shape[1:], np.int32), g.cumsum(0, dtype=np.int32)])
    f = np.arange(f0, f0 + nf)
    lo, hi = np.clip(f - h, a, b) - a, np.clip(f - h + lab.integrate, a, b) - a
    g = c[hi] - c[lo]
    k, table = lab.kernel, integral(g)
    near = box_sum(g, k, table=table)
    # A ball fills its patch; an edge (a person, a door frame during a pan) runs on past it.
    # Compare with a 3x longer strip in each direction.
    longer = np.maximum(box_sum(g, 3 * k, k, table), box_sum(g, k, 3 * k, table))
    return g, near, np.where(near >= lab.min_compact * longer, near, 0)


def calibrate(session_dir: str | Path, cfg: FrameConfig, lab: LabelConfig) -> Calibration:
    none_dir = Path(session_dir) / NONE_KIND
    files = sorted(p for p in none_dir.glob("*") if p.suffix in EXTENSIONS) if none_dir.is_dir() else []
    empty = [p for p in files if p.name.lower().startswith(EMPTY_PREFIX)]
    s = cfg.sensor
    active, frames, loaded = np.zeros((s, s), np.int64), 0, {}
    for p in empty or files:
        ev = _in_sensor(read_events(p), cfg)
        if len(ev):
            t0 = int(ev["t"][0])
            n = _n_frames(ev, cfg, t0)
            active += pixel_activity(ev, cfg, t0, n)
            frames += n
            if p in empty:
                loaded[p] = (ev, t0, n)
    hot = active > lab.hot_frac * frames if frames else np.zeros((s, s), bool)
    if loaded:
        peaks = []
        for ev, t0, n in loaded.values():
            ev = ev[~hot[ev["y"], ev["x"]]]
            for f0 in range(0, n, CHUNK):
                _, near, _ = patch_maps(ev, cfg, lab, t0, f0, min(CHUNK, n - f0), n)
                peaks.append(near.reshape(len(near), -1).max(1))
        noise = float(np.percentile(np.concatenate(peaks), lab.noise_pct))
        thr = max(float(lab.min_blob), lab.noise_factor * noise)
        source = f"{len(loaded)} empty take(s), noise p{lab.noise_pct:g} = {noise:.1f}"
    elif files:
        thr, source = float(lab.min_blob), f"no empty take: hot pixels from {len(files)} none take(s), fixed threshold"
    else:
        thr, source = float(lab.min_blob), "no none takes: nothing masked, fixed threshold"
    return Calibration(hot, thr, source)


# --------------------------------------------------------------------------- overrides

def read_overrides(session_dir: str | Path, lab: LabelConfig) -> dict:
    path = Path(session_dir) / OVERRIDES_FILE
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected an object keyed by take path")
    for take, ov in data.items():
        bad = set(ov) - OVERRIDE_KEYS
        if bad:
            raise ValueError(f"{path}: {take}: unknown keys {sorted(bad)}, allowed {sorted(OVERRIDE_KEYS)}")
        if "zone" in ov and ov["zone"] not in lab.names:
            raise ValueError(f"{path}: {take}: zone {ov['zone']!r} not in {lab.names}")
        if "trim" in ov and (not isinstance(ov["trim"], list) or len(ov["trim"]) != 2):
            raise ValueError(f"{path}: {take}: trim must be [start_s, end_s]")
        if "ego_motion" in ov and not isinstance(ov["ego_motion"], bool):
            raise ValueError(f"{path}: {take}: ego_motion must be true or false")
        for r in ov.get("ignore", []):
            if len(r) != 4:
                raise ValueError(f"{path}: {take}: ignore rectangles are [x0, y0, x1, y1]")
        if take != "*" and not (Path(session_dir) / take).exists():
            print(f"warning: {path}: {take} does not exist")
    return data


def take_overrides(session_dir: str | Path, take: str | Path, lab: LabelConfig) -> dict:
    """Overrides for one take: its own entry on top of the session-wide "*" entry."""
    data = _session(session_dir, None, lab, need_calibration=False)[1]
    # absolute(), not resolve(): a take may be a symlink to a file outside the session
    rel = Path(os.path.abspath(take)).relative_to(os.path.abspath(session_dir)).as_posix()
    base, own = dict(data.get("*", {})), data.get(rel, {})
    ignore = list(base.get("ignore", [])) + list(own.get("ignore", []))
    base.update(own)
    if ignore:
        base["ignore"] = ignore
    return base


def trim_frames(ov: dict, cfg: FrameConfig, n_frames: int) -> tuple[int, int]:
    start, end = ov.get("trim", [None, None])
    a = 0 if start is None else max(0, int(round(start * 1e6 / cfg.bin_us)))
    b = n_frames if end is None else min(n_frames, int(round(end * 1e6 / cfg.bin_us)))
    return a, max(a, b)


_CACHE: dict = {}


def _session(session_dir, cfg, lab, need_calibration=True):
    key = str(Path(session_dir).resolve())
    ov_key, cal_key = (key, "ov", repr(lab)), (key, "cal", repr(cfg), repr(lab))
    if ov_key not in _CACHE:
        _CACHE[ov_key] = read_overrides(session_dir, lab)
    if need_calibration and cal_key not in _CACHE:
        _CACHE[cal_key] = calibrate(session_dir, cfg, lab)
    return _CACHE.get(cal_key), _CACHE[ov_key]


def session_calibration(session_dir: str | Path, cfg: FrameConfig, lab: LabelConfig) -> Calibration:
    return _session(session_dir, cfg, lab)[0]


def clear_cache() -> None:
    """Forget calibrations and overrides (after editing overrides.json in a running process)."""
    _CACHE.clear()


# --------------------------------------------------------------------------- tracking

@dataclass
class Track:
    x: np.ndarray               # (T,) image x in [0, 1], NaN = no target (after gap filling)
    y: np.ndarray               # (T,) image y in [0, 1], NaN = no target (before gap filling)
    mass: np.ndarray            # (T,) events in the patch, 0 where nothing was detected
    threshold: float
    masked_pixels: int
    camera: np.ndarray | None = None  # (T,) int8 scene state (STILL, MOVING, UNKNOWN) when step 6 ran


def _drop_short_runs(ok: np.ndarray, min_run: int) -> np.ndarray:
    if min_run <= 1 or not ok.any():
        return ok
    d = np.diff(np.concatenate([[0], ok.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    out = ok.copy()
    for a, b in zip(starts, ends):
        if b - a < min_run:
            out[a:b] = False
    return out


def _masked(ev: np.ndarray, cfg: FrameConfig, lab: LabelConfig, t0: int, n_frames: int,
            calib: Calibration | None, ignore) -> tuple[np.ndarray, np.ndarray, float]:
    """Events left after hot pixels, busy cells and ignore rectangles; the mask; the threshold."""
    s, ds = cfg.sensor, cfg.downsample
    ev = _in_sensor(ev, cfg)
    mask = calib.hot.copy() if calib is not None else np.zeros((s, s), bool)
    if n_frames >= lab.clutter_min_frames and len(ev):
        ds_cfg = replace(cfg, sensor=s // ds)
        cells = ev.copy()
        cells["x"] //= ds
        cells["y"] //= ds
        busy = pixel_activity(cells, ds_cfg, t0, n_frames) > lab.clutter_frac * n_frames
        mask |= np.kron(busy, np.ones((ds, ds), bool))[:s, :s]
    for x0, y0, x1, y1 in ignore:
        mask[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)] = True
    thr = calib.min_blob if calib is not None else float(lab.min_blob)
    return ev[~mask[ev["y"], ev["x"]]], mask, thr


def _centroid(g: np.ndarray, cy: int, cx: int, cfg: FrameConfig, lab: LabelConfig) -> tuple[float, float]:
    """Event-weighted centre of the kernel patch at (cy, cx) of one frame's grid, image [0, 1]."""
    k, half = lab.kernel, lab.kernel // 2
    patch = g[max(0, cy - half):cy + k - half, max(0, cx - half):cx + k - half]
    py, px = np.nonzero(patch)
    w = patch[py, px].astype(np.float64)
    px = px + max(0, cx - half)
    py = py + max(0, cy - half)
    ds, s = cfg.downsample, cfg.sensor
    return (np.dot(w, px) / w.sum() + 0.5) * ds / s, (np.dot(w, py) / w.sum() + 0.5) * ds / s


def track_target(ev: np.ndarray, cfg: FrameConfig, lab: LabelConfig, t0: int, n_frames: int,
                 calib: Calibration | None = None, ignore=()) -> Track:
    """Follow the target through a recording (steps 1-5 of the module docstring)."""
    ev, mask, thr = _masked(ev, cfg, lab, t0, n_frames, calib, ignore)

    xs = np.full(n_frames, np.nan, np.float32)
    ys = np.full(n_frames, np.nan, np.float32)
    mass = np.zeros(n_frames, np.float32)
    last, last_f = None, -10 ** 9
    for f0 in range(0, n_frames, CHUNK):
        g, _, score = patch_maps(ev, cfg, lab, t0, f0, min(CHUNK, n_frames - f0), n_frames)
        H, W = score.shape[1:]
        for i in range(len(g)):
            f = f0 + i
            sc = score[i]
            if last is not None and f - last_f <= lab.gate_frames:
                r = int(np.ceil(lab.gate_cells + lab.gate_speed * (f - last_f)))
                ya, yb = max(0, last[0] - r), min(H, last[0] + r + 1)
                xa, xb = max(0, last[1] - r), min(W, last[1] + r + 1)
                sub = sc[ya:yb, xa:xb]
                j = int(sub.argmax())
                cy, cx = ya + j // sub.shape[1], xa + j % sub.shape[1]
            else:
                j = int(sc.argmax())
                cy, cx = j // W, j % W
            m = sc[cy, cx]
            if m < thr:
                continue
            xs[f], ys[f] = _centroid(g[i], cy, cx, cfg, lab)
            mass[f] = m
            last, last_f = (cy, cx), f

    ok = _drop_short_runs(~np.isnan(xs), lab.min_run)
    xs[~ok], ys[~ok], mass[~ok] = np.nan, np.nan, 0
    return Track(fill_gaps(xs, lab.fill_gap_frames), ys, mass, thr, int(mask.sum()))


# --------------------------------------------------------------------------- moving camera

STILL, MOVING, UNKNOWN = 0, 1, 2    # scene state per frame: background silent / shift known / busy, no shift
EGO_INTEGRATE = 3                    # frames OR-ed into one activity map (a drone gives few events)


def active_maps(ev: np.ndarray, cfg: FrameConfig, t0: int, n_frames: int) -> np.ndarray:
    """(T, S, S) bool: cells with an event within EGO_INTEGRATE frames centred on each frame."""
    h, s = EGO_INTEGRATE // 2, cfg.size
    out = np.zeros((n_frames, s, s), bool)
    for f0 in range(0, n_frames, CHUNK):
        f1 = min(n_frames, f0 + CHUNK)
        a, b = max(0, f0 - h), min(n_frames, f1 + EGO_INTEGRATE - 1 - h)
        g = tracking_grid(ev, cfg, t0 + a * cfg.bin_us, b - a) > 0
        c = np.concatenate([np.zeros((1, s, s), np.int32), g.cumsum(0, dtype=np.int32)])
        f = np.arange(f0, f1)
        lo, hi = np.clip(f - h, a, b) - a, np.clip(f - h + EGO_INTEGRATE, a, b) - a
        out[f0:f1] = (c[hi] - c[lo]) > 0
    return out


@dataclass
class SceneMotion:
    shift: np.ndarray           # (T, 2) float32: (dy, dx) cells the scene moved over the last ego_lag frames
    state: np.ndarray           # (T,) int8: STILL, MOVING (shift known) or UNKNOWN (busy, no shift found)
    match: np.ndarray           # (T,) share of the active cells the shift explains


def scene_motion(A: np.ndarray, lab: LabelConfig) -> SceneMotion:
    """Shift of the whole scene per frame: the cross-correlation peak of the active cells
    with those ego_lag frames earlier (FFT, unbiased for the shrinking overlap)."""
    T, H, W = A.shape
    L, S = lab.ego_lag, lab.ego_search
    n = A.reshape(T, -1).sum(1)
    shift = np.zeros((T, 2), np.float32)
    match = np.zeros(T, np.float32)
    state = np.where(n >= lab.ego_min_cells, UNKNOWN, STILL).astype(np.int8)
    P = int(2 ** np.ceil(np.log2(max(H, W) + S + 1)))      # zero padding: no wrap-around
    d = np.arange(-S, S + 1)
    overlap = np.outer(H - np.abs(d), W - np.abs(d)) / (H * W)
    for f0 in range(L, T, CHUNK):
        f = np.arange(f0, min(T, f0 + CHUNK))
        f = f[(n[f] >= lab.ego_min_cells) & (n[f - L] >= lab.ego_min_cells)]
        if not len(f):
            continue
        a = np.fft.rfft2(A[f].astype(np.float32), s=(P, P))
        b = np.fft.rfft2(A[f - L].astype(np.float32), s=(P, P))
        c = np.fft.irfft2(a * np.conj(b), s=(P, P))[:, d % P][:, :, d % P]   # c[dy, dx]
        iy, ix = np.unravel_index((c / overlap).reshape(len(f), -1).argmax(1), overlap.shape)
        m = c[np.arange(len(f)), iy, ix] / np.minimum(n[f], n[f - L])
        match[f] = m
        ok = m >= lab.ego_min_match
        shift[f[ok]] = np.stack([d[iy[ok]], d[ix[ok]]], 1)
        state[f[ok]] = MOVING
    return SceneMotion(shift, state, match)


def _dilate(a: np.ndarray) -> np.ndarray:
    """3x3 binary dilation of a (H, W) bool map."""
    p = np.pad(a, 1)
    out = np.zeros_like(a)
    for dy in range(3):
        for dx in range(3):
            out |= p[dy:dy + a.shape[0], dx:dx + a.shape[1]]
    return out


def own_motion(now: np.ndarray, before_dilated: np.ndarray, cy: int, cx: int, shift,
               lab: LabelConfig) -> tuple[bool, float, float]:
    """Did the patch at (cy, cx) move on its own over the last ego_lag frames?
    `now` is the frame's activity map, `before_dilated` the dilated map ego_lag frames
    earlier. Returns (moved on its own, match where the scene shift puts it, best match)."""
    r, R = lab.ego_patch // 2, lab.ego_search + lab.ego_own
    H, W = now.shape
    py, px = cy - shift[0], cx - shift[1]           # where the scene shift puts it ego_lag frames ago
    if not (r <= py < H - r and r <= px < W - r):   # it came in over the border: nothing to compare
        return False, 0.0, 0.0
    p = np.pad(now, r)[cy:cy + 2 * r + 1, cx:cx + 2 * r + 1].astype(np.float32)
    if p.sum() == 0:
        return False, 0.0, 0.0
    win = np.pad(before_dilated, R + r)[cy:cy + 2 * (R + r) + 1, cx:cx + 2 * (R + r) + 1]
    views = np.lib.stride_tricks.sliding_window_view(win, p.shape)     # [i, j]: displacement R - i, R - j
    q = np.einsum("ijkl,kl->ij", views, p, dtype=np.float32)[::-1, ::-1] / p.sum()   # q[dy + R, dx + R]
    sy, sx = int(round(shift[0])) + R, int(round(shift[1])) + R
    q_scene = float(q[max(0, sy - 1):sy + 2, max(0, sx - 1):sx + 2].max())
    j = int(q.argmax())
    by, bx = divmod(j, q.shape[1])
    q_best = float(q[by, bx])
    rel = float(np.hypot(by - sy, bx - sx))
    moved = q_scene < lab.ego_scene_match and q_best >= lab.ego_self_match and rel >= lab.ego_min_rel
    return moved, q_scene, q_best


def _peaks(sc: np.ndarray, thr: float, n: int, radius: int) -> list[tuple[int, int]]:
    """Up to n strongest cells of a score map at or above thr, at least `radius` apart."""
    sc = sc.astype(np.float32, copy=True)
    out = []
    for _ in range(n):
        j = int(sc.argmax())
        cy, cx = divmod(j, sc.shape[1])
        if sc[cy, cx] < thr:
            break
        out.append((cy, cx))
        sc[max(0, cy - radius):cy + radius + 1, max(0, cx - radius):cx + radius + 1] = -1
    return out


def track_target_ego(ev: np.ndarray, cfg: FrameConfig, lab: LabelConfig, t0: int, n_frames: int,
                     calib: Calibration | None = None, ignore=()) -> Track:
    """Follow a target that moves relative to the scene while the camera may move (step 6).
    Patches are `ego_kernel` cells; the threshold is the session's (calibrated for `kernel`)."""
    ev, mask, thr = _masked(ev, cfg, lab, t0, n_frames, calib, ignore)
    lab = replace(lab, kernel=lab.ego_kernel)
    A = active_maps(ev, cfg, t0, n_frames)
    motion = scene_motion(A, lab)
    vel = np.where((motion.state == MOVING)[:, None], motion.shift / lab.ego_lag, 0.0)
    drift = np.cumsum(vel, 0)                       # scene position per frame, cells
    L, radius = lab.ego_lag, lab.kernel // 2

    xs = np.full(n_frames, np.nan, np.float32)
    ys = np.full(n_frames, np.nan, np.float32)
    mass = np.zeros(n_frames, np.float32)
    last, last_f, own_f = None, -10 ** 9, -10 ** 9  # own_f: last frame the target showed own motion
    before_cache: dict = {}

    def moved(f: int, c: tuple[int, int]) -> bool:
        if f < L or motion.state[f] != MOVING:
            return False
        if f not in before_cache:
            before_cache.clear()
            before_cache[f] = _dilate(A[f - L])
        return own_motion(A[f], before_cache[f], c[0], c[1], motion.shift[f], lab)[0]

    for f0 in range(0, n_frames, CHUNK):
        g, _, score = patch_maps(ev, cfg, lab, t0, f0, min(CHUNK, n_frames - f0), n_frames)
        for i in range(len(g)):
            f = f0 + i
            cands = _peaks(score[i], thr, lab.ego_candidates, radius)
            if not cands:
                continue
            st = motion.state[f]
            if last is not None and f - last_f <= lab.gate_frames:
                py, px = np.asarray(last, float) + drift[f] - drift[last_f]
                r = lab.gate_cells + lab.gate_speed * (f - last_f)
                cands = [c for c in cands if np.hypot(c[0] - py, c[1] - px) <= r]
                if not cands:
                    continue
                own = next((c for c in cands if moved(f, c)), None) if st == MOVING else None
                if own is not None or st == STILL:
                    pick = own or cands[0]
                    own_f = f
                else:
                    # Without motion of its own, a hovering target sits where the scene put it:
                    # only that close, not anywhere in the widening gate (a background corner).
                    dist, pick = min((np.hypot(c[0] - py, c[1] - px), c) for c in cands)
                    if dist > lab.gate_cells:
                        continue
                    if f - own_f > lab.ego_hold:    # background firing, no own motion for too long
                        last = None
                        continue
            else:
                if st == STILL:
                    pick = cands[0]
                elif st == MOVING:
                    pick = next((c for c in cands if moved(f, c)), None)
                    if pick is None:
                        continue
                else:
                    continue                        # busy scene, shift unknown: cannot tell
                own_f = f
            cy, cx = pick
            xs[f], ys[f] = _centroid(g[i], cy, cx, cfg, lab)
            mass[f] = score[i][cy, cx]
            last, last_f = pick, f

    ok = _drop_short_runs(~np.isnan(xs), lab.min_run)
    xs[~ok], ys[~ok], mass[~ok] = np.nan, np.nan, 0
    return Track(fill_gaps(xs, lab.fill_gap_frames), ys, mass, thr, int(mask.sum()), motion.state)


def uses_ego_motion(kind: str, lab: LabelConfig, overrides: dict) -> bool:
    return bool(overrides.get("ego_motion", kind in lab.ego_kinds))


def track_recording(path: str | Path, cfg: FrameConfig, lab: LabelConfig, ev: np.ndarray,
                    t0: int, n_frames: int, overrides: dict | None = None) -> Track:
    """Track the target of a take that lives in <session>/<kind>/, with the session's
    calibration and the take's ignore rectangles. Kinds in `lab.ego_kinds` (or takes with
    "ego_motion": true) are tracked relative to the scene's motion (step 6)."""
    session_dir = Path(path).parent.parent
    calib = session_calibration(session_dir, cfg, lab)
    ov = overrides if overrides is not None else take_overrides(session_dir, path, lab)
    track = track_target_ego if uses_ego_motion(Path(path).parent.name, lab, ov) else track_target
    return track(ev, cfg, lab, t0, n_frames, calib, ov.get("ignore", ()))
