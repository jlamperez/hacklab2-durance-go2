"""Auto-labelled recordings -> ball-position HDF5 files for the ball SNN in `Training/`.

The tracker in `autolabel` finds the ball per 10 ms frame (session calibration, busy-cell
masking, `overrides.json`). This writes what it found as per-frame ball positions in
sensor pixels, in the label layout of the Tübingen table-tennis dataset that
`Training/train_ball_size.py` was built on:

    out/ball/<session>__<take>.h5      target takes, one file per take
    out/no_ball/<session>__<take>.h5   `none` takes, unchanged events, no labels (for --no-ball)
    out/labels.csv                     per take: frames labelled, seconds, label source

Each file keeps the recording's `reflex-events-v1` layout (`events/{t,x,y,p}`, t in µs,
all root attributes) and adds, for target takes only:

    gt_x, gt_y   float64, sensor pixels (0..320), event-weighted centre of the tracked patch
    gt_t         int64 µs, same clock as events/t: the centre of the 10 ms frame

Only frames where the tracker detected the ball get a row (gaps the tracker bridges are
left out, so every label has a measured y). Takes dropped in `overrides.json` are
skipped; `trim` limits the labels, not the events. `ignore` rectangles apply.

What the label is: the centre of the ball's event blob over 10 ms, not the physical
ball's centre. For a moving ball the blob is the leading and trailing edges, so expect a
few pixels of offset along the direction of motion. Check with `reflex.river.review`.

    python -m reflex.river.export_positions data/river --out data/river_positions
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

from . import autolabel
from .data import NONE_KIND, FrameConfig, LabelConfig, find_recordings, read_events


def export_take(path: Path, kind: str, out: Path, cfg: FrameConfig, lab: LabelConfig) -> dict:
    import h5py

    ov = autolabel.take_overrides(path.parent.parent, path, lab)
    ev = read_events(path)
    t0 = int(ev["t"][0])
    n = int((int(ev["t"][-1]) - t0) // cfg.bin_us) + 1
    track = autolabel.track_recording(path, cfg, lab, ev, t0, n, ov)
    a, b = autolabel.trim_frames(ov, cfg, n)
    found = np.zeros(n, bool)
    found[a:b] = np.isfinite(track.x[a:b]) & np.isfinite(track.y[a:b]) & (track.mass[a:b] > 0)
    f = np.flatnonzero(found)
    gx = track.x[f].astype(np.float64) * cfg.sensor
    gy = track.y[f].astype(np.float64) * cfg.sensor
    gt = t0 + f.astype(np.int64) * cfg.bin_us + cfg.bin_us // 2

    shutil.copyfile(path, out)
    with h5py.File(out, "r+") as h:
        h.attrs.setdefault("width", cfg.sensor)
        h.attrs.setdefault("height", cfg.sensor)
        h.attrs.setdefault("t_unit", "us")
        h.create_dataset("gt_x", data=gx)
        h.create_dataset("gt_y", data=gy)
        h.create_dataset("gt_t", data=gt)
        h.attrs["gt_source"] = "reflex.river.autolabel tracker, 10 ms frames, patch centroid"
        h.attrs["gt_threshold"] = float(track.threshold)
    return {"frames": len(f), "seconds": len(f) * cfg.bin_us / 1e6,
            "of_frames": b - a, "threshold": round(float(track.threshold), 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", help="folder with <session>/<kind>/*.h5")
    ap.add_argument("--out", default="data/river_positions")
    ap.add_argument("--targets", nargs="+", default=["ball"], help="kinds to label (others skipped)")
    args = ap.parse_args()

    cfg, lab = FrameConfig(), LabelConfig()
    out = Path(args.out)
    for sub in ("ball", "no_ball"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rows = []
    for path, session, kind in find_recordings(args.root):
        ov = autolabel.take_overrides(path.parent.parent, path, lab)
        name = f"{session}__{path.stem}.h5"
        if ov.get("drop"):
            rows.append({"file": name, "session": session, "kind": kind, "status": "dropped"})
            continue
        if kind == NONE_KIND:
            shutil.copyfile(path, out / "no_ball" / name)
            rows.append({"file": f"no_ball/{name}", "session": session, "kind": kind, "status": "no labels"})
        elif kind in args.targets:
            r = export_take(path, kind, out / "ball" / name, cfg, lab)
            rows.append({"file": f"ball/{name}", "session": session, "kind": kind, "status": "labelled", **r})
        else:
            rows.append({"file": name, "session": session, "kind": kind, "status": "skipped"})
        print(rows[-1], flush=True)
    keys = ["file", "session", "kind", "status", "frames", "seconds", "of_frames", "threshold"]
    with open(out / "labels.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)
    total = sum(r.get("seconds", 0) for r in rows)
    print(f"{sum(r['status'] == 'labelled' for r in rows)} labelled takes, {total:.1f} s of ball positions -> {out}")


if __name__ == "__main__":
    main()
