"""Look at the auto-labels of recorded sessions before training on them.

    python -m reflex.river.review data/river/sat_pm               # -> out/labels/sat_pm.png
    python -m reflex.river.review data/river/* --kinds ball       # only the target takes
    python -m reflex.river.review data/river_synth/* --auto       # auto-labels vs .gt.npz

One row per take. The grey image is where the events are along the image's horizontal
axis (dog's left at the top) over time; the red line is the tracked target, the coloured
band under it is the label per frame. With `--auto`, takes that have ground truth
(synthetic data) are labelled by the tracker anyway, the true position is drawn in green,
and the agreement is printed. Fix a bad take in `<session>/overrides.json` (see
reflex/river/autolabel.py) and run this again.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .autolabel import session_calibration
from .data import (NONE_KIND, FrameConfig, LabelConfig, bin_labels, find_recordings, gt_track,
                   load_recording, read_events)

COLORS = ["#d9d9d9", "#4c78a8", "#54a24b", "#f58518", "#b279a2", "#e45756", "#72b7b2", "#ff9da6"]
PAGE = 24                       # takes per image


def gt_labels(rec, cfg, lab) -> np.ndarray | None:
    g = rec.path.with_suffix(".gt.npz")
    if not g.exists():
        return None
    gt = np.load(g)
    ev = read_events(rec.path)
    t0, n = int(ev["t"][0]), rec.trim[1] if rec.trim[1] else len(rec.labels)
    a, b = rec.trim
    xn = gt_track({k: gt[k] for k in gt.files}, cfg, t0, n)
    return bin_labels(xn, cfg, lab)[a:b], xn[a:b]


def agreement(pred: np.ndarray, truth: np.ndarray, n_classes: int) -> np.ndarray:
    conf = np.zeros((n_classes, n_classes), np.int64)
    np.add.at(conf, (truth, pred), 1)
    return conf


def summarize(conf: np.ndarray, names) -> str:
    n = conf.sum()
    acc = np.trace(conf) / max(1, n)
    fp = conf[0, 1:].sum() / max(1, conf[0].sum())             # truth none -> a zone
    miss = conf[1:, 0].sum() / max(1, conf[1:].sum())          # truth zone -> none
    wrong = (conf[1:, 1:].sum() - np.trace(conf[1:, 1:])) / max(1, conf[1:, 1:].sum())
    return (f"frames {n}  agree {acc:.1%}  false target {fp:.1%}  missed {miss:.1%}  "
            f"wrong zone {wrong:.1%}")


def _plot(recs, cfg, lab, out: Path, title: str, with_gt: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(COLORS[:lab.n_classes])
    fig, axes = plt.subplots(2 * len(recs), 1, figsize=(15, 1.45 * len(recs) + 0.6), squeeze=False,
                             gridspec_kw={"height_ratios": [6, 1] * len(recs), "hspace": 0.45})
    S = cfg.size
    edges = [lab.margin + (1 - 2 * lab.margin) * k / lab.n_bins for k in range(1, lab.n_bins)]
    for i, rec in enumerate(recs):
        ax, band = axes[2 * i, 0], axes[2 * i + 1, 0]
        n = len(rec.counts)
        dur = n * cfg.bin_us * 1e-6
        kymo = rec.counts.sum(axis=(1, 2), dtype=np.float32)       # (T, S): x in the dog frame
        ax.imshow(np.sqrt(kymo.T), aspect="auto", cmap="gray_r", interpolation="nearest",
                  extent=[0, dur, S, 0], vmax=max(1.0, np.sqrt(np.percentile(kymo, 99.5))))
        t = (np.arange(n) + 0.5) * cfg.bin_us * 1e-6
        if rec.track is not None:
            x = rec.track.x[rec.trim[0]:rec.trim[1]]
            x = 1 - x if cfg.flip_x else x
            ax.plot(t, x * S, color="#e45756", lw=1.2)
        if with_gt:
            g = gt_labels(rec, cfg, lab)
            if g is not None:
                xg = 1 - g[1] if cfg.flip_x else g[1]
                ax.plot(t, xg * S, color="#2ca02c", lw=0.9, ls="--")
        for e in edges:
            ax.axhline(e * S, color="#4c78a8", lw=0.5, ls=":")
        hist = np.bincount(rec.labels, minlength=lab.n_classes) / max(1, n)
        extra = f"  overrides {rec.overrides}" if rec.overrides else ""
        src = rec.source + (", ego-motion" if getattr(rec.track, "camera", None) is not None else "")
        ax.set_title(f"{rec.session}/{rec.kind}/{rec.path.name}  {dur:.1f} s  [{src}]  "
                     + "  ".join(f"{nm} {h:.0%}" for nm, h in zip(lab.names, hist)) + extra,
                     fontsize=8, loc="left")
        ax.set_yticks([0, S])
        ax.set_yticklabels(["left", "right"], fontsize=7)
        ax.set_xticks([])
        band.imshow(rec.labels[None], aspect="auto", cmap=cmap, vmin=0, vmax=lab.n_classes - 1,
                    extent=[0, dur, 0, 1], interpolation="nearest")
        band.set_yticks([])
        band.tick_params(labelsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in range(lab.n_classes)]
    fig.legend(handles, lab.names, loc="upper right", ncol=lab.n_classes, fontsize=8)
    fig.suptitle(title, fontsize=9, x=0.01, ha="left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Review auto-labels: one image per session")
    ap.add_argument("sessions", nargs="+", help="session folders, e.g. data/river/sat_pm")
    ap.add_argument("--out", default="out/labels")
    ap.add_argument("--kinds", nargs="+", help="only these kind folders (default: all)")
    ap.add_argument("--auto", action="store_true", help="ignore .gt.npz and compare the auto-labels with it")
    ap.add_argument("--method", default="blob", choices=["blob", "median"])
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--n-bins", type=int, default=3)
    ap.add_argument("--flip-x", action="store_true")
    ap.add_argument("--ego-kinds", nargs="*", default=list(LabelConfig.ego_kinds),
                    help="kinds tracked relative to the scene's motion (moving camera)")
    tunable = []
    for k, v in vars(LabelConfig()).items():         # every tracker setting can be tried here
        if k not in ("n_bins", "method") and isinstance(v, (int, float)):
            ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
            tunable.append(k)
    a = ap.parse_args()
    cfg = FrameConfig(flip_x=a.flip_x)
    lab = LabelConfig(n_bins=a.n_bins, method=a.method, ego_kinds=tuple(a.ego_kinds),
                      **{k: getattr(a, k) for k in tunable})
    total = np.zeros((lab.n_classes, lab.n_classes), np.int64)
    for sdir in map(Path, a.sessions):
        if not sdir.is_dir():
            continue
        if lab.method == "blob":
            print(f"{sdir}: {session_calibration(sdir, cfg, lab).summary()}")
        recs, conf = [], np.zeros_like(total)
        for path, session, kind in find_recordings(sdir.parent):
            if path.parent.parent != sdir or (a.kinds and kind not in a.kinds):
                continue
            targets = () if kind == NONE_KIND else (kind,)
            rec = load_recording(path, session, kind, cfg, lab, targets, use_gt=not a.auto)
            if rec is None:
                print(f"  {kind}/{path.name}: empty or dropped")
                continue
            recs.append(rec)
            hist = np.bincount(rec.labels, minlength=lab.n_classes) / max(1, len(rec.labels))
            line = f"  {kind}/{path.name}  [{rec.source}]  " + "  ".join(
                f"{nm}={h:.0%}" for nm, h in zip(lab.names, hist))
            g = gt_labels(rec, cfg, lab) if a.auto and kind != NONE_KIND else None
            if g is not None:
                c = agreement(rec.labels, g[0], lab.n_classes)
                conf += c
                line += f"  | vs truth: agree {np.trace(c) / max(1, c.sum()):.0%}"
            print(line)
        if conf.sum():
            print(f"  {sdir.name}: {summarize(conf, lab.names)}")
            total += conf
        if recs and not a.no_plot:
            for p0 in range(0, len(recs), PAGE):
                page = recs[p0:p0 + PAGE]
                suffix = f"_{p0 // PAGE + 1}" if len(recs) > PAGE else ""
                out = Path(a.out) / f"{sdir.name}{suffix}.png"
                _plot(page, cfg, lab, out, f"{sdir}  labels: {lab.method}", a.auto)
                print(f"  wrote {out}")
    if total.sum():
        print(f"ALL: {summarize(total, lab.names)}")


if __name__ == "__main__":
    main()
