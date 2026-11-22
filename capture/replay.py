"""Replay recorded event-camera takes: a session folder, a folder of .h5 / .bin files, or files.

Shows the events of the take as a moving window, with its label and instruction, and a
timeline of the event rate with the marks from the guided capture: the labelled stretch
between the start and end mark, the extra M marks, and the play position. Click the
timeline to jump.

Reads the take files of capture.record (.h5, needs h5py) and river-tools' .bin samples
(numpy and matplotlib only).

    python -m capture.replay data/river_samples/sat_follow1
    python -m capture.replay data/river_samples/samples            # unlabelled river-tools samples
    python -m capture.replay take_003_left.bin --snapshot out.png   # save one frame and exit

Keys:
    SPACE           pause / play
    LEFT / RIGHT    back / forward 0.5 s (one frame when paused)
    UP / DOWN, P/N  previous / next take
    HOME            restart the take
    + / -           longer / shorter window of events per frame
    [ / ]           slower / faster
    L               loop the take on / off (off: go on to the next take)
    G               guide lines for the image thirds on / off
    Q               quit
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from reflex.events.h5 import read_bin, read_h5

EVENT_DTYPE = np.dtype([("x", ">u2"), ("y", ">u2"), ("polarity", "u1"), ("timestamp", ">i8")])
CAMERA_SIZE = 320
COLORS = np.array([[255, 80, 40], [40, 160, 255]], dtype=np.uint8)
LABEL_COLORS = {"none": "#91a0b3", "left": "#ffd166", "center": "#6ce3c5", "right": "#c77dff",
                "close": "#ff4b4b", "drone": "#4cc9f0"}
RATE_BIN_US = 10_000


@dataclass
class TakeData:
    path: Path
    t: np.ndarray            # µs since the first event, int64
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def duration_us(self) -> int:
        return int(self.t[-1]) if len(self.t) else 0

    def rel(self, key: str) -> float | None:
        """A mark from the metadata in µs since the first event, or None."""
        v = self.meta.get(key)
        return None if v in (None, "") else float(v) - self.meta["_t0"]


def load_take(path: Path) -> TakeData:
    meta: dict = {}
    if path.suffix == ".h5":
        ts, x, y, p, meta = read_h5(path)
    else:
        ev = read_bin(path)
        ts, x, y, p = ev["timestamp"], ev["x"], ev["y"], ev["polarity"]
        side = path.with_suffix(".json")          # older guided-capture takes
        if side.exists():
            meta = json.loads(side.read_text(encoding="utf-8"))
    # Durance's gesture dataset: the class is in the name, e.g. gesture_03-hello-fast-19.bin
    m = re.match(r"gesture_(\d+)-(.+)-(fast|large|normal)-(\d+)$", path.stem)
    if m and "label" not in meta:
        meta.update(label=m.group(2), instruction=f"Durance gesture dataset, class {int(m.group(1))}, "
                                                  f"variant {m.group(3)}, session {path.parent.name}")
    ok = (x < CAMERA_SIZE) & (y < CAMERA_SIZE) & (p < 2)
    ts = np.asarray(ts, dtype=np.int64)[ok]
    t0 = int(ts[0]) if len(ts) else 0
    meta["_t0"] = t0
    return TakeData(path, ts - t0, np.asarray(x)[ok].astype(np.intp), np.asarray(y)[ok].astype(np.intp),
                    np.asarray(p)[ok].astype(np.intp), meta)


def find_takes(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            index = p / "takes.csv"
            if index.exists():
                with index.open(newline="", encoding="utf-8") as f:
                    rows = sorted(csv.DictReader(f), key=lambda r: int(r["take"]))
                files += [p / r["file"] for r in rows if (p / r["file"]).exists()]
            else:
                h5 = sorted(p.glob("*.h5"))
                # A .bin that was converted is shown once, from its .h5.
                files += sorted([*h5, *(b for b in p.glob("*.bin") if not b.with_suffix(".h5").exists())])
        elif p.suffix in (".h5", ".bin") and p.exists():
            files.append(p)
    return files


def render(take: TakeData, t_us: float, window_us: float) -> np.ndarray:
    """Image of the events in (t - window, t]."""
    img = np.zeros((CAMERA_SIZE, CAMERA_SIZE, 3), dtype=np.uint8)
    lo, hi = np.searchsorted(take.t, [t_us - window_us, t_us], side="right")
    img[take.y[lo:hi], take.x[lo:hi]] = COLORS[take.p[lo:hi]]
    return img


def event_rate(take: TakeData) -> tuple[np.ndarray, np.ndarray]:
    """Events per second in RATE_BIN_US bins: (bin centers in s, rate)."""
    n = max(1, take.duration_us // RATE_BIN_US + 1)
    counts = np.bincount(take.t // RATE_BIN_US, minlength=n)
    centers = (np.arange(len(counts)) + 0.5) * RATE_BIN_US / 1e6
    return centers, counts * (1e6 / RATE_BIN_US)


class Viewer:
    def __init__(self, files: list[Path], window_ms: float, speed: float, loop: bool):
        import matplotlib.pyplot as plt

        self.plt = plt
        for k in list(plt.rcParams):      # free the keys matplotlib uses for its own shortcuts
            if k.startswith("keymap."):
                plt.rcParams[k] = []
        self.files, self.index = files, 0
        self.window_us, self.speed, self.loop = window_ms * 1000, speed, loop
        self.paused, self.guides = False, False
        self.pos_us = 0.0

        bg, muted, text = "#0c121c", "#91a0b3", "#edf2f7"
        self.fig = fig = plt.figure(figsize=(9, 10), facecolor=bg)
        fig.canvas.manager.set_window_title("Replay takes")
        self.ax = ax = fig.add_axes([0.08, 0.30, 0.84, 0.56], facecolor="black")
        ax.set_xticks([]), ax.set_yticks([])
        self.view = ax.imshow(np.zeros((CAMERA_SIZE, CAMERA_SIZE, 3), np.uint8), interpolation="nearest")
        self.guide_lines = [ax.axvline(CAMERA_SIZE * f, color="#ffffff", alpha=0.25, lw=1, visible=False)
                            for f in (1 / 3, 2 / 3)]
        self.head = fig.text(0.08, 0.955, "", color=muted, fontsize=10)
        self.title = fig.text(0.08, 0.91, "", color=text, fontsize=22, weight="bold")
        self.instr = fig.text(0.08, 0.875, "", color=muted, fontsize=9)
        self.state = fig.text(0.08, 0.255, "", color=muted, fontsize=10)
        self.tl = fig.add_axes([0.08, 0.10, 0.84, 0.13], facecolor="#111a27")
        self.tl.tick_params(colors=muted, labelsize=8)
        for s in self.tl.spines.values():
            s.set_color("#293547")
        fig.text(0.08, 0.035, "SPACE play/pause  ←/→ seek  ↑/↓ take  +/- window  [/] speed  "
                              "L loop  G guides  click timeline to jump  Q quit", color=muted, fontsize=8)
        fig.canvas.mpl_connect("key_press_event", self.on_key)
        fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.open(0)

    # -- take handling
    def open(self, index: int) -> None:
        self.index = index % len(self.files)
        self.take = load_take(self.files[self.index])
        self.pos_us = 0.0
        tl = self.tl
        tl.clear()
        tl.set_facecolor("#111a27")
        x, rate = event_rate(self.take)
        tl.fill_between(x, rate, color="#3a86ff", alpha=0.6, lw=0)
        tl.set_xlim(0, max(self.take.duration_us / 1e6, 0.01))
        # Cap at the 99th percentile so one burst does not flatten the rest of the curve.
        tl.set_ylim(0, max(float(np.percentile(rate, 99)) * 1.3, 1))
        tl.set_xlabel("s", color="#91a0b3", fontsize=8)
        tl.set_yticks([])
        m = self.take.meta
        label = m.get("label", "")
        t_start, t_end = self.take.rel("t_start_us"), self.take.rel("t_end_us")
        if t_start is not None and t_end is not None:
            tl.axvspan(t_start / 1e6, t_end / 1e6, color=LABEL_COLORS.get(label, "#ffffff"), alpha=0.25, lw=0)
            tl.axvline(t_start / 1e6, color="#6ce3c5", lw=1.5)
            tl.axvline(t_end / 1e6, color="#ff4b4b", lw=1.5)
        for mk in m.get("marks_us", []) or []:
            tl.axvline((mk - m["_t0"]) / 1e6, color="#ffd166", lw=1, ls="--")
        self.playhead = tl.axvline(0, color="#ffffff", lw=1.2)

        name = self.take.path.name
        if label:
            extra = "   ".join(s for s in (f"phase {m['phase']}" if m.get("phase") not in (None, "") else "", f"{m['distance_m']} m" if m.get("distance_m") else "",
                                           f"camera: {m['camera']}" if m.get("camera") else "", f"ball: {m['ball']}" if m.get("ball") else "") if s)
            self.head.set_text(f"{self.index + 1}/{len(self.files)}   {name}   {extra}")
            self.title.set_text(label.upper())
            self.title.set_color(LABEL_COLORS.get(label, "#edf2f7"))
            self.instr.set_text(m.get("instruction", ""))
        else:
            self.head.set_text(f"{self.index + 1}/{len(self.files)}   {name}")
            self.title.set_text("(no label)")
            self.title.set_color("#91a0b3")
            self.instr.set_text("")

    # -- input
    def on_key(self, event) -> None:
        k = (event.key or "").lower()
        step = (1000 / 50) * 1000 if self.paused else 500_000      # one 20 ms frame, or 0.5 s
        if k == "q":
            self.plt.close(self.fig)
        elif k == " ":
            self.paused = not self.paused
        elif k == "right":
            self.pos_us = min(self.pos_us + step, self.take.duration_us)
        elif k == "left":
            self.pos_us = max(self.pos_us - step, 0)
        elif k in ("down", "n"):
            self.open(self.index + 1)
        elif k in ("up", "p"):
            self.open(self.index - 1)
        elif k == "home":
            self.pos_us = 0.0
        elif k in ("+", "="):
            self.window_us = min(self.window_us * 1.5, 500_000)
        elif k == "-":
            self.window_us = max(self.window_us / 1.5, 1_000)
        elif k == "]":
            self.speed = min(self.speed * 2, 8)
        elif k == "[":
            self.speed = max(self.speed / 2, 1 / 16)
        elif k == "l":
            self.loop = not self.loop
        elif k == "g":
            self.guides = not self.guides
            for line in self.guide_lines:
                line.set_visible(self.guides)

    def on_click(self, event) -> None:
        if event.inaxes is self.tl and event.xdata is not None:
            self.pos_us = float(np.clip(event.xdata * 1e6, 0, self.take.duration_us))

    # -- drawing
    def draw(self) -> None:
        self.view.set_data(render(self.take, self.pos_us, self.window_us))
        self.playhead.set_xdata([self.pos_us / 1e6])
        self.state.set_text(f"{self.pos_us / 1e6:6.2f} / {self.take.duration_us / 1e6:.2f} s   "
                            f"window {self.window_us / 1000:.0f} ms   speed {self.speed:g}x   "
                            f"{'paused' if self.paused else 'playing'}   loop {'on' if self.loop else 'off'}")

    def run(self) -> None:
        plt = self.plt
        plt.show(block=False)
        last = time.monotonic()
        while plt.fignum_exists(self.fig.number):
            now = time.monotonic()
            if not self.paused:
                self.pos_us += (now - last) * 1e6 * self.speed
                if self.pos_us > self.take.duration_us:
                    if self.loop or len(self.files) == 1:
                        self.pos_us = 0.0
                    else:
                        self.open(self.index + 1)
            last = now
            self.draw()
            self.fig.canvas.draw_idle()
            plt.pause(0.02)

    def snapshot(self, out: Path, at_s: float | None) -> None:
        m = self.take
        if at_s is None:   # middle of the labelled stretch, else the busiest moment
            t_start, t_end = m.rel("t_start_us"), m.rel("t_end_us")
            if t_start is not None and t_end is not None:
                at_us = (t_start + t_end) / 2
            else:
                x, rate = event_rate(m)
                at_us = float(x[int(rate.argmax())] * 1e6)
        else:
            at_us = at_s * 1e6
        self.pos_us = float(np.clip(at_us, 0, m.duration_us))
        self.draw()
        self.fig.savefig(out, facecolor=self.fig.get_facecolor())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="session folder(s), folder(s) of .h5 / .bin files, or files")
    ap.add_argument("--window-ms", type=float, default=20.0, help="events shown per frame")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed")
    ap.add_argument("--loop", action="store_true", help="repeat each take instead of going on to the next")
    ap.add_argument("--snapshot", help="save one frame of the first take to this image and exit")
    ap.add_argument("--at", type=float, default=None, help="with --snapshot: time in s (default: middle of the take's action)")
    args = ap.parse_args()

    files = find_takes(args.paths)
    if not files:
        raise SystemExit("No .bin takes found in " + ", ".join(args.paths))
    if args.snapshot:
        import matplotlib
        matplotlib.use("Agg")
    viewer = Viewer(files, args.window_ms, args.speed, args.loop)
    if args.snapshot:
        viewer.snapshot(Path(args.snapshot), args.at)
        print(f"Saved {args.snapshot}")
        return
    print(f"{len(files)} takes")
    viewer.run()


if __name__ == "__main__":
    main()
