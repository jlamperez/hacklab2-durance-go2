"""Look at what is inside the .npz files: dataset windows or raw event recordings.

    pixi run view-data data/synth                      # browse the training windows
    pixi run view-data data/synth --label left         # only one label
    pixi run view-data data/synth --kind crossing      # only one kind of episode
    pixi run view-data data/synth --sheet out/sheet.png   # save a grid instead of opening a window
    pixi run view-data out/demo_ep0.npz                # play a raw event recording
    pixi run view-data out/demo_ep0.npz --video out/ep0.mp4

Keys, dataset browser:  right/left next/previous, up/down jump 20, space play the time
steps, r random, q quit.
Keys, recording player: space pause, right/left step, q quit.

Colors: ON events (brighter) light blue, OFF events (darker) red.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reflex.events import io as ev_io
from reflex.events import repr as ev_repr


def _rgb(frame: np.ndarray, saturation: float = 1.5) -> np.ndarray:
    return ev_repr.render(frame.astype(np.float32), saturation)[..., ::-1]


# ----------------------------------------------------------------------------- datasets
def load_dataset(path: Path, label: str | None, kind: str | None):
    meta = json.loads((path / "meta.json").read_text())
    parts = [np.load(p) for p in sorted(path.glob("shard_*.npz"))]
    data = {k: np.concatenate([p[k] for p in parts]) for k in ("x", "y", "episode", "kind")}
    keep = np.ones(len(data["y"]), dtype=bool)
    if label:
        keep &= data["y"] == meta["labels"].index(label)
    if kind:
        keep &= data["kind"] == meta["kinds"].index(kind)
    idx = np.nonzero(keep)[0]
    if len(idx) == 0:
        raise SystemExit("no windows match that filter")
    return data, meta, idx


def _caption(data, meta, i: int) -> str:
    return (f"label: {meta['labels'][data['y'][i]].upper()}   kind: {meta['kinds'][data['kind'][i]]}   "
            f"episode {data['episode'][i]}   {int(data['x'][i].sum())} events")


def save_sheet(data, meta, idx, out: Path, n: int, seed: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    pick = rng.choice(idx, min(n, len(idx)), replace=False)
    cols = 6
    rows = int(np.ceil(len(pick) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.0), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), pick):
        ax.imshow(_rgb(data["x"][i].sum(0)), interpolation="nearest")
        ax.set_title(f"{meta['labels'][data['y'][i]]} / {meta['kinds'][data['kind'][i]]}", fontsize=9)
    fig.suptitle(f"{len(pick)} of {len(idx)} windows ({meta['window']['window_ms']:.0f} ms each)   {meta.get('source', '')}",
                 fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


def browse_dataset(data, meta, idx, duration: float) -> None:
    import matplotlib.pyplot as plt
    n_bins = data["x"].shape[1]
    fig = plt.figure(figsize=(12, 6.2))
    grid = fig.add_gridspec(3, n_bins // 2, height_ratios=[4, 1, 1])
    big = fig.add_subplot(grid[0, :])
    small = [fig.add_subplot(grid[1 + k // (n_bins // 2), k % (n_bins // 2)]) for k in range(n_bins)]
    state = {"pos": 0, "play": False, "bin": 0}
    rng = np.random.default_rng()

    def draw():
        i = idx[state["pos"]]
        x = data["x"][i]
        shown = x[state["bin"]] if state["play"] else x.sum(0)
        big.clear()
        big.imshow(_rgb(shown), interpolation="nearest")
        big.set_title(f"[{state['pos'] + 1}/{len(idx)}]   {_caption(data, meta, i)}"
                      + (f"   time step {state['bin'] + 1}/{n_bins}" if state["play"] else "   (all time steps)"))
        big.axis("off")
        for k, ax in enumerate(small):
            ax.clear()
            ax.imshow(_rgb(x[k]), interpolation="nearest")
            ax.set_title(f"step {k + 1}", fontsize=8, color="tab:red" if state["play"] and k == state["bin"] else "black")
            ax.axis("off")
        fig.canvas.draw_idle()

    def on_key(event):
        step = {"right": 1, "left": -1, "up": 20, "down": -20}.get(event.key)
        if step is not None:
            state["pos"] = (state["pos"] + step) % len(idx)
        elif event.key == "r":
            state["pos"] = int(rng.integers(len(idx)))
        elif event.key == " ":
            state["play"], state["bin"] = not state["play"], 0
        elif event.key == "q":
            plt.close(fig)
            return
        draw()

    def tick():
        if state["play"]:
            state["bin"] = (state["bin"] + 1) % n_bins
            draw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    timer = fig.canvas.new_timer(interval=150)
    timer.add_callback(tick)
    timer.start()
    if duration:
        closer = fig.canvas.new_timer(interval=int(duration * 1000))
        closer.add_callback(lambda: plt.close(fig))
        closer.start()
    fig.suptitle("right/left: next/previous   up/down: jump 20   space: play time steps   r: random   q: quit",
                 fontsize=9)
    draw()
    plt.show()


# --------------------------------------------------------------------------- recordings
def recording_frames(events, h, w, window_ms: float, step_ms: float):
    t0, t1 = int(events["t"][0]), int(events["t"][-1])
    for t_end in range(t0 + int(window_ms * 1000), t1, int(step_ms * 1000)):
        lo, hi = np.searchsorted(events["t"], [t_end - int(window_ms * 1000), t_end])
        yield t_end, hi - lo, _rgb(ev_repr.to_frame(events[lo:hi], h, w))


def describe_recording(events, meta) -> str:
    dur = (events["t"][-1] - events["t"][0]) * 1e-6
    extra = "   ".join(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}"
                       for k, v in meta.items() if k not in ("height", "width"))
    return (f"{len(events)} events in {dur:.2f} s ({len(events) / max(dur, 1e-9) / 1e3:.0f} k/s), "
            f"{meta['width']}x{meta['height']}, ON {100 * events['p'].mean():.0f} %   {extra}")


def save_video(events, meta, out: Path, window_ms: float, step_ms: float, fps: int) -> None:
    import imageio.v2 as imageio
    out.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out, fps=fps, codec="libx264", quality=8, macro_block_size=2) as writer:
        for _, _, img in recording_frames(events, meta["height"], meta["width"], window_ms, step_ms):
            writer.append_data(np.kron(img, np.ones((3, 3, 1), dtype=np.uint8)))
    print(f"wrote {out}")


def play_recording(events, meta, window_ms: float, step_ms: float, duration: float) -> None:
    import matplotlib.pyplot as plt
    frames = list(recording_frames(events, meta["height"], meta["width"], window_ms, step_ms))
    fig, ax = plt.subplots(figsize=(10, 6))
    state = {"i": 0, "play": True}
    t_start = int(events["t"][0])

    def draw():
        t_end, n, img = frames[state["i"]]
        ax.clear()
        ax.imshow(img, interpolation="nearest")
        ax.set_title(f"t = {(t_end - t_start) / 1e3:7.1f} ms   {n} events in the last {window_ms:.0f} ms"
                     f"   [{state['i'] + 1}/{len(frames)}]" + ("" if state["play"] else "   PAUSED"))
        ax.axis("off")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == " ":
            state["play"] = not state["play"]
        elif event.key in ("right", "left"):
            state["play"] = False
            state["i"] = (state["i"] + (1 if event.key == "right" else -1)) % len(frames)
        elif event.key == "q":
            plt.close(fig)
            return
        draw()

    def tick():
        if state["play"]:
            state["i"] = (state["i"] + 1) % len(frames)
            draw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    timer = fig.canvas.new_timer(interval=40)
    timer.add_callback(tick)
    timer.start()
    if duration:
        closer = fig.canvas.new_timer(interval=int(duration * 1000))
        closer.add_callback(lambda: plt.close(fig))
        closer.start()
    fig.suptitle(describe_recording(events, meta) + "\nspace: pause   right/left: step   q: quit", fontsize=9)
    draw()
    plt.show()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="dataset folder (with meta.json and shards) or one event recording .npz")
    ap.add_argument("--label", help="dataset: only this label (none/left/center/right)")
    ap.add_argument("--kind", help="dataset: only this kind of episode")
    ap.add_argument("--sheet", help="dataset: save a grid of random windows to this image and exit")
    ap.add_argument("--n", type=int, default=48, help="windows in the sheet")
    ap.add_argument("--video", help="recording: save an .mp4 and exit")
    ap.add_argument("--window-ms", type=float, default=10.0, help="recording: events shown per frame")
    ap.add_argument("--step-ms", type=float, default=4.0, help="recording: time between frames")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--duration", type=float, default=0.0, help="close the window after this many seconds")
    args = ap.parse_args()

    path = Path(args.path)
    if path.is_dir():
        data, meta, idx = load_dataset(path, args.label, args.kind)
        counts = np.bincount(data["y"][idx], minlength=len(meta["labels"]))
        print(f"{len(idx)} windows of shape {data['x'].shape[1:]}: "
              + ", ".join(f"{lab} {c}" for lab, c in zip(meta["labels"], counts)))
        if args.sheet:
            save_sheet(data, meta, idx, Path(args.sheet), args.n, args.seed)
        else:
            browse_dataset(data, meta, idx, args.duration)
    else:
        events, meta = ev_io.load_npz(path)
        print(describe_recording(events, meta))
        if args.video:
            save_video(events, meta, Path(args.video), args.window_ms, args.step_ms, fps=50)
        else:
            play_recording(events, meta, args.window_ms, args.step_ms, args.duration)


if __name__ == "__main__":
    main()
