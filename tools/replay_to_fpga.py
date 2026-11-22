"""Replay labelled recordings into the FPGA, framed like despot, and score the decisions.

Takes capture/ HDF5 takes (raw GenX320 events), bins them like despot_send_and_forget with
`frequency: 100`, `downsampling: 4`: per 10 ms, the SUM of events per 4x4 block and polarity,
2x80x80 float32, channel = the recording's polarity. Sends each frame to spleat-server (the
server multiplies by its `scaling`; 256 for models/zones3), waits for the reply, and applies
the same readout as fpga_perception (zones_readout: spikes over the last 20 frames).

It answers: given input framed like training, does the network on the board pick the right
zone? Independent of despot's image orientation and channel order, which it does not use.

The server keeps neuron state across frames and connections, so `--gap` empty frames are sent
before each take. Run with nothing else feeding the server (stop the pipeline first).

    pixi run --manifest-path tools/river/pixi.toml python tools/replay_to_fpga.py \
        data/river/val/ball/*.h5 data/river/val/none/sat_cross1_*.h5 --host 192.168.77.1 --port 3334
"""
from __future__ import annotations

import argparse
import socket
import struct
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reflex.river.data import FrameConfig, frame_counts, read_events  # noqa: E402
from reflex.runtime.spleat_client import cbor_encode, parse_message, read_message  # noqa: E402
from reflex.runtime.zones_readout import ZonesReadout  # noqa: E402

LAYOUT = "chw"
SCALE = 1.0          # multiplied into every frame (1/16: the mean, for a server at scaling 4096)
EXPECTED = {"from-left": "left", "to-left": "left", "from-right": "right", "to-right": "right",
            "from-center": "center", "no-ball": "none", "none": "none"}


def take_frames(path: Path, cfg: FrameConfig, before_s: float) -> tuple[np.ndarray, str]:
    import h5py
    with h5py.File(path, "r") as f:
        label = str(f.attrs.get("label", "?"))
        t0, t1 = int(f.attrs["t_start_us"]), int(f.attrs["t_end_us"])
    ev = read_events(path)
    start = t0 - int(before_s * 1e6)
    n = max(1, (t1 - start) // cfg.bin_us)
    return frame_counts(ev, cfg, t0=start, n_frames=n).astype(np.float32) * SCALE, label


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("takes", nargs="+", type=Path)
    ap.add_argument("--host", default="192.168.77.1")
    ap.add_argument("--port", type=int, default=3334)
    ap.add_argument("--model-dir", type=Path, default=ROOT / "models/zones3")
    ap.add_argument("--before", type=float, default=0.0, help="seconds before the labelled start to include")
    ap.add_argument("--gap", type=int, default=100, help="empty frames before each take (settles the state)")
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--layout", choices=("chw", "swap", "hwc"), default="chw",
                    help="payload layout: chw as built (despot's order assumed), swap the two channels, "
                         "or hwc (channels interleaved per pixel)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply every frame (1/16 = 0.0625: send the mean, for a server at scaling 4096)")
    args = ap.parse_args()
    global SCALE, LAYOUT
    SCALE, LAYOUT = args.scale, args.layout

    cfg = FrameConfig(sensor=320, bin_us=10_000, downsample=4, norm="sum")
    readout = ZonesReadout.from_model_dir(args.model_dir, args.margin)
    empty = np.zeros((2, cfg.size, cfg.size), np.float32)
    msg_id = 0
    rows = []
    with socket.create_connection((args.host, args.port), timeout=10) as sock:
        sock.settimeout(10)

        def infer(frame: np.ndarray) -> np.ndarray:
            nonlocal msg_id
            if LAYOUT == "swap":
                frame = frame[::-1]
            elif LAYOUT == "hwc":
                frame = np.ascontiguousarray(frame.transpose(1, 2, 0))
            body = cbor_encode({"id": msg_id, "payload": frame.astype("<f4").tobytes()})
            sock.sendall(struct.pack(">I", len(body)) + body)
            msg_id += 1
            m = parse_message(read_message(sock), (len(readout.names), 1, 1))
            return m["values"].reshape(-1)

        for path in args.takes:
            frames, label = take_frames(path, cfg, args.before)
            for _ in range(args.gap):
                infer(empty)
            readout.reset()
            t = time.monotonic()
            decisions = Counter(readout.push(infer(f)).label for f in frames)
            dt = time.monotonic() - t
            want = EXPECTED.get(label, "?")
            zones = {k: v for k, v in decisions.items() if k != "none"}
            top = max(zones, key=zones.get) if zones else "none"
            ok = (top == want) if want != "none" else (sum(zones.values()) <= 0.1 * len(frames))
            rows.append((path.name, label, want, top, ok, decisions, len(frames)))
            share = " ".join(f"{k} {decisions.get(k, 0) / len(frames):4.0%}" for k in readout.names)
            print(f"{'OK ' if ok else 'BAD'} {path.name:42s} {label:12s} -> top zone {top:6s} | {share} "
                  f"| {len(frames)} frames, {len(frames) / dt:.0f}/s", flush=True)

    good = sum(r[4] for r in rows)
    print(f"\n{good}/{len(rows)} takes OK (ball takes: the most frequent zone decision is the take's "
          f"side; none takes: zones in at most 10 % of frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
