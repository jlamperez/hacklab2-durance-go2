#!/usr/bin/env python3
"""Send one synthetic frame to the SPLEAT server and wait for the reply, on the Kria.

To find out what makes the FPGA hang: an empty frame, one active pixel, or a dense frame.
The FPGA returns its output as events (one per output spike, decoded by backward-server's
LegacyDecoder until an "end of inference" event), so the amount the DMA has to bring back
grows with the network's activity. If an empty frame comes back and a dense one does not,
the output volume is the problem.

Needs spleat_merge_proxy.py in the same folder. No camera, no despot:

    python3 spleat_poke.py --pattern zeros            # server on 3334, as with the proxy
    python3 spleat_poke.py --pattern pixel
    python3 spleat_poke.py --pattern dense --fraction 0.3

Reload the bitstream between tests if one hangs:
    systemctl restart spleat-xmutil-loadapp spleat-server
"""

import argparse
import socket
import struct
import sys
import time

import numpy as np

from spleat_merge_proxy import cbor_item, encode_frame, read_message


def make_frame(pattern, size, fraction, seed, channels=1):
    frame = np.zeros((size, size), "<f4")
    if pattern == "pixel":
        frame[size // 2, size // 2] = 1.0
    elif pattern == "blob":  # a ball-sized disc in the middle
        yy, xx = np.mgrid[:size, :size]
        frame[(yy - size // 2) ** 2 + (xx - size // 2) ** 2 <= 12 ** 2] = 1.0
    elif pattern == "dense":
        frame[np.random.default_rng(seed).random((size, size)) < fraction] = 1.0
    elif pattern == "ones":
        frame[:] = 1.0
    return np.repeat(frame[None], channels, axis=0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=3334)
    ap.add_argument("--pattern", choices=("zeros", "pixel", "blob", "dense", "ones"), default="zeros")
    ap.add_argument("--fraction", type=float, default=0.05, help="active pixels for --pattern dense")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--channels", type=int, default=1,
                    help="input channels (Durance's example: --channels 2 --size 80 --port 3333)")
    ap.add_argument("--count", type=int, default=1, help="frames to send, one at a time")
    ap.add_argument("--timeout", type=float, default=10.0, help="seconds to wait for each reply")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    frame = make_frame(args.pattern, args.size, args.fraction, args.seed, args.channels)
    print(f"pattern {args.pattern}: {int(np.count_nonzero(frame))} active pixels of {frame.size}")
    sock = socket.create_connection((args.host, args.port), timeout=args.timeout)
    for k in range(args.count):
        t0 = time.monotonic()
        sock.sendall(encode_frame(k, frame.tobytes()))
        try:
            body = read_message(sock)
        except socket.timeout:
            print(f"frame {k}: NO REPLY within {args.timeout:.0f} s (check: journalctl -u spleat-server -n 5)")
            return 1
        dt = time.monotonic() - t0
        msg, _ = cbor_item(body, 0)
        line = f"frame {k}: reply in {dt * 1e3:.0f} ms, latency {msg.get('latency')}"
        if msg.get("error"):
            line += f", error {msg['error']!r}"
        pay = msg.get("payload")
        if isinstance(pay, bytes) and pay:
            v = np.frombuffer(pay, "<f4")
            line += (f", {v.size} values, max {v.max():.3g}, sum {v.sum():.4g}, "
                     f"nonzero {int(np.count_nonzero(v))}")
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
