"""Read the FPGA's results from SPLEAT's server (backward-server) on the Kria.

    pixi run spleat-read                                  # 192.168.77.1:3333, models/ball_small
    pixi run spleat-read --host durance-kria.local --ball-dir models/ball_small --count 50
    pixi run spleat-read --save out/spleat_capture.bin   # also keep every message as received

It only reads: the server pushes one message per inference to every connected client
(Durance's gesture_node never sends anything). The camera client on the board
(despot_send_and_forget) is what feeds the server frames; without it, nothing arrives.

Protocol (docs/fpga-pipeline.md, from Durance's gesture_node/src/spleat_client.cpp and
qualia_sim.cpp): each message is a 4-byte big-endian length, then a CBOR map
{"id": int, "payload": bytes, "latency": int, "error": str}. "payload" holds the output
values as little-endian float32, as many as the server config's outputs
(channels x height x width).

Assumptions, not documented by Durance and checked by watching real messages:
- the payload is in channel, row, column order (the order of the config's keys);
- the values are the head's spike counts summed over the time steps, which is what the
  ball readout expects. If they are spikes of one step or potentials, the decoded P(ball)
  is wrong while the argmax position may still be right.

The CBOR decoder below covers what these messages use (integers, byte and text strings,
arrays, maps, floats, true/false/null), with no third-party package, so it also runs on
the board's Python.
"""
from __future__ import annotations

import argparse
import json
import socket
import struct
import time
from pathlib import Path

import numpy as np

MAX_MESSAGE = 16 * 1024 * 1024        # sanity limit; ours is 9x80x80 float32 = 230 kB


class ProtocolError(ValueError):
    pass


# ---------------------------------------------------------------- CBOR (RFC 8949), subset


def cbor_decode(data: bytes):
    value, end = _cbor_item(data, 0)
    if end != len(data):
        raise ProtocolError(f"{len(data) - end} trailing bytes after the CBOR item")
    return value


def _cbor_item(b: bytes, i: int):
    if i >= len(b):
        raise ProtocolError("truncated CBOR")
    major, info = b[i] >> 5, b[i] & 0x1F
    i += 1
    if major == 7:
        if info == 20:
            return False, i
        if info == 21:
            return True, i
        if info in (22, 23):
            return None, i
        if info == 25:
            return float(np.frombuffer(b[i:i + 2], ">f2")[0]), i + 2
        if info == 26:
            return struct.unpack(">f", b[i:i + 4])[0], i + 4
        if info == 27:
            return struct.unpack(">d", b[i:i + 8])[0], i + 8
        raise ProtocolError(f"unsupported CBOR simple value {info}")
    if info < 24:
        n = info
    elif info in (24, 25, 26, 27):
        size = 1 << (info - 24)
        if i + size > len(b):
            raise ProtocolError("truncated CBOR length")
        n = int.from_bytes(b[i:i + size], "big")
        i += size
    else:
        raise ProtocolError(f"unsupported CBOR length encoding {info} (indefinite lengths)")
    if major == 0:
        return n, i
    if major == 1:
        return -1 - n, i
    if major in (2, 3):
        if i + n > len(b):
            raise ProtocolError("truncated CBOR string")
        raw = bytes(b[i:i + n])
        return (raw if major == 2 else raw.decode("utf-8")), i + n
    if major == 4:
        out = []
        for _ in range(n):
            v, i = _cbor_item(b, i)
            out.append(v)
        return out, i
    if major == 5:
        out = {}
        for _ in range(n):
            k, i = _cbor_item(b, i)
            v, i = _cbor_item(b, i)
            out[k] = v
        return out, i
    raise ProtocolError(f"unsupported CBOR major type {major} (tags)")


def cbor_encode(value) -> bytes:
    """Encoder for the same subset; used by tests to play the server."""
    def head(major, n):
        if n < 24:
            return bytes([major << 5 | n])
        for info, size in ((24, 1), (25, 2), (26, 4), (27, 8)):
            if n < 1 << (8 * size):
                return bytes([major << 5 | info]) + n.to_bytes(size, "big")
        raise ValueError(n)
    if value is None:
        return b"\xf6"
    if isinstance(value, bool):
        return b"\xf5" if value else b"\xf4"
    if isinstance(value, int):
        return head(0, value) if value >= 0 else head(1, -1 - value)
    if isinstance(value, float):
        return b"\xfb" + struct.pack(">d", value)
    if isinstance(value, bytes):
        return head(2, len(value)) + value
    if isinstance(value, str):
        raw = value.encode()
        return head(3, len(raw)) + raw
    if isinstance(value, list):
        return head(4, len(value)) + b"".join(cbor_encode(v) for v in value)
    if isinstance(value, dict):
        return head(5, len(value)) + b"".join(cbor_encode(k) + cbor_encode(v) for k, v in value.items())
    raise TypeError(type(value))


# ---------------------------------------------------------------- messages


def read_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("server closed the connection")
        buf += chunk
    return bytes(buf)


def read_message(sock: socket.socket) -> bytes:
    """One raw message body (the CBOR map), without its 4-byte length."""
    (n,) = struct.unpack(">I", read_exact(sock, 4))
    if n > MAX_MESSAGE:
        raise ProtocolError(f"message length {n} is beyond the {MAX_MESSAGE} byte sanity limit")
    return read_exact(sock, n)


def parse_message(body: bytes, shape: tuple[int, int, int] | None) -> dict:
    """Decoded map; `values` is the payload as float32, reshaped to `shape` when it fits."""
    msg = cbor_decode(body)
    if not isinstance(msg, dict):
        raise ProtocolError(f"message is a CBOR {type(msg).__name__}, not a map")
    payload = msg.get("payload")
    if not isinstance(payload, bytes):
        raise ProtocolError(f"no byte-string 'payload' (keys: {sorted(map(str, msg))})")
    if len(payload) % 4:
        raise ProtocolError(f"payload of {len(payload)} bytes is not a whole number of float32")
    values = np.frombuffer(payload, "<f4")
    out = {k: v for k, v in msg.items() if k != "payload"}
    out["n_values"] = int(values.size)
    out["shape_ok"] = shape is not None and values.size == int(np.prod(shape))
    out["values"] = values.reshape(shape) if out["shape_ok"] else values
    return out


# ---------------------------------------------------------------- command line


def server_shape(bundle_config: Path | None) -> tuple[int, int, int] | None:
    """Output shape from a config.yaml written by spleat-bundle (outputs: channels/height/width)."""
    if bundle_config is None or not bundle_config.exists():
        return None
    section, vals = None, {}
    for line in bundle_config.read_text().splitlines():
        if line and not line.startswith(" "):
            section = line.rstrip(":").strip()
        elif section == "outputs" and ":" in line:
            k, v = (s.strip() for s in line.split(":", 1))
            vals[k] = int(float(v))
    return (vals["channels"], vals["height"], vals["width"]) if len(vals) == 3 else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="192.168.77.1")
    ap.add_argument("--port", type=int, default=3333)
    ap.add_argument("--ball-dir", type=Path, default=Path("models/ball_small"),
                    help="export with <prefix>_readout.json, to decode (x, y, P(ball)); '' to skip")
    ap.add_argument("--config", type=Path, default=Path("out/spleat/ball_small_model/config.yaml"),
                    help="the config.yaml deployed to the board, for the expected output shape")
    ap.add_argument("--count", type=int, default=0, help="stop after N messages (0: until Ctrl+C)")
    ap.add_argument("--save", type=Path, help="append every raw message (4-byte length + body) to this file")
    args = ap.parse_args(argv)

    shape = server_shape(args.config)
    readout, found = None, []
    if str(args.ball_dir) and args.ball_dir.is_dir():
        found = sorted(args.ball_dir.glob("*_readout.json"))
        readout = json.loads(found[0].read_text()) if len(found) == 1 else None
    print(f"expecting {shape and 'x'.join(map(str, shape)) or 'any'} float32 per message; "
          f"decoding with {found[0] if readout else 'nothing (no readout)'}", flush=True)
    save = args.save.open("ab") if args.save else None

    from reflex.runtime.detectors import decode_ball_output
    with socket.create_connection((args.host, args.port), timeout=10) as sock:
        sock.settimeout(None)
        print(f"connected to {args.host}:{args.port}; waiting for messages (the camera client must run)", flush=True)
        n, t0, t_prev = 0, time.monotonic(), None
        try:
            while not args.count or n < args.count:
                body = read_message(sock)
                now = time.monotonic()
                if save:
                    save.write(struct.pack(">I", len(body)) + body)
                m = parse_message(body, shape)
                v = m["values"]
                line = (f"#{n:<5d} id {m.get('id')!s:>6} {len(body):7d} B  {m['n_values']} values"
                        f"{'' if m['shape_ok'] else ' (UNEXPECTED COUNT)'}  latency {m.get('latency')}  "
                        f"min {v.min():.3g} max {v.max():.3g} sum {v.sum():.4g} nonzero {int((v != 0).sum())}")
                if m.get("error"):
                    line += f"  error {m['error']!r}"
                if readout and m["shape_ok"]:
                    x, y, p = decode_ball_output(v, readout)
                    line += f"  -> x {x:.0f} y {y:.0f} P(ball) {p:.2f}"
                if t_prev is not None:
                    line += f"  dt {1e3 * (now - t_prev):.0f} ms"
                print(line, flush=True)
                n, t_prev = n + 1, now
        except KeyboardInterrupt:
            pass
        finally:
            if save:
                save.close()
        if n > 1:
            print(f"{n} messages in {time.monotonic() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
