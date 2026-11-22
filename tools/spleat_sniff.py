#!/usr/bin/env python3
"""Capture what a camera client (despot_send_and_forget) sends to the SPLEAT server.

Runs on the Kria, standard library only (the board has no pip). Stop the real server first,
so this script can take its port:

    systemctl stop spleat-server
    python3 spleat_sniff.py                      # listens on 3333, saves /root/spleat_in.bin
    despot_send_and_forget --to_display 0        # second shell; Ctrl+C it once this ends
    systemctl start spleat-server

It accepts one client, records the raw byte stream for a few seconds, then tries to split it
into messages (4-byte big-endian length + CBOR, the format of the server's output messages)
and prints each message's keys, and for byte strings the size and what it would hold as
float32, uint16 or uint8 values. The raw file can be copied to the laptop and decoded again:

    python3 tools/spleat_sniff.py --decode spleat_in.bin
"""

import argparse
import socket
import struct
import sys
import time


def cbor_item(b, i):
    major, info = b[i] >> 5, b[i] & 0x1F
    i += 1
    if major == 7:
        simple = {20: False, 21: True, 22: None, 23: None}
        if info in simple:
            return simple[info], i
        if info == 25:
            return struct.unpack(">e", b[i:i + 2])[0], i + 2
        if info == 26:
            return struct.unpack(">f", b[i:i + 4])[0], i + 4
        if info == 27:
            return struct.unpack(">d", b[i:i + 8])[0], i + 8
        raise ValueError(f"unsupported simple value {info}")
    if info < 24:
        n = info
    elif info <= 27:
        size = 1 << (info - 24)
        n = int.from_bytes(b[i:i + size], "big")
        i += size
    else:
        raise ValueError(f"indefinite length (major {major}) not supported")
    if major == 0:
        return n, i
    if major == 1:
        return -1 - n, i
    if major in (2, 3):
        raw = bytes(b[i:i + n])
        if len(raw) < n:
            raise ValueError("truncated string")
        return (raw if major == 2 else raw.decode("utf-8", "replace")), i + n
    if major == 4:
        out = []
        for _ in range(n):
            v, i = cbor_item(b, i)
            out.append(v)
        return out, i
    if major == 5:
        out = {}
        for _ in range(n):
            k, i = cbor_item(b, i)
            v, i = cbor_item(b, i)
            out[k] = v
        return out, i
    if major == 6:  # tag (e.g. RFC 8746 typed arrays): keep the tag number
        v, i = cbor_item(b, i)
        return {"__tag__": n, "value": v}, i
    raise ValueError(f"major type {major}")


def factor_shapes(count):
    """Shapes C x H x W (square, common sizes) that hold exactly `count` values."""
    out = []
    for c in (1, 2, 3, 4, 5, 9):
        for side in (20, 40, 64, 80, 160, 320):
            if c * side * side == count:
                out.append(f"{c}x{side}x{side}")
    return out


def describe_bytes(raw, indent):
    n = len(raw)
    print(f"{indent}bytes: {n}")
    for name, fmt, size in (("float32", "<f", 4), ("uint16", "<H", 2), ("uint8", "B", 1)):
        if n % size:
            continue
        count = n // size
        vals = [struct.unpack_from(fmt, raw, k * size)[0] for k in range(count)]
        nonzero = sum(1 for v in vals if v)
        distinct = sorted(set(vals))
        shown = distinct[:8] if len(distinct) <= 8 else distinct[:5] + ["..."] + distinct[-2:]
        print(f"{indent}  as {name}: {count} values, shapes {factor_shapes(count) or '-'}, "
              f"min {min(vals)}, max {max(vals)}, nonzero {nonzero}, "
              f"{len(distinct)} distinct: {shown}")


def describe(value, indent="  "):
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, (bytes, dict, list)):
                print(f"{indent}{k!r}:")
                describe(v, indent + "  ")
            else:
                print(f"{indent}{k!r}: {v!r}")
    elif isinstance(value, list):
        print(f"{indent}array of {len(value)}: {value[:8]!r}")
    elif isinstance(value, bytes):
        describe_bytes(value, indent)
    else:
        print(f"{indent}{value!r}")


def decode_stream(data, max_messages):
    print(f"stream: {len(data)} bytes; first 32: {data[:32].hex(' ')}")
    i, k = 0, 0
    while i + 4 <= len(data) and k < max_messages:
        length = struct.unpack(">I", data[i:i + 4])[0]
        body = data[i + 4:i + 4 + length]
        if length == 0 or length > 64 * 1024 * 1024:
            print(f"at {i}: length prefix {length} looks wrong; not 4-byte-length framing?")
            break
        if len(body) < length:
            print(f"at {i}: message of {length} bytes cut by the end of the capture")
            break
        print(f"message {k}: {length} bytes")
        try:
            value, used = cbor_item(body, 0)
        except (ValueError, IndexError) as exc:
            print(f"  not CBOR ({exc}); first 32 bytes: {body[:32].hex(' ')}")
        else:
            if used != length:
                print(f"  CBOR item uses {used} of {length} bytes")
            describe(value)
        i += 4 + length
        k += 1
    if k:
        print(f"{k} messages decoded")


def capture(port, seconds, max_bytes, out):
    srv = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
    srv.bind(("::", port))
    srv.listen(1)
    print(f"listening on port {port}; start the camera client now")
    conn, addr = srv.accept()
    print(f"client {addr[0]} connected; recording {seconds} s or {max_bytes} bytes")
    conn.settimeout(0.5)
    data = bytearray()
    end = time.monotonic() + seconds
    while time.monotonic() < end and len(data) < max_bytes:
        try:
            chunk = conn.recv(1 << 16)
        except socket.timeout:
            continue
        if not chunk:
            print("client closed the connection")
            break
        data += chunk
    conn.close()
    srv.close()
    with open(out, "wb") as f:
        f.write(data)
    print(f"saved {len(data)} bytes to {out}")
    return bytes(data)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=3333)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--max-bytes", type=int, default=20_000_000)
    ap.add_argument("--out", default="/root/spleat_in.bin")
    ap.add_argument("--messages", type=int, default=3, help="messages to describe")
    ap.add_argument("--decode", metavar="FILE", help="decode a saved capture, no network")
    args = ap.parse_args(argv)
    if args.decode:
        with open(args.decode, "rb") as f:
            data = f.read()
    else:
        data = capture(args.port, args.seconds, args.max_bytes, args.out)
    decode_stream(data, args.messages)
    return 0


if __name__ == "__main__":
    sys.exit(main())
