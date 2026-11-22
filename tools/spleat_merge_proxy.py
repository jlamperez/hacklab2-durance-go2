#!/usr/bin/env python3
"""Adapter between Durance's camera client and the SPLEAT server, on the Kria.

despot_send_and_forget always sends 2 channels (one per polarity), float32 event counts,
2 x H x W. Our ball detector takes 1 channel: 1.0 where a pixel had any event. This proxy
takes despot's place towards the server:

    despot_send_and_forget --TCP 3333--> this proxy --TCP 3334--> backward-server

For each frame it merges the channels (any event -> 1.0), and sends the same message
format on: 4-byte big-endian length + CBOR {"id": n, "payload": float32 bytes}. Whatever
the server sends back on this connection is counted and summarised, and saved with --save.

Board setup (numpy is on Durance's image, Python 3.12):

    sed -i 's/port: 3333/port: 3334/' /root/.config/spleat-server/config.yaml
    systemctl restart spleat-server
    python3 spleat_merge_proxy.py &
    despot_send_and_forget --to_display 0

Format seen on 22 Nov 2026 with spleat_sniff.py: 2x320x320 float32 little-endian, CBOR map
with keys "id" and "payload" only.
"""

import argparse
import socket
import struct
import sys
import threading
import time

import numpy as np


def read_exact(sock, n):
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("connection closed")
        got += k
    return bytes(buf)


def read_message(sock):
    (length,) = struct.unpack(">I", read_exact(sock, 4))
    return read_exact(sock, length)


def cbor_head(major, n):
    if n < 24:
        return bytes([(major << 5) | n])
    for info, size in ((24, 1), (25, 2), (26, 4), (27, 8)):
        if n < (1 << (8 * size)):
            return bytes([(major << 5) | info]) + n.to_bytes(size, "big")
    raise ValueError(n)


def cbor_item(b, i):
    """Small CBOR decoder: unsigned ints, byte and text strings, maps, arrays, floats."""
    major, info = b[i] >> 5, b[i] & 0x1F
    i += 1
    if major == 7:
        if info == 26:
            return struct.unpack(">f", b[i:i + 4])[0], i + 4
        if info == 27:
            return struct.unpack(">d", b[i:i + 8])[0], i + 8
        if info == 25:
            return struct.unpack(">e", b[i:i + 2])[0], i + 2
        return {20: False, 21: True}.get(info), i
    if info < 24:
        n = info
    else:
        size = 1 << (info - 24)
        n = int.from_bytes(b[i:i + size], "big")
        i += size
    if major == 0:
        return n, i
    if major == 1:
        return -1 - n, i
    if major == 2:
        return b[i:i + n], i + n
    if major == 3:
        return b[i:i + n].decode("utf-8", "replace"), i + n
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
    raise ValueError(f"CBOR major type {major}")


def encode_frame(msg_id, payload):
    body = (b"\xa2" + cbor_head(3, 2) + b"id" + cbor_head(0, msg_id)
            + cbor_head(3, 7) + b"payload" + cbor_head(2, len(payload)) + payload)
    return struct.pack(">I", len(body)) + body


def merge(payload, channels, mode):
    frame = np.frombuffer(payload, "<f4").reshape(channels, -1)
    if mode == "any":
        out = (frame.sum(axis=0) > 0).astype("<f4")
    elif mode == "sum":
        out = frame.sum(axis=0, dtype="<f4")
    else:  # "on" / "off": keep one polarity channel only (order not confirmed)
        out = (frame[0 if mode == "ch0" else 1] > 0).astype("<f4")
    return out.tobytes()


def read_replies(sock, stats, save, free):
    try:
        while True:
            body = read_message(sock)
            stats["replies"] += 1
            stats["last_reply"] = time.monotonic()
            free.release()
            if save:
                save.write(struct.pack(">I", len(body)) + body)
            if stats["replies"] <= 3 or stats["replies"] % 100 == 0:
                try:
                    msg, _ = cbor_item(body, 0)
                except (ValueError, IndexError) as exc:
                    print(f"reply {stats['replies']}: {len(body)} bytes, not CBOR ({exc})")
                    continue
                summary = {k: v for k, v in msg.items() if not isinstance(v, bytes)}
                print(f"reply {stats['replies']}: {len(body)} bytes {summary}", flush=True)
                pay = msg.get("payload")
                if isinstance(pay, bytes) and len(pay) % 4 == 0 and pay:
                    vals = np.frombuffer(pay, "<f4")
                    print(f"  payload: {vals.size} float32, min {vals.min():.3g}, "
                          f"max {vals.max():.3g}, sum {vals.sum():.4g}, "
                          f"nonzero {int(np.count_nonzero(vals))}", flush=True)
                    if vals.size % (80 * 80) == 0:
                        per = vals.reshape(-1, 80 * 80)
                        print("  per channel max: "
                              + " ".join(f"{m:.3g}" for m in per.max(axis=1)), flush=True)
    except (ConnectionError, OSError) as exc:
        print(f"server connection ended: {exc}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--listen", type=int, default=3333, help="port the camera client uses")
    ap.add_argument("--server", default="127.0.0.1")
    ap.add_argument("--server-port", type=int, default=3334)
    ap.add_argument("--channels", type=int, default=2, help="channels despot sends")
    ap.add_argument("--mode", choices=("any", "sum", "ch0", "ch1"), default="any",
                    help="any: 1.0 where either channel has an event (what the model saw "
                         "in training); sum: event counts; ch0/ch1: one channel only")
    ap.add_argument("--save", help="append the server's replies to this file")
    ap.add_argument("--max-inflight", type=int, default=1,
                    help="frames sent and not answered yet; camera frames arriving while "
                         "the limit is reached are dropped (the FPGA is slower than despot)")
    args = ap.parse_args(argv)

    up = socket.create_connection((args.server, args.server_port))
    up.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"connected to server {args.server}:{args.server_port}", flush=True)
    stats = {"replies": 0, "last_reply": time.monotonic()}
    save = open(args.save, "ab") if args.save else None
    free = threading.Semaphore(args.max_inflight)
    threading.Thread(target=read_replies, args=(up, stats, save, free), daemon=True).start()

    srv = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
    srv.bind(("::", args.listen))
    srv.listen(1)
    print(f"listening on {args.listen}; start despot_send_and_forget", flush=True)
    cam, addr = srv.accept()
    print(f"camera client {addr[0]} connected", flush=True)

    sent, dropped, t0, last = 0, 0, time.monotonic(), time.monotonic()
    try:
        while True:
            msg, _ = cbor_item(read_message(cam), 0)
            if not free.acquire(blocking=False):
                dropped += 1
                now = time.monotonic()
                if now - last >= 2.0:
                    print(f"{sent} sent, {dropped} dropped, {stats['replies']} replies, "
                          f"last reply {now - stats['last_reply']:.1f} s ago", flush=True)
                    last = now
                continue
            payload = merge(msg["payload"], args.channels, args.mode)
            up.sendall(encode_frame(msg.get("id", sent), payload))
            sent += 1
            if sent == 1:
                print(f"first frame: {len(msg['payload'])} bytes in, {len(payload)} out", flush=True)
            now = time.monotonic()
            if now - last >= 2.0:
                print(f"{sent} sent ({sent / (now - t0):.1f}/s), {dropped} dropped, "
                      f"{stats['replies']} replies", flush=True)
                last = now
    except (ConnectionError, KeyboardInterrupt) as exc:
        print(f"stopped: {exc!r}; {sent} sent, {dropped} dropped, {stats['replies']} replies")
    finally:
        if save:
            save.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
